"""Reproducible CUDA training and validation; official test remains separate."""
from __future__ import annotations

import argparse
import contextlib
import json
import hashlib
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import LineDataset, WidthBatchSampler, collate_lines, load_characters, assert_ctc_feasible
from model import CRNN, greedy_decode
from metrics import Metrics


def environment(device):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_gb": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 3) if device.type == "cuda" else None,
            "platform": platform.platform()}


def make_loader(dataset, batch_size, shuffle, workers, seed=42):
    return DataLoader(dataset, batch_sampler=WidthBatchSampler(dataset.widths, batch_size, shuffle, seed),
                      collate_fn=collate_lines, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


@torch.inference_mode()
def evaluate(model, loader, characters, device, amp=False, predictions_file=None):
    model.eval()
    metrics = Metrics()
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            log_probs, lengths = model(batch["images"].to(device, non_blocking=True), batch["widths"])
        decoded = greedy_decode(log_probs, lengths, characters)
        for sample_id, reference, prediction in zip(batch["ids"], batch["texts"], decoded):
            metrics.update(reference, prediction)
            if predictions_file:
                predictions.append({"id": sample_id, "reference": reference, "prediction": prediction})
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = metrics.compute()
    result.update({"seconds": elapsed, "lines_per_second": len(loader.dataset) / elapsed})
    if predictions_file:
        with open(predictions_file, "w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, best_cer, config, characters):
    temp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "best_cer": best_cer, "config": config,
                "characters": characters,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, temp)
    os.replace(temp, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Steps per epoch; zero means all")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.max_width % 4 or args.height < 16 or args.batch_size < 1:
        raise ValueError("Use --max-width divisible by 4, height >=16, batch-size >=1")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install a compatible CUDA PyTorch build.")
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    characters = load_characters(args.data_dir / "vocab.json")
    train_set = LineDataset(args.data_dir / "train.jsonl", characters, args.height, args.max_width, args.max_train_samples)
    val_set = LineDataset(args.data_dir / "val.jsonl", characters, args.height, args.max_width, args.max_val_samples)
    if not train_set or not val_set:
        raise ValueError("Both training and validation splits must be nonempty")
    for row, width in zip(train_set.rows, train_set.widths):
        if any(char not in characters for char in row["text"]):
            raise ValueError(f"Training OOV in {row['id']}")
        assert_ctc_feasible([row["text"]], [width])
    train_loader = make_loader(train_set, args.batch_size, True, args.workers, args.seed)
    val_loader = make_loader(val_set, args.batch_size, False, args.workers, args.seed)
    model = CRNN(len(characters) + 1, hidden_size=args.hidden_size, channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"environment": environment(device), "train_samples": len(train_set),
                   "validation_samples": len(val_set), "parameters": sum(p.numel() for p in model.parameters()),
                   "amp": amp, "normalization": "NFC + collapse whitespace; logical Unicode targets",
                   "ctc_blank": 0, "reading_direction": "reverse valid CNN time axis before BiLSTM",
                   "manifest_sha256": {name: hashlib.sha256((args.data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                                       for name in ("train", "val")}})
    start_epoch, best_cer = 1, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["characters"] != characters:
            raise ValueError("Resume vocabulary differs")
        for name in ("height", "max_width", "hidden_size", "channels", "seed"):
            if checkpoint["config"][name] != config[name]:
                raise ValueError(f"Resume configuration differs: {name}")
        if checkpoint["config"].get("manifest_sha256") != config["manifest_sha256"]:
            raise ValueError("Resume manifests differ")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_cer = checkpoint["epoch"] + 1, checkpoint["best_cer"]
        state = checkpoint.get("rng")
        if state:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())
            if device.type == "cuda" and state["cuda"]:
                torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
        train_loader.batch_sampler.epoch = start_epoch - 1
    (args.run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False), flush=True)
    history_path = args.run_dir / "history.jsonl"
    if start_epoch == 1:
        baseline = evaluate(model, val_loader, characters, device, amp)
        baseline.update({"epoch": 0, "stage": "untrained_validation"})
        history_path.write_text(json.dumps(baseline) + "\n", encoding="utf-8")
        print(json.dumps(baseline), flush=True)
    overall_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        losses, n_samples, steps, grad_norm = 0.0, 0, 0, 0.0
        for step, batch in enumerate(train_loader, 1):
            if args.max_steps and step > args.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                log_probs, lengths = model(images, batch["widths"])
            # CTC always float32, including when CNN/LSTM use mixed precision.
            loss = criterion(log_probs.float(), targets, lengths, batch["target_lengths"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite CTC loss, epoch {epoch} step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                    print(json.dumps({"epoch": epoch, "step": step, "event": "AMP overflow; skipped step"}), flush=True)
                    continue
                raise RuntimeError("Nonfinite gradients")
            scaler.step(optimizer)
            scaler.update()
            size = len(batch["ids"])
            losses += float(loss.detach()) * size
            n_samples += size
            steps += 1
            grad_norm = float(norm)
            if step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps_total": len(train_loader),
                                  "ctc_loss": round(losses / n_samples, 4),
                                  "elapsed_seconds": round(time.perf_counter() - epoch_start, 1)}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if not n_samples:
            raise RuntimeError("No optimizer steps completed")
        train_seconds = time.perf_counter() - epoch_start
        validation = evaluate(model, val_loader, characters, device, amp)
        record = {"epoch": epoch, "train_ctc_loss": losses / n_samples, "train_samples": n_samples,
                  "optimizer_steps": steps, "train_seconds": train_seconds,
                  "train_lines_per_second": n_samples / train_seconds,
                  "validation": validation, "last_gradient_norm": grad_norm,
                  "lr": optimizer.param_groups[0]["lr"],
                  "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                  "elapsed_seconds": time.perf_counter() - overall_start}
        scheduler.step(validation["cer"])
        improved = validation["cer"] < best_cer
        best_cer = min(best_cer, validation["cer"])
        save_checkpoint(args.run_dir / "last.pt", model, optimizer, scaler, scheduler, epoch, best_cer, config, characters)
        if improved:
            save_checkpoint(args.run_dir / "best.pt", model, optimizer, scaler, scheduler, epoch, best_cer, config, characters)
        with open(history_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    print(json.dumps({"finished": True, "best_validation_cer": best_cer,
                      "best_checkpoint": str(args.run_dir / 'best.pt')}), flush=True)


if __name__ == "__main__":
    main()
