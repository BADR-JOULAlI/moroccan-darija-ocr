# Moroccan Official Gazette (Bulletin Officiel) — Transcribed Arabic Text-Line Images

A corpus of **12,890 Arabic text-line images** cropped from scanned issues of the
Moroccan Official Gazette (Bulletin Officiel), each paired with a manually verified
ground-truth transcription. The corpus targets research on Arabic printed-text
recognition (OCR) under real-world degradation: scan quality ranges from sharp
print to heavily blurred reproductions.

## Contents

| File | Description |
|---|---|
| `images.zip` | 12,890 grayscale line-crop images (PNG). Filenames match the `image` field of the JSONL files. |
| `merged.jsonl` | All 12,890 transcribed lines, one JSON object per line (UTF-8). |
| `real_holdout_test.jsonl` | Predefined held-out **test split of 1,212 lines**. |
| `checksums.txt` | SHA-256 checksums of all files. |

## Record format

Each line of the JSONL files is a JSON object:

```json
{"image": "<filename>.png", "text": "<ground-truth transcription>"}
```

- `text` is Unicode Arabic in **logical (memory) order**, unnormalized.
  Consumers training CTC/attention recognizers on visual frame order should apply
  their own logical-to-display conversion (e.g., the Unicode Bidirectional
  Algorithm) if their pipeline requires it.
- Character inventory: Arabic letters, digits, punctuation, and space.
  The full corpus contains 881,683 characters (94 unique symbols); the test
  split contains 87,812 characters. The rarest symbols of the corpus occur in
  the training portion only.

## Suggested protocol

`real_holdout_test.jsonl` is the evaluation split; its 1,212 filenames also appear
in `merged.jsonl`. For training, **exclude every filename present in the test
split from `merged.jsonl`** (leakage guard), which leaves 11,678 lines for
training/validation. Keeping this split fixed makes results comparable across
studies using the corpus.

## Provenance

Line images were cropped from publicly available scanned issues of the Bulletin
Officiel (the official gazette of the Kingdom of Morocco). Transcriptions were
produced and verified manually. The Bulletin Officiel is an official public
publication; this record redistributes derived line crops for research purposes.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

If you use this corpus, please cite this Zenodo record (see the DOI on this page).

## Version

1.0.0 — initial public release.
