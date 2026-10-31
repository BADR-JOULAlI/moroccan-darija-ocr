# Moroccan Darija OCR

CNN + BiLSTM + CTC OCR pipeline for Moroccan Arabic line images, with a trained Arabic baseline and an explicit path for Darija adaptation.

The current baseline was trained on the Moroccan Official Gazette and tested on 1,212 held-out lines. The project reached 1.49% CER and 6.94% WER on that Arabic printed-text test. These numbers do not represent Darija performance.

See the complete setup, training commands, dataset audit, and adaptation instructions in [arabic_ocr/README.md](arabic_ocr/README.md).

The repository intentionally excludes raw datasets and AtlasOCR's large Parquet shards. AtlasOCR inspection found page/paragraph images without published line-level boxes, so the adaptation tools refuse unverified multiline supervision.
