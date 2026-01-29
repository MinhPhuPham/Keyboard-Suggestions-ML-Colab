# Resources Directory

This directory should contain the following files for the package to work:

## Required Files

| File | Purpose | Export From |
|------|---------|-------------|
| `gru_keyboard_ios.mlmodelc` | CoreML model | Notebook Cell 10 |
| `word_to_index.json` | Tokenization | mobile_export_cell12.py |
| `vocab_high.json` | High-freq words | mobile_export_cell12.py |
| `vocab_medium.json` | Medium-freq words | mobile_export_cell12.py |
| `vocab_low.json` | Low-freq words | mobile_export_cell12.py |
| `prefix_index.json` | Prefix lookup | mobile_export_cell12.py |
| `soundex_index.json` | Phonetic codes | mobile_export_cell12.py |
| `keyboard_adjacent.json` | Key proximity | mobile_export_cell12.py |

## How to Add

1. Run the training notebook to export models
2. Run `mobile_export_cell12.py` to generate JSON files
3. Copy files to this Resources directory
4. Build the package - resources will be bundled automatically
