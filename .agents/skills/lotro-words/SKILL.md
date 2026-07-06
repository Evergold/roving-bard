---
name: lotro-words
description: Generates or updates the lotro_words.txt dictionary of location names by directly parsing the LOTRO game client's client_local_English.dat file.
---

# LOTRO Custom Wordlist Extractor Skill

This skill allows Roving Bard to generate or update the custom location vocabulary dictionary directly from a player's local *The Lord of the Rings Online* (LOTRO) installation files. This ensures the locations are 100% official and match the strings exactly as they appear in the player's install.

## How it works
1. **Self-Calibration**: The extraction script scans the game client's main localization archive (`client_local_English.dat`) for known starter locations (e.g. `Tinnudir`, `Celondim`, `Bree`).
2. **Identification**: It dynamically identifies the internal database table/string blocks that house location names by checking match density.
3. **Extraction**: Once identified, it parses and decompresses all English location strings from those tables, supporting both UTF-8 and UTF-16-LE character encodings.
4. **Integration**: It isolates new location words not present in the default list and saves them to `app/lotro_words-EN.txt`.

## How to use
Run the extraction script from your terminal:
```bash
python3 .agents/skills/lotro-words/scripts/extract_words.py
```
When prompted, provide either your main LOTRO installation directory or the direct absolute path to `client_local_English.dat`.

## Merged Loading
Upon Roving Bard backend launch, the system automatically checks for the presence of `app/lotro_words-EN.txt` and merges its words with `app/lotro_words.txt` dynamically in-memory for all OCR/VLM spell-checking steps.
