---
name: lotro-words
description: Generates or updates the custom dictionary of location names by directly parsing the LOTRO game client's localization files (English, German, or French).
---

# LOTRO Custom Wordlist Extractor Skill

This skill allows Roving Bard to generate or update the custom location vocabulary dictionary directly from a player's local *The Lord of the Rings Online* (LOTRO) installation files. This ensures the locations are 100% official and match the strings exactly as they appear in the player's install.

## How it works
1. **Self-Calibration**: The extraction script scans the game client's main localization archive (`client_local_English.dat`, `client_local_DE.dat`, or `client_local_FR.dat`) for known starter locations (e.g. `Tinnudir`, `Celondim`, `Bree`).
2. **Identification**: It dynamically identifies the internal database table/string blocks that house location names by checking match density.
3. **Extraction**: Once identified, it parses and decompresses all localized location strings from those tables, supporting both UTF-8 and UTF-16-LE character encodings.
4. **Integration**: It isolates new location words not present in the default list and saves them to a locale-specific file (e.g., `app/lotro_words-EN.txt`, `app/lotro_words-DE.txt`, or `app/lotro_words-FR.txt`).

## How to use
Run the extraction script from your terminal:
```bash
python3 .agents/skills/lotro-words/scripts/extract_words.py
```
When prompted, select the language locale found in your app locales directory or provide either your main LOTRO installation directory or the direct absolute path to the localization DAT file (e.g., `client_local_DE.dat`).

## Merged Loading
Upon Roving Bard backend launch, the system automatically checks for the presence of the active locale-specific wordlist (checking `lotro_words-EN.txt`, `lotro_words-DE.txt`, or `lotro_words-FR.txt` in order) and merges its words with `app/lotro_words.txt` dynamically in-memory for all OCR/VLM spell-checking steps.
