#!/usr/bin/env python3
import os
import sys

# Add app to path to import extract_lotro_words
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
sys.path.insert(0, os.path.join(workspace_root, "roving-bard"))

from app.player import extract_lotro_words

def get_available_locales():
    locales = []
    # Check roving-bard/app/locales/
    locales_dir = os.path.join(workspace_root, "roving-bard", "app", "locales")
    if os.path.exists(locales_dir):
        for name, lang, code in [
            ("client_local_English.dat", "English", "EN"),
            ("client_local_DE.dat", "German", "DE"),
            ("client_local_FR.dat", "French", "FR")
        ]:
            path = os.path.join(locales_dir, name)
            if os.path.exists(path):
                locales.append((name, lang, code, path))
    return locales

def prompt_manual_input():
    path = input("Enter LOTRO install directory or path to localization DAT file (e.g. client_local_DE.dat): ").strip()
    if os.path.isdir(path):
        for name, lang, code in [
            ("client_local_English.dat", "English", "EN"),
            ("client_local_DE.dat", "German", "DE"),
            ("client_local_FR.dat", "French", "FR")
        ]:
            check_path = os.path.join(path, name)
            if os.path.exists(check_path):
                return check_path, code, lang
        return os.path.join(path, "client_local_English.dat"), "EN", "English"
    else:
        base = os.path.basename(path).lower()
        if "de" in base:
            return path, "DE", "German"
        elif "fr" in base:
            return path, "FR", "French"
        else:
            return path, "EN", "English"

def main():
    print("====================================================")
    print("            LOTRO Wordlist Extractor Skill          ")
    print("====================================================")
    
    locales = get_available_locales()
    if locales:
        print("Found the following game data files:")
        for idx, (name, lang, code, path) in enumerate(locales):
            print(f"  [{idx + 1}] {name} ({lang} Locale)")
        print()
        
        if len(locales) == 1:
            name, lang, code, path = locales[0]
            use_locale = input(f"Use {name}? [Y/n]: ").strip().lower()
            if use_locale in ("", "y", "yes"):
                dat_path = path
                lang_code = code
                lang_name = lang
            else:
                dat_path, lang_code, lang_name = prompt_manual_input()
        else:
            selection = input(f"Select a file to parse (1-{len(locales)}) or press Enter for [1]: ").strip()
            if not selection:
                idx = 0
            else:
                try:
                    idx = int(selection) - 1
                    if idx < 0 or idx >= len(locales):
                        idx = 0
                except ValueError:
                    idx = 0
            name, lang, code, path = locales[idx]
            dat_path = path
            lang_code = code
            lang_name = lang
    else:
        dat_path, lang_code, lang_name = prompt_manual_input()
        
    if not os.path.exists(dat_path):
        print(f"Error: File not found at {dat_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loaded locations from {lang_name} locale at: {dat_path}")
    
    app_dir = os.path.join(workspace_root, "roving-bard", "app")
    default_words_path = os.path.join(app_dir, "lotro_words.txt")
    output_path = os.path.join(app_dir, f"lotro_words-{lang_code}.txt")
    
    raw_extracted = set()
    try:
        raw_extracted = extract_lotro_words(dat_path, default_words_path, output_path)
        print("Save completed successfully.")
    except Exception as e:
        print(f"Error parsing/saving location wordlist: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Print simple differences log
    default_words = set()
    if os.path.exists(default_words_path):
        with open(default_words_path, 'r', encoding='utf-8') as f:
            default_words.update(line.strip() for line in f if line.strip())
            
    extracted_words = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            extracted_words.update(line.strip() for line in f if line.strip())
            
    new_words = sorted(raw_extracted - default_words)
    
    known_instruments = {
        "Bagpipe", "Bassoon", "Clarinet", "Drum", "Fiddle", "Flute", "Harp", "Horn", 
        "Lute", "Pibgorn", "Bagpipes", "Drums", "Fiddles", "Flutes", "Harps", "Horns", "Lutes"
    }
    default_locations = {w for w in default_words if w not in known_instruments}
    unmatched_locations = sorted(default_locations - raw_extracted)
    
    print("\n====================================================")
    print("                DIFFERENCES LOG                    ")
    print("====================================================")
    print(f"Total default words: {len(default_words)}")
    print(f"Total extracted locations: {len(extracted_words)}")
    print(f"New locations added: {len(new_words)}")
    
    if unmatched_locations:
        print(f"\n[WARNING] {len(unmatched_locations)} default location(s) were not matched in the game data:")
        for loc in unmatched_locations[:15]:
            print(f"  - {loc}")
        if len(unmatched_locations) > 15:
            print(f"  ... and {len(unmatched_locations) - 15} more.")
            
    if new_words:
        print("\nFirst 15 new location entries:")
        for w in new_words[:15]:
            print(f"  + {w}")
    print("====================================================")

if __name__ == "__main__":
    main()
