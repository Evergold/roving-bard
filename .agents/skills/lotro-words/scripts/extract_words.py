#!/usr/bin/env python3
import os
import sys

# Add app to path to import extract_lotro_words
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
sys.path.insert(0, os.path.join(workspace_root, "roving-bard"))

from app.player import extract_lotro_words

def get_locale_dat_path(filename):
    # Check roving-bard/app/locales/
    path = os.path.join(workspace_root, "roving-bard", "app", "locales", filename)
    if os.path.exists(path):
        return path
    return None

def prompt_manual_input(filename):
    path = input(f"Enter LOTRO install directory or path to {filename}: ").strip()
    if os.path.isdir(path):
        return os.path.join(path, filename)
    return path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LOTRO Wordlist Extractor")
    # Case-insensitive choices check via custom validator
    def locale_type(s):
        s_upper = s.upper()
        if s_upper not in ("EN", "DE", "FR"):
            raise argparse.ArgumentTypeError(f"Invalid locale choice: {s} (choose from EN, DE, FR)")
        return s_upper
        
    parser.add_argument("locale", nargs="?", default="EN", type=locale_type, help="Locale to extract (EN, DE, FR; default: EN)")
    args = parser.parse_args()
    
    lang_code = args.locale
    locale_map = {
        "EN": ("client_local_English.dat", "English"),
        "DE": ("client_local_DE.dat", "German"),
        "FR": ("client_local_FR.dat", "French")
    }
    filename, lang_name = locale_map[lang_code]
    
    print("====================================================")
    print("            LOTRO Wordlist Extractor Skill          ")
    print("====================================================")
    print(f"Targeting Locale: {lang_name} ({lang_code})")
    print()
    
    locale_path = get_locale_dat_path(filename)
    if locale_path:
        print(f"Found {filename} in app locales directory:\n  {locale_path}\n")
        use_locale = input("Use this file? [Y/n]: ").strip().lower()
        if use_locale in ("", "y", "yes"):
            dat_path = locale_path
        else:
            dat_path = prompt_manual_input(filename)
    else:
        dat_path = prompt_manual_input(filename)
        
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
