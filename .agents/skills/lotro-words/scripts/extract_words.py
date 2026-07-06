#!/usr/bin/env python3
import os
import sys

# Add app to path to import extract_lotro_words
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
sys.path.insert(0, os.path.join(workspace_root, "roving-bard"))

from app.player import extract_lotro_words

def get_locale_dat_path():
    # Check roving-bard/app/locales/client_local_English.dat
    path = os.path.join(workspace_root, "roving-bard", "app", "locales", "client_local_English.dat")
    if os.path.exists(path):
        return path
    return None

def main():
    print("====================================================")
    print("            LOTRO Wordlist Extractor Skill          ")
    print("====================================================")
    
    locale_path = get_locale_dat_path()
    if locale_path:
        print(f"Found client_local_English.dat in app locales directory:\n  {locale_path}\n")
        use_locale = input("Use this file? [Y/n]: ").strip().lower()
        if use_locale in ("", "y", "yes"):
            dat_path = locale_path
        else:
            install_dir = input("Enter LOTRO install directory or path to client_local_English.dat: ").strip()
            dat_path = install_dir
            if os.path.isdir(install_dir):
                dat_path = os.path.join(install_dir, "client_local_English.dat")
    else:
        install_dir = input("Enter LOTRO install directory or path to client_local_English.dat: ").strip()
        dat_path = install_dir
        if os.path.isdir(install_dir):
            dat_path = os.path.join(install_dir, "client_local_English.dat")
        
    if not os.path.exists(dat_path):
        print(f"Error: File not found at {dat_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loaded locations from English locale at: {dat_path}")
    
    app_dir = os.path.join(workspace_root, "roving-bard", "app")
    default_words_path = os.path.join(app_dir, "lotro_words.txt")
    output_path = os.path.join(app_dir, "lotro_words-EN.txt")
    
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
