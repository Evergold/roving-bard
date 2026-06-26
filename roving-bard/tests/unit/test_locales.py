import json
import os

def test_locales_key_parity():
    locales_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app", "locales")
    
    # Check that locales directory exists
    assert os.path.exists(locales_dir), f"Locales directory not found at: {locales_dir}"
    
    # Load master locale en-US.json
    en_us_path = os.path.join(locales_dir, "en-US.json")
    assert os.path.exists(en_us_path), "en-US.json master locale not found"
    
    with open(en_us_path, "r", encoding="utf-8") as f:
        en_us_dict = json.load(f)
        
    en_us_keys = set(en_us_dict.keys())
    
    # Find all other locale json files
    locale_files = [f for f in os.listdir(locales_dir) if f.endswith(".json") and f != "en-US.json"]
    
    errors = []
    
    for filename in locale_files:
        filepath = os.path.join(locales_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lang_dict = json.load(f)
        except Exception as e:
            errors.append(f"Failed to parse {filename}: {str(e)}")
            continue
            
        lang_keys = set(lang_dict.keys())
        
        # Check for missing keys
        missing_keys = en_us_keys - lang_keys
        if missing_keys:
            errors.append(f"{filename} is missing keys: {sorted(list(missing_keys))}")
            
        # Check for extra keys
        extra_keys = lang_keys - en_us_keys
        if extra_keys:
            errors.append(f"{filename} has unexpected extra keys: {sorted(list(extra_keys))}")
            
    assert not errors, "Locale validation failed:\n" + "\n".join(errors)
