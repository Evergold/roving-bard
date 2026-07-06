#!/usr/bin/env python3
import os
import sys
import subprocess

def print_help():
    print("Roving Bard Developer Tool")
    print("Usage: ./dev_tool.py [command]")
    print("\nAvailable commands:")
    print("  lotro-words [locale] Run the LOTRO location wordlist extractor skill (choices: EN, DE, FR; default: EN)")
    print("  stride-lint          Run the STRIDE security threat linter skill")
    print("  test                 Run all backend unit tests")

def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
        
    cmd = sys.argv[1].lower()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    app_cwd = os.path.join(script_dir, "roving-bard")
    
    if cmd == "lotro-words":
        script_path = os.path.join(script_dir, ".agents", "skills", "lotro-words", "scripts", "extract_words.py")
        subprocess.run(["uv", "run", "--no-sync", "python", script_path] + sys.argv[2:], cwd=app_cwd)
    elif cmd == "stride-lint":
        script_path = os.path.join(script_dir, ".agents", "skills", "stride-linting", "scripts", "run_stride.py")
        subprocess.run(["uv", "run", "--no-sync", "python", script_path] + sys.argv[2:], cwd=app_cwd)
    elif cmd == "test":
        subprocess.run(["uv", "run", "--no-sync", "pytest", "tests/unit/"], cwd=app_cwd)
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
