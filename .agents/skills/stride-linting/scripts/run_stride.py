#!/usr/bin/env python3
import os
import re
import sys

# Color formatting helpers
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_result(pillar, check_name, status, details):
    status_str = ""
    if status == "PASS":
        status_str = f"{GREEN}[PASS]{RESET}"
    elif status == "WARN":
        status_str = f"{YELLOW}[WARN]{RESET}"
    else:
        status_str = f"{RED}[FAIL]{RESET}"
    print(f"[{pillar:<10}] {check_name:<40} {status_str} - {details}")

def main():
    print(f"{BOLD}===================================================={RESET}")
    print(f"{BOLD}          STRIDE Security Linter Dev-Tool           {RESET}")
    print(f"{BOLD}===================================================={RESET}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    app_dir = os.path.join(workspace_root, "roving-bard", "app")

    fast_api_path = os.path.join(app_dir, "fast_api_app.py")
    player_path = os.path.join(app_dir, "player.py")
    gui_path = os.path.join(app_dir, "gui.html")

    if not os.path.exists(fast_api_path) or not os.path.exists(player_path):
        print(f"{RED}Error: Roving Bard codebase not found.{RESET}", file=sys.stderr)
        sys.exit(1)

    with open(fast_api_path, 'r', encoding='utf-8') as f:
        fast_api_code = f.read()

    with open(player_path, 'r', encoding='utf-8') as f:
        player_code = f.read()

    gui_code = ""
    if os.path.exists(gui_path):
        with open(gui_path, 'r', encoding='utf-8') as f:
            gui_code = f.read()

    # --- 1. Spoofing Checks ---
    if "verify_api_key" in fast_api_code or "APIKeyHeader" in fast_api_code:
        print_result("Spoofing", "API key dependency check", "PASS", "FastAPI verification dependency found.")
    else:
        print_result("Spoofing", "API key dependency check", "FAIL", "No API key headers or dependency found in routes.")

    # --- 2. Tampering Checks ---
    if "os.path.basename" in player_code or "os.path.basename" in fast_api_code:
        print_result("Tampering", "Path traversal checks in audio endpoints", "PASS", "Basename isolation/sanitization detected.")
    else:
        print_result("Tampering", "Path traversal checks in audio endpoints", "WARN", "Verify if os.path.basename is enforced on raw file inputs.")

    # --- 3. Repudiation Checks ---
    if "print(" in fast_api_code or "logger" in fast_api_code:
        print_result("Repudiation", "Action logging in API endpoints", "PASS", "Server diagnostic logging found.")
    else:
        print_result("Repudiation", "Action logging in API endpoints", "WARN", "No standard stdout print or logging statement found.")

    # --- 4. Information Disclosure Checks ---
    if gui_code:
        sensitive_keys = re.findall(r"(?:AGENT_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*['\"].+['\"]", gui_code)
        if sensitive_keys:
            print_result("Info Leak", "Hardcoded API keys in frontend", "FAIL", f"Found hardcoded keys in gui.html: {sensitive_keys}")
        else:
            print_result("Info Leak", "Hardcoded API keys in frontend", "PASS", "No hardcoded keys found in gui.html.")
    else:
        print_result("Info Leak", "Frontend checks", "WARN", "gui.html not found.")

    # --- 5. Denial of Service Checks ---
    if "content-length" in fast_api_code.lower() or "max_file_size" in fast_api_code.lower() or "size" in fast_api_code.lower():
        print_result("DoS", "Max file size constraints on uploads", "PASS", "File size boundary checks found.")
    else:
        print_result("DoS", "Max file size constraints on uploads", "WARN", "Add file size limits to prevent disk exhaustion.")

    # --- 6. Elevation of Privilege Checks ---
    if "127.0.0.1" in fast_api_code or "::1" in fast_api_code:
        print_result("Privilege", "Loopback auto-authentication bypass", "WARN", "Server contains automatic authentication bypass for local loopback IPs.")
    else:
        print_result("Privilege", "Loopback bypass check", "PASS", "No hardcoded loopback IP authorization bypasses detected.")

    print(f"{BOLD}===================================================={RESET}")

if __name__ == "__main__":
    main()
