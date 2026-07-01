# Roving Bard: Adaptive, Location-Aware Audio for LOTRO

Roving Bard is a game-aware music player agent that captures the screen, recognises the in-game location via OCR (with local VLM and Gemini Vision fallback), and seamlessly transitions background music to match the active region. It is built on the Google Agent Development Kit (ADK) and includes a single-page browser GUI.

---

## 📁 Project Structure

```
capstone/
├── README.md                     # This documentation
├── threat_model.md               # STRIDE threat model assessment
├── roving-bard/                  # Core application directory
│   ├── app/                      # Backend FastAPI & Agent logic
│   │   ├── agent.py              #   ADK ReAct agent definition
│   │   ├── tools.py              #   Agent tools (screen check, playback, Gemini Vision)
│   │   ├── player.py             #   SafeMusicPlayer, ScreenGrabber, LocalOCRParser, TrackMapper
│   │   ├── fast_api_app.py       #   FastAPI server & REST endpoints
│   │   ├── gui.html              #   Dashboard UI (HTML5/Vanilla CSS/JS)
│   │   ├── lotro_words.txt       #   OCR dictionary (229 pre-populated locations)
│   │   └── app_utils/            #   Shared utilities (telemetry, typing)
│   ├── audio/                    # Audio library & mapping config
│   │   ├── .cache/               #   Synthesized FLAC files
│   │   ├── mapping.yaml          #   Location → track mapping rules & config
│   │   └── MuseScore_General.sf3 #   Bundled SoundFont
│   ├── capture/                  # Screen capture staging directory
│   ├── tests/                    # Unit, integration, and eval tests
│   ├── run_player.py             # Standalone CLI player loop
│   ├── pyproject.toml            # Python packaging and dependencies
│   ├── uv.lock                   # Lockfile
│   └── AGENTS.md                 # Operational agent development rules
```

---

## ✨ Features

- **Automatic Minimap Scanning**: Captures the game screen, crops the minimap, and parses coordinates + location names using Tesseract OCR.
- **VLM Preprocessing & 4x Scaling**: Resizes the raw location crop by **4x (Lanczos)** before passing it to Vision-Language Models (VLMs) to ensure high-contrast character transcription.
- **Vision-Language Model Fallbacks**: If local Tesseract OCR is inconclusive, queries a VLM (Gemini 2.5 Flash Lite, Florence-2, Moondream, Qwen2-VL, etc.).
- **VLM Warmup & Unload REST Endpoints**: Warm up models in the background or trigger immediate unloads to manage GPU VRAM.
- **On-Demand SoundFont Downloader**: Download the uncompressed `MuseScore_General.sf2` (ULTRA) SoundFont directly from the Preferences menu, with background thread updates.
- **10-Band Parametric EQ**: Adjust playback frequencies (32 Hz – 16 kHz) in real-time with scipy-powered IIR peaking filters.
- **Audio Segments**: Save, edit, and play custom loops/slices of files with dedicated EQ and volume profiles.

---

## 🛠️ Requirements

### Python Environment
- **Python** ≥ 3.11, < 3.14
- **uv** — Fast Python package manager — [Install](https://docs.astral.sh/uv/getting-started/installation/)
- **google-agents-cli** — CLI for ADK agents — Install with `uv tool install google-agents-cli`

### System Dependencies
- **Tesseract OCR**: Local OCR for minimap text (`sudo apt install tesseract-ocr`)
- **SDL2 + SDL_mixer**: Pygame audio output (`sudo apt install libsdl2-mixer-2.0-0` if not bundled)
- **libsndfile**: For soundfile EQ processing (`sudo apt install libsndfile1`)

### Core Python Dependencies (managed via `pyproject.toml`)
- `google-adk[gcp]` — Google Agent Development Kit & Cloud integrations
- `pygame` — Audio output & player controls
- `pytesseract` & `opencv-python` — Screen OCR and screenshot processing
- `scipy` & `soundfile` — Parametric EQ processing
- `litellm` — Fallback vision-language model API calls
- `torch` & `transformers` — Local VLM execution (Florence-2)

---

## 🚀 Quick Start

### 1. Install Dependencies
Run from the `roving-bard` directory:
```bash
agents-cli install
```

### 2. Start the FastAPI Web Server
To start the server with auto-reloading enabled for development:
```bash
uv run uvicorn app.fast_api_app:app --reload
```
The server will boot on `http://127.0.0.1:8000`.

### 3. Open the GUI Dashboard
Once the server is running, navigate to:
```
http://localhost:8000/gui
```
* **Auto-Scanning**: Turn on the **Auto Scanning** toggle in the status header to begin scanning your screen at the configured interval.
* **Localhost Access**: Loopback connections from localhost bypass API key authentication and display a green checkmark next to the status badge.

### 4. Alternative: Standalone CLI Player Loop
If you prefer running a command-line polling loop without the GUI:
```bash
uv run python run_player.py
```
This loop runs independently, scanning the screen and playing music directly in the terminal.

---

## ⚙️ Configuration (`audio/mapping.yaml`)

Edit the rules for mapping locations or coordinates to track files:
```yaml
active_soundfont: "MuseScore_General.sf3"
polling_interval: 2.0
minimap_bounds:
  x: 0.8          # 80% from left
  y: 0.05         # 5% from top
  width: 0.15     # 15% of screen width
  height: 0.15    # 15% of screen height
mappings:
  - location_name: "Town"
    track_file: "town.wav"
  - ns_min: 10.0  # Coordinate range match
    ns_max: 20.0
    ew_min: -80.0
    ew_max: -60.0
    track_file: "cave.wav"
```

---

## ⌨️ Development Commands

| Command | Purpose |
|---|---|
| `uv run uvicorn app.fast_api_app:app --reload` | Run the development API and GUI server |
| `agents-cli playground` | Launch interactive ADK development loop |
| `uv run pytest` | Run unit and integration tests |
| `agents-cli lint` | Run code quality checks (ruff, ty, codespell) |
| `agents-cli deploy` | Deploy the server to Cloud dev environment |

---

## 🔒 Security & Threat Modeling

A comprehensive **STRIDE Threat Modeling Assessment** is maintained at [threat_model.md](file:///home/chuubi/Desktop/vibe-coding-2026/capstone/threat_model.md) in the project root. Be sure to review boundaries and data sanitization guidelines before modifying endpoints.
