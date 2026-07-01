# Roving Bard: Adaptive, Location-Aware Audio for LOTRO

A game-aware music player agent that captures the screen, recognises the in-game
location via OCR (with Gemini Vision fallback), and seamlessly transitions
background music to match the environment. Built on Google ADK with a
full-featured browser GUI.

## Project Structure

```
roving-bard/
├── app/                          # Core application code
│   ├── __init__.py               #   Package init & metadata
│   ├── agent.py                  #   ADK ReAct agent definition
│   ├── tools.py                  #   Agent tools (screen check, playback, Gemini Vision)
│   ├── player.py                 #   SafeMusicPlayer, ScreenGrabber, LocalOCRParser, TrackMapper
│   ├── fast_api_app.py           #   FastAPI server & REST endpoints
│   ├── gui.html                  #   Browser dashboard (262 KB single-file app)
│   ├── lotro_words.txt           #   OCR dictionary (229 pre-populated locations)
│   └── app_utils/                #   Shared utilities
│       ├── telemetry.py          #     OpenTelemetry / Cloud Trace setup
│       └── typing.py             #     Shared type definitions
│
├── audio/                        #  Audio library directory
│   ├── .cache/                   #   Synthesized FLAC files (auto-generated & cleared)
│   ├── .gitkeep                  #   Keeps directory in version control
│   ├── mapping.yaml              #   Location → track mapping rules & config
│   ├── segments.yaml             #   Saved segment definitions (git-ignored)
│   ├── file_tags.yaml            #   Per-file tag metadata (git-ignored)
│   ├── MuseScore_General.sf3     #   Pre-packaged FOSS HQ SoundFont (CC-BY 3.0)
│   ├── MuseScore_General_LICENSE.txt # SoundFont license attribution
│   ├── *.wav / *.mp3 / *.ogg     #   Audio tracks (git-ignored)
│   ├── *.flac / *.mp4            #   Audio tracks (git-ignored)
│   ├── *.abc                     #   ABC notation files (auto-converted to MIDI)
│   └── *.sf2 / *.sf3             #   Custom SoundFont files for MIDI playback
│
├── capture/                      #  Screen capture staging directory
│   └── .gitkeep                  #   Also accepts manually placed screenshots
│
├── tests/                        #  Test suite
│   ├── conftest.py               #   Shared pytest fixtures
│   ├── unit/                     #   Unit tests (auth, player, audio formats, ABC/MIDI)
│   ├── integration/              #   Integration tests (agent, FastAPI e2e)
│   └── eval/                     #   ADK evaluation configs & datasets
│
├── run_player.py                 #  CLI entry point — standalone polling loop
├── pyproject.toml                #  Dependencies & tool config (uv / hatch)
├── uv.lock                       #  Locked dependency graph
├── Dockerfile                    #  Container build (API-server mode)
├── agents-cli-manifest.yaml      #  agents-cli project manifest
└── AGENTS.md                     #  AI-assisted development rules
```

### Key directories

| Directory   | Purpose |
|-------------|---------|
| `audio/`    | Drop audio files here. Supports `.wav`, `.mp3`, `.ogg`, `.flac`, `.mp4`, `.abc` (ABC notation), and `.mid` (MIDI). SoundFont files (`.sf2` / `.sf3`) placed here are auto-discovered for MIDI playback. Config files (`mapping.yaml`, `segments.yaml`, `file_tags.yaml`) also live here. |
| `capture/`  | Staging area for screen captures. The player writes cropped screenshots here; you can also place images manually for offline testing without a live game window. |
| `tests/`    | `unit/` for isolated tests, `integration/` for server e2e, `eval/` for ADK evaluation datasets. |

## Features

- **Automatic location detection** — captures the game screen, crops the minimap, and parses coordinates + location name via Tesseract OCR.
- **FOSS SoundFont Packaging** — Packages the high-quality, compressed `MuseScore_General.sf3` directly in the repository under the CC-BY 3.0 license.
- **On-Demand Lossless Downloader** — Trigger a background download of the 215 MB uncompressed `MuseScore_General.sf2` (ULTRA) SoundFont directly from the Preferences menu, with real-time percentage and spinner feedback.
- **Dynamic SoundFont Scanning** — Automatically scans the `audio/` directory for any custom `.sf2`/`.sf3` files and lists them in the Preferences dropdown.
- **Smart Playback Restart** — Changing the SoundFont immediately invalidates/clears the `.cache/` directory and restarts active MIDI/ABC playback from the current `start_time` to apply the new instrument sounds.
- **Gemini Vision fallback** — when local OCR is inconclusive, falls back to a multimodal LLM (configurable: Gemini, GPT-4o, Claude).
- **Expanded OCR Dictionary** — Pre-populated [lotro_words.txt](file:///home/chuubi/Desktop/vibe-coding-2026/capstone/roving-bard/app/lotro_words.txt) with **229 game locations** (including Sírlond, Evendim, and Gundabad) to ensure extremely high local OCR accuracy.
- **Manual Bounding Box Adjuster** — Click the *Bounding Box Selector* card in the UI to manually adjust coordinates (X, Y, Width, Height) and cycle OCR preprocessing passes on-demand.
- **Audio Library Sorting by Type** — Toggle the file icon in the Audio Library to group audio files and segments by their extension before applying alphabetical sorting.
- **10-band parametric EQ** — real-time equalizer (32 Hz – 16 kHz) using IIR peaking filters via scipy.
- **Segment system** — save, edit, and export named sub-ranges of tracks with custom volume, bounds, and EQ presets.
- **Browser GUI** — glassmorphism dark/light theme, 7-language localisation (EN-US, EN-UK, FR, DE, ES, IT, RU), interactive EQ panel, seek bar with range highlighting, audio upload, file tagging, mapping editor, and screenshot viewer.
- **REST API** — authenticated FastAPI endpoints for playback control, config management, segments, EQ, file management, and screenshots.
- **ADK agent integration** — agent tools for screen checking, playback, volume control, and status queries; ADK Playground support.

## Requirements

### Python

- **Python** ≥ 3.11, < 3.14
- **uv** — Python package manager — [Install](https://docs.astral.sh/uv/getting-started/installation/)
- **agents-cli** — `uv tool install google-agents-cli`
- **Core Python Dependencies** (managed via `pyproject.toml`):
  - `google-adk[gcp]` — Google Agent Development Kit & Cloud Trace/logging integrations
  - `pygame` — Audio output & player controls
  - `pytesseract` & `opencv-python` — Screen OCR and screenshot processing
  - `scipy` & `soundfile` — 10-band parametric EQ processing
  - `litellm` — Fallback vision-language model calls
  - `torch` & `transformers` — Local VLM execution (Florence-2 / Hugging Face models)


### System dependencies

| Dependency | Purpose | Install (Debian/Ubuntu) |
|---|---|---|
| **Tesseract OCR** | Local OCR for minimap text | `sudo apt install tesseract-ocr` |
| **SDL2 + SDL_mixer** | Audio playback via pygame | Usually bundled with pygame; `sudo apt install libsdl2-mixer-2.0-0` if needed |
| **libsndfile** | Audio I/O for EQ processing (soundfile) | `sudo apt install libsndfile1` |
| **FluidSynth** *(optional)* | Legacy MIDI synthesis engine (optional system fallback) | `sudo apt install fluidsynth` |

### Environment variables

| Variable | Purpose |
|---|---|
| `AGENT_API_KEY` | API key for GUI / REST authentication (also checked: `GOOGLE_API_KEY`, `GEMINI_API_KEY`). Bypassed automatically for localhost loopback clients. |
| `SDL_SOUNDFONTS` | *(optional)* Explicit path to a SoundFont file; overrides auto-discovery |
| `INTEGRATION_TEST` | Set to `TRUE` to mock LiteLLM responses during testing |
| `LOGS_BUCKET_NAME` | *(optional)* GCS bucket for artifact / log storage |

### SoundFont Setup & Legacy FluidSynth Status

MIDI and ABC notation files require a SoundFont (`.sf2` or `.sf3`) for instrument synthesis. Historically, this required setting up system-wide FluidSynth and legacy SoundFonts. Roving Bard simplifies this by bundling a modern FOSS SoundFont directly in the repository:

#### 🌟 The Bundled SoundFont Advantage (`MuseScore_General.sf3`)
* **Zero Configuration**: MIDI and ABC playback work completely out-of-the-box. There is no need to install external packages or configure environment variables.
* **Modern Instrument Fidelity**: Unlike legacy SoundFonts (e.g., `FluidR3_GM` or `TimGM6mb`), the **MuseScore General** SoundFont features professional-grade, highly realistic instrument samples.
* **Minimal Footprint**: By utilizing the compressed `.sf3` format (which stores samples using Ogg Vorbis compression), the file size is reduced to just **40 MB** (compared to 140+ MB for FluidR3 or 215 MB for uncompressed `.sf2` files), keeping the repository clone footprint extremely lightweight.

#### 🏛️ Legacy FluidSynth & FluidR3 Support
System-installed FluidSynth and legacy SoundFonts are now treated as **optional fallbacks**. The player resolves SoundFonts in the following order:

1. **`SDL_SOUNDFONTS` env var** — if set and the file exists, used directly.
2. **`audio/` directory** — any custom `.sf2` / `.sf3` files placed here (searched first, including the bundled `MuseScore_General.sf3` and the on-demand `MuseScore_General.sf2` ULTRA version).
3. **System paths** — scans standard Linux fallback locations:
   - `/usr/share/sounds/sf2/FluidR3_GM.sf2`
   - `/usr/share/sounds/sf2/default-GM.sf2`
   - `/usr/share/sounds/sf2/TimGM6mb.sf2`
   - `/usr/share/sounds/sf3/FluidR3_GM.sf3`
   - `/usr/share/midi/soundfont/FluidR3_GM.sf2`

## Quick Start

Install required packages:

```bash
agents-cli install
```

### 1. Start the Music Player Runner

Start the automatic screen monitoring loop in your terminal:

```bash
uv run python run_player.py
```

The runner will:
- Auto-generate test audio files (`town.wav`, `forest.wav`, `boss.wav`, `cave.wav`) in the `audio/` directory if they don't exist.
- Capture the screen, crop to the minimap area, and run local OCR (Tesseract) using the expanded 229-word dictionary to parse location and coordinates.
- Smoothly crossfade background music based on the active region.
- Automatically fall back to Gemini Vision via LiteLLM if local OCR is inconclusive.

### 2. Browser GUI

Once the runner is active, open the GUI dashboard:

```
http://localhost:8000/gui
```

*(Note: Localhost loopback connections automatically bypass API key authentication and display a green checkmark next to the status badge).*

The dashboard provides full playback controls, a 10-band EQ, segment management, audio upload, live screenshot viewer, and configuration editing — all with dark/light theme toggle and 7-language localisation.

### 3. Interactive ADK Playground

To talk directly with the agent:

```bash
agents-cli playground
```

## Configuration

All configuration lives in `audio/mapping.yaml`:

```yaml
active_soundfont: "MuseScore_General.sf3"
minimap_bounds:        # Screen region to crop for OCR
  x: 0.8              # 80% from left
  y: 0.05             # 5% from top
  width: 0.15         # 15% of screen width
  height: 0.15        # 15% of screen height

transitions:
  fade_out_ms: 1500   # Crossfade timing
  fade_in_ms: 1500

playlist_directory: "audio"
model_name: "gemini/gemini-1.5-flash"   # Fallback vision model
polling_interval: 2.0                    # Seconds between screen checks

mappings:                                # Location → track rules
  - location_name: "Town"
    track_file: "town.wav"
  - location_name: "Forest"
    track_file: "forest.wav"
  - ns_min: 10.0                         # Coordinate-range match
    ns_max: 20.0
    ew_min: -80.0
    ew_max: -60.0
    track_file: "cave.wav"
```

## Commands

| Command | Description |
|---|---|
| `agents-cli install` | Install dependencies using uv |
| `agents-cli playground` | Launch local ADK development environment |
| `agents-cli lint` | Run code quality checks (ruff, ty, codespell) |
| `agents-cli eval` | Evaluate agent behaviour (see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests |

## 🛠️ Project Management

| Command | What It Does |
|---|---|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customisations |

## REST API

All endpoints require authentication (bypassed automatically for localhost loopback).

| Endpoint | Method | Purpose |
|---|---|---|
| `/gui` | GET | Browser dashboard |
| `/api/status` | GET | Current playback state & available SoundFonts |
| `/api/control` | POST | Player actions (play, stop, pause, resume, seek, volume, bounds, select) |
| `/api/config` | POST | Hot-reload mapping configuration |
| `/api/screenshot` | GET | Latest cropped screenshot (PNG) |
| `/api/screenshot/refresh` | POST | Re-capture screen |
| `/api/audio-files` | GET | List audio files with tags |
| `/api/upload-audio` | POST | Upload audio file to playlist directory |
| `/api/soundfont/download` | POST | Trigger background download of `MuseScore_General.sf2` |
| `/api/soundfont/download/status` | GET | Read download progress percentage |
| `/api/segments` | GET/POST/DELETE | Manage saved segments |
| `/api/file-tags` | POST | Update file tags |
| `/api/eq` | GET/POST | Read/write 10-band EQ gains |
| `/api/env-status` | GET | Check configured API key env vars |
| `/feedback` | POST | Submit structured feedback |

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` — it auto-reloads on save. The GUI at `app/gui.html` is served directly by the FastAPI server and requires no build step.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

The Dockerfile builds a lightweight API-server container (port 8080). Note that the container runs in **simulated audio mode** since it does not include system audio libraries or Tesseract — it is designed for serving the API and GUI.

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging via OpenTelemetry. Disabled automatically during integration tests.
