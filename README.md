# Roving Bard: Adaptive, Location-Aware Audio for LOTRO

Roving Bard is a game-aware music player agent that captures the screen, recognises the in-game location via OCR (with local VLM and Gemini Vision fallback), and seamlessly transitions background music to match the active region. It is built on the Google Agent Development Kit (ADK) and includes a single-page browser GUI.

---

## 📁 Project Structure

```
project/
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
- **On-Demand SoundFont Downloader**: Download the uncompressed, lossless `MuseScore_General.sf2` SoundFont directly from the Preferences menu, with background thread updates.
- **High-Fidelity Cross-Platform Audio Engine (Windows, Linux, macOS)**: Features a robust, low-latency `sounddevice` engine (built on PortAudio). It delivers studio-grade real-time SoundFont MIDI/ABC synthesis, smooth logarithmic volume-fading transitions, and high-fidelity parametric EQ filtering natively across all major operating systems.
- **10-Band Parametric EQ**: Adjust playback frequencies (32 Hz – 16 kHz) in real-time using cross-platform scipy-powered IIR peaking filters.
- **Audio Library & Track Management**: Scan, search, and manage your music library with full metadata tagging support. Supports uploading new tracks directly from the GUI, extracting embedded tag details (titles, artists, albums, durations, and bitrates), and sorting/filtering tracks by type (Wave, MP3, Ogg Vorbis, FLAC, AAC, M4A, ABC Notation, and MIDI).
- **Audio Segments & Custom Slices**: Define, save, and play custom audio loops/slices of tracks with independent volume, panning, and 10-band EQ settings.

---

## 🛠️ Requirements

### Python Environment
- **Python** ≥ 3.11, < 3.14
- **uv** — Fast Python package manager — [Install](https://docs.astral.sh/uv/getting-started/installation/)
- **google-agents-cli** — CLI for ADK agents — Install with `uv tool install google-agents-cli`

### System Dependencies
- **PortAudio**: Required for system audio output (`sudo apt install libportaudio2` on Linux).
- **Tesseract OCR**: Local OCR for minimap text (`sudo apt install tesseract-ocr`).
- **libsndfile**: For soundfile EQ processing (`sudo apt install libsndfile1`).
- **FluidSynth** (Optional): `sudo apt install fluidsynth` on Linux. System-installed FluidSynth soundfonts, such as `FluidR3_GM.sf2` or `TimGM6mb.sf2`, are supported as legacy fallback options but are optional and not required due to our bundled SoundFont.
- **Ollama** (Optional): Local VLM service manager required for running offline models like Moondream or Qwen (`curl -fsSL https://ollama.com/install.sh | sh` on Linux). (Note: The `ollama` Python package is not required as the backend communicates with the local Ollama server via direct HTTP REST API calls).

### Core Python Dependencies (managed via `pyproject.toml`)
- `google-adk[gcp]` — Google Agent Development Kit & Cloud integrations
- `sounddevice` — Unified audio output and device routing
- `pytesseract` & `opencv-python` — Screen OCR and screenshot processing
- `scipy` & `soundfile` — Parametric EQ processing
- `litellm` — Fallback vision-language model API calls
- `torch` & `transformers` — Local VLM execution

### Environment Variables & API Keys
To enable all features and display active integration badges in the GUI dashboard, export the following environment variables before starting the server:

- `AGENT_API_KEY` (Optional / Recommended): Required to secure REST API routes when accessed by remote network clients. (Loopback connections from localhost bypass authorization for developer convenience).
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Optional): Required only if you select the cloud-based **Gemini 2.5 Flash Lite** vision fallback. Local OCR (Tesseract) and local VLM models (Florence-2, Moondream, Qwen, etc.) run entirely offline and do not require keys.

## ⚡ GPU Acceleration Support

Roving Bard utilizes local VLMs (such as Florence-2, Moondream, and Qwen2-VL) to perform zero-shot visual character transcription. To run these models at low latency, the backend leverages hardware acceleration across all major GPU vendors (Nvidia, Apple, AMD, Intel) on Windows, Linux, and macOS.

### 🎮 Supported GPU Hardware & Frameworks

| Vendor | OS | Acceleration API | Primary Driver Stack | Platform Nuances |
|---|---|---|---|---|
| **Nvidia** | Windows, Linux | **CUDA** | NVIDIA Proprietary Driver & CUDA Toolkit | Default target for GGUF/Ollama and PyTorch. Ensure PyTorch CUDA version matches GPU compute capability (e.g., PyTorch 12.x builds require compute capability &ge; 7.5; older cards like GTX 1070/Pascal use 6.1 and will trigger compatibility warnings in PyTorch but are fully accelerated directly in Ollama/llama.cpp GGUF runs). |
| **Apple Silicon (M1–M4)** | macOS | **Metal (MPS)** | Apple Metal Framework (Native) | Apple macOS uses Unified Memory Architecture (UMA) where system RAM and VRAM are shared dynamically. Ollama auto-allocates layers to Metal. In PyTorch, acceleration uses `torch.backends.mps` natively. |
| **AMD** | Windows, Linux | **ROCm / DirectML** | AMD ROCm (Linux) / Radeon Software (Windows) | On Linux, ROCm requires matching kernel drivers. PyTorch requires installing the AMD-compatible build (`torch+rocm`). On Windows, Ollama utilizes DirectML/HIP drivers to offload model layers. |
| **Intel** | Windows, Linux | **oneAPI (SYCL) / DirectML** | Intel oneAPI Base Toolkit / Intel Graphics Drivers | Intel Arc discrete GPUs and Xe integrated graphics are supported in Ollama via oneAPI/SYCL. For local PyTorch execution, acceleration is enabled via the Intel Extension for PyTorch (IPEX) or OpenVINO. |

### 🔧 Hardware Resolution Layer

Roving Bard resolves and coordinates the hardware execution target across two distinct runtime environments:

1. **Python-Native Models (e.g., Florence-2)**:
   Executed natively by our FastAPI backend using PyTorch. The backend dynamically determines and binds the target compute device at model initialization:
   * **Nvidia/AMD GPUs**: PyTorch checks `torch.cuda.is_available()` to bind model memory to `"cuda"`. On AMD Linux configurations, ROCm-enabled PyTorch builds map AMD hardware directly to CUDA calls natively.
   * **Apple Silicon (M1–M4)**: Checks `torch.backends.mps.is_available()` to bind model memory to `"mps"`, leveraging Apple's native Metal Performance Shaders.
   * **Intel Arc/CPUs**: Discrete Intel Arc GPUs run accelerated via the Intel Extension for PyTorch (IPEX) or OpenVINO, while standard configurations fall back to optimized, multi-threaded CPU tensor operations.

2. **Offline Ollama/GGUF Models (e.g., Moondream, Qwen2-VL, Qwen2.5-VL)**:
   Executed by the local Ollama server. Because Roving Bard communicates with Ollama via loopback HTTP REST API calls, Ollama acts as a middle-tier hardware manager, dynamically offloading GGUF model layers to CUDA (Nvidia), Metal (Apple), ROCm/DirectML (AMD), or oneAPI/SYCL (Intel) depending on what accelerator hardware it discovers on the host system.

### 🛠️ Hardware Memory Management
* **VRAM Monitoring**: The FastAPI backend queries active GPU memory usage via PyTorch (`torch.cuda.max_memory_allocated()`) or Metal drivers, exposing real-time peak VRAM consumption on the dashboard stats panel.
* **Auto-Deallocation on Switch**: To prevent VRAM fragmentation and multi-model collisions, the backend monitors the Ollama process state using `/api/ps`. When switching methods, the active local model is dynamically unloaded (`keep_alive: "0s"`) and verified clear before the new model is loaded, preventing silent fallback to CPU execution.

---

## 🚀 Quick Start

### 1. Clone the Repository
Clone the project repository and navigate into the repository root directory:
```bash
git clone https://github.com/Evergold/roving-bard.git
cd roving-bard
```

### 2. Install Dependencies
Run dependency sync from the repository root:
```bash
agents-cli install
```

### 3. Start the FastAPI Web Server
To start the development server with auto-reloading enabled, navigate into the project subdirectory and run:
```bash
cd roving-bard
uv run uvicorn app.fast_api_app:app --reload
```
The server will boot on `http://127.0.0.1:8000`.

### 4. Open the GUI Dashboard
Once the server is running, navigate to:
```
http://localhost:8000/gui
```
* **Auto-Scanning**: Turn on the **Auto Scanning** toggle in the status header to begin scanning your screen at the configured interval.
* **Localhost Access**: Loopback connections from localhost bypass API key authentication and display a green checkmark next to the status badge.

### 5. Alternative: Standalone CLI Player Loop
If you prefer running a command-line polling loop without the GUI, navigate into the project subdirectory and run:
```bash
cd roving-bard
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

## 🎹 SoundFont Setup & MIDI Playback

Roving Bard synthesizes `.mid` (MIDI) and `.abc` (ABC notation) music tracks in real-time. This synthesis is driven by a SoundFont file (`.sf2` or `.sf3`).

### 1. Bundled SoundFont (SF3)
To ensure MIDI and ABC playback works out of the box, a lightweight, compressed SoundFont is pre-bundled:
- **File**: `roving-bard/audio/MuseScore_General.sf3`
- **Configuration**: Set `active_soundfont: "MuseScore_General.sf3"` in `audio/mapping.yaml`.
- **Why it is Better**: The bundled `MuseScore_General.sf3` file provides significantly higher instrument synthesis quality than system-default SoundFonts. Additionally, it is highly compressed (~35 MB vs. ~215 MB for the uncompressed `.sf2` version), avoiding the need to download large files over the internet on startup, reducing load times, and optimizing memory usage during playback.

### 2. Legacy SoundFont Fallbacks (Optional)
The engine automatically searches standard system directories for legacy FluidSynth soundfonts (e.g. `/usr/share/sounds/sf2/FluidR3_GM.sf2` or `/usr/share/midi/soundfont/default.sf2`) as fallbacks. These are supported but entirely optional, as the bundled MuseScore SoundFont provides superior acoustic performance and memory efficiency.

### 3. High-Quality SoundFont Downloader (SF2)
For high-fidelity audio synthesis, you can download the full, lossless **MuseScore General** SoundFont:
- **File**: `MuseScore_General.sf2` (approx. 215 MB)
- **How to Download**:
  - Open the browser GUI dashboard (`http://localhost:8000/gui`).
  - Navigate to **Preferences** (top-left) and under SoundFont select **MuseScore General (ULTRA)**.
  - Alternatively, trigger the download via the REST API:
    ```bash
    curl -X POST http://localhost:8000/api/soundfont/download
    ```
- **Hot-swapping**: Once downloaded, select it from the SoundFont dropdown menu in the GUI or update `audio/mapping.yaml` to `active_soundfont: "MuseScore_General.sf2"`. The engine will reload it instantly without requiring a server restart.

### 3. Dynamic Instrument Selection for ABC Tracks
When playing ABC notation files, you can choose and hot-swap the active instrument directly from the dashboard:
- **How to Use**: Click the **Instrument** button on the playback control bar in the GUI to open the instrument popover grid. Select from 12+ instrument presets (including Lute, Harp, Bagpipe, Flute, Clarinet, Fiddle, and Drums).
- **Backend Synthesis**: The backend automatically recompiles the ABC notes to MIDI byte streams, setting the active instrument program number, and synthesizes the track on-the-fly.

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
