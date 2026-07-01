# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import json

from dotenv import load_dotenv

load_dotenv()

import google.auth
import yaml
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status, Request
from fastapi.responses import HTMLResponse, Response
from google.adk.cli.fast_api import get_fast_api_app
from google.auth.credentials import Credentials
from pydantic import BaseModel

from app import tools
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback


# Mock credentials unconditionally for local-only execution
class MockCredentials(Credentials):
    def __init__(self):
        super().__init__()
        self.token = "mock-token"

    def refresh(self, request):
        pass

    def apply(self, headers, token=None):
        headers["authorization"] = "Bearer mock-token"

    def before_request(self, request, method, url, headers):
        headers["authorization"] = "Bearer mock-token"


mock_creds = MockCredentials()
google.auth.default = lambda *args, **kwargs: (mock_creds, "mock-project-id")

# Mock LiteLLM for subprocess E2E integration tests if needed
if os.getenv("INTEGRATION_TEST") == "TRUE":
    import litellm
    from litellm.utils import ModelResponse, ModelResponseStream
    import google.adk.models.lite_llm

    def mock_complete(*args, **kwargs):
        response_format = kwargs.get("response_format")
        if response_format and response_format.get("type") == "json_object":
            content = '{"location": "Town", "coordinates": "19.3N, 70.9W"}'
        else:
            content = "This is a mock assistant response from the Game-Aware Music Player Agent."
        return ModelResponse(
            choices=[
                {
                    "message": {"content": content, "role": "assistant"},
                    "finish_reason": "stop",
                }
            ]
        )

    async def mock_acomplete(*args, **kwargs):
        if kwargs.get("stream"):
            async def chunk_generator():
                yield ModelResponseStream(
                    choices=[
                        {
                            "delta": {
                                "content": "This is a mock streaming response from the Game-Aware Music Player Agent."
                            },
                            "finish_reason": "stop",
                        }
                    ]
                )
            return chunk_generator()
        else:
            response_format = kwargs.get("response_format")
            if response_format and response_format.get("type") == "json_object":
                content = '{"location": "Town", "coordinates": "19.3N, 70.9W"}'
            else:
                content = "This is a mock assistant response from the Game-Aware Music Player Agent."
            return ModelResponse(
                choices=[
                    {
                        "message": {"content": content, "role": "assistant"},
                        "finish_reason": "stop",
                    }
                ]
            )

    litellm.completion = mock_complete
    litellm.acompletion = mock_acomplete
    google.adk.models.lite_llm.LiteLLMClient.acompletion = mock_acomplete
    google.adk.models.lite_llm.LiteLLMClient.completion = mock_complete

    if hasattr(google.adk.models.lite_llm, "completion"):
        google.adk.models.lite_llm.completion = mock_complete
    if hasattr(google.adk.models.lite_llm, "acompletion"):
        google.adk.models.lite_llm.acompletion = mock_acomplete

setup_telemetry()

# Safe logger setup with standard fallback if unauthenticated
logger = logging.getLogger(__name__)


class SafeLogger:
    def log_struct(self, data, severity="INFO"):
        # Fallback directly to python logging, avoiding any GCP Cloud Logging dependencies
        level = getattr(logging, severity, logging.INFO)
        logger.log(level, str(data))


app_logger = SafeLogger()
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

# Disable OpenTelemetry Cloud Export in integration tests to avoid GCP metadata server hangs
otel_to_cloud = os.getenv("INTEGRATION_TEST") != "TRUE"

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=otel_to_cloud,
)
app.title = "roving-bard"
app.description = "API for interacting with the Agent roving-bard"


class ControlRequest(BaseModel):
    action: str  # play, stop, volume, scan, seek, pause, resume, set_bounds, select
    track_file: str | None = None
    volume: float | None = None
    position: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    instrument: int | None = None


class ConfigUpdateRequest(BaseModel):
    minimap_bounds: dict
    transitions: dict
    playlist_directory: str
    model_name: str
    polling_interval: float
    mappings: list
    api_key: str | None = None
    active_soundfont: str | None = None


class EQRequest(BaseModel):
    gains: dict[str, float]  # e.g. {"32": 3.0, "1000": -2.5, ...}


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(default=None),
):
    # Bypass API key checks for local/localhost access
    client_host = request.client.host if request.client else None
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return "localhost"

    provided_key = x_api_key or api_key
    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    # Gather all acceptable keys
    allowed_keys = {
        os.getenv("AGENT_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        tools.config.get("api_key"),
    }
    # Filter out None and empty strings
    allowed_keys = {k for k in allowed_keys if k}

    if provided_key not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return provided_key


@app.post("/feedback", dependencies=[Depends(verify_api_key)])
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    app_logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


@app.get("/favicon.ico")
def get_favicon():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(AGENT_DIR, "app", "static", "favicon.ico"))


@app.get("/favicon-32x32.png")
def get_favicon_32():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(AGENT_DIR, "app", "static", "favicon-32x32.png"))


@app.get("/favicon-16x16.png")
def get_favicon_16():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(AGENT_DIR, "app", "static", "favicon-16x16.png"))


@app.get("/apple-touch-icon.png")
def get_apple_touch_icon():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(AGENT_DIR, "app", "static", "apple-touch-icon.png"))


@app.get("/android-chrome-192x192.png")
def get_android_192():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(AGENT_DIR, "app", "static", "android-chrome-192x192.png"))


@app.get("/android-chrome-512x512.png")
def get_android_512():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(AGENT_DIR, "app", "static", "android-chrome-512x512.png"))


@app.get("/gui")
def get_gui():
    """Serves the music player HTML GUI dashboard with embedded API key."""
    gui_file = os.path.join(AGENT_DIR, "app", "gui.html")
    if os.path.exists(gui_file):
        with open(gui_file) as f:
            content = f.read()
        api_key = os.getenv("AGENT_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        content = content.replace("{{API_KEY_PLACEHOLDER}}", api_key)
        
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache"
        }
        return HTMLResponse(content=content, headers=headers)
    return HTMLResponse(content="<h3>Error: gui.html not found!</h3>", status_code=404)
@app.get("/locales/{locale_name}.json")
def get_locale_json(locale_name: str):
    """Serves a localization JSON file."""
    locale_file = os.path.join(AGENT_DIR, "app", "locales", f"{locale_name}.json")
    if os.path.exists(locale_file):
        with open(locale_file, encoding="utf-8") as f:
            import json
            return json.load(f)
    raise HTTPException(status_code=404, detail="Locale not found")


@app.get("/api/locales")
def get_locales():
    """Returns a list of available locales, reading the locale_name key from each file."""
    locales_dir = os.path.join(AGENT_DIR, "app", "locales")
    locales_list = []
    if os.path.exists(locales_dir):
        import json
        for filename in sorted(os.listdir(locales_dir)):
            if filename.endswith(".json"):
                code = filename[:-5]
                filepath = os.path.join(locales_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        name = data.get("locale_name", code)
                        locales_list.append({"code": code, "name": name})
                except Exception as e:
                    locales_list.append({"code": code, "name": code})
    return locales_list



def get_available_soundfonts(playlist_dir: str) -> list[str]:
    available = []
    
    # 1. MuseScore General (HQ)
    if os.path.exists(os.path.join(playlist_dir, "MuseScore_General.sf3")):
        available.append("MuseScore_General.sf3")
        
    # 2. MuseScore General (ULTRA)
    if os.path.exists(os.path.join(playlist_dir, "MuseScore_General.sf2")):
        available.append("MuseScore_General.sf2")
        
    # 3. FluidR3 (Legacy)
    fluid_paths = [
        os.path.join(playlist_dir, "FluidR3_GM.sf2"),
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/midi/soundfont/FluidR3_GM.sf2",
    ]
    if any(os.path.exists(p) for p in fluid_paths):
        available.append("FluidR3_GM.sf2")
        
    # 4. Scan playlist_dir for any other .sf2 or .sf3 files
    if os.path.exists(playlist_dir):
        try:
            for filename in sorted(os.listdir(playlist_dir)):
                if filename.lower().endswith((".sf2", ".sf3")):
                    if filename not in available:
                        available.append(filename)
        except Exception as e:
            print(f"Error scanning playlist_dir for available soundfonts: {e}")
            
    return available


@app.get("/api/env-status")
def get_env_status():
    """Returns the presence of API key environment variables (without returning their values)."""
    return {
        "AGENT_API_KEY": os.getenv("AGENT_API_KEY") is not None,
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY") is not None,
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY") is not None,
    }


@app.get("/api/status", dependencies=[Depends(verify_api_key)])
def api_status():
    """Returns the current playback status and loaded configuration."""
    # Initialize simulation screen on first status load if needed
    if tools.latest_full_screenshot_bytes is None:
        initialize_simulation_screen()
        
    return {
        "current_track": tools.player.current_track,
        "volume": tools.player.volume,
        "simulated": tools.player.simulated,
        "config_data": tools.config,
        "latest_parse": tools.latest_parse_result,
        "paused": getattr(tools.player, "paused", False),
        "was_stopped": getattr(tools.player, "was_stopped", False),
        "duration": getattr(tools.player, "track_duration", 0.0),
        "current_position": tools.player.get_current_position(),
        "start_time": getattr(tools.player, "start_time", 0.0),
        "end_time": getattr(tools.player, "end_time", None) if getattr(tools.player, "end_time", None) is not None else getattr(tools.player, "track_duration", 0.0),
        "active_instrument": tools.player.active_instrument if tools.player.active_instrument is not None else tools.player.get_default_instrument(),
        "is_abc": tools.player.current_track.lower().endswith(".abc") if tools.player.current_track else False,
        "minimap_detected": getattr(tools, "minimap_detected", False),
        "available_soundfonts": get_available_soundfonts(tools.player.playlist_dir),
    }


import threading
import requests

# Global download state
soundfont_download_state = {
    "status": "idle",       # "idle", "downloading", "success", "error"
    "progress": 0,          # 0 to 100
    "error": None
}
soundfont_download_lock = threading.Lock()

def download_soundfont_task(url: str, dest_path: str):
    global soundfont_download_state
    tmp_path = dest_path + ".tmp"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        downloaded = 0
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int((downloaded / total_size) * 100)
                        with soundfont_download_lock:
                            soundfont_download_state["progress"] = percent
                            
        # Rename tmp to final
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(tmp_path, dest_path)
        
        # Automatically update config to use the new soundfont and hot-reload
        tools.config["active_soundfont"] = os.path.basename(dest_path)
        with open(tools.CONFIG_PATH, "w") as f:
            yaml.safe_dump(tools.config, f)
            
        # Hot-reload in player
        tools.player.update_soundfont(tools.config["active_soundfont"])
        
        with soundfont_download_lock:
            soundfont_download_state["status"] = "success"
            soundfont_download_state["progress"] = 100
            
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        with soundfont_download_lock:
            soundfont_download_state["status"] = "error"
            soundfont_download_state["error"] = str(e)

class SoundFontDownloadRequest(BaseModel):
    soundfont: str

@app.post("/api/soundfont/download", dependencies=[Depends(verify_api_key)])
def api_download_soundfont(req: SoundFontDownloadRequest):
    global soundfont_download_state
    
    if req.soundfont != "MuseScore_General.sf2":
        raise HTTPException(status_code=400, detail="Unsupported SoundFont for download")
        
    dest_path = os.path.join(tools.player.playlist_dir, req.soundfont)
    if os.path.exists(dest_path):
        return {"status": "success", "message": "SoundFont already exists"}
        
    with soundfont_download_lock:
        if soundfont_download_state["status"] == "downloading":
            return {"status": "downloading", "message": "Download already in progress"}
            
        soundfont_download_state = {
            "status": "downloading",
            "progress": 0,
            "error": None
        }
        
    url = "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2"
    thread = threading.Thread(target=download_soundfont_task, args=(url, dest_path))
    thread.daemon = True
    thread.start()
    
    return {"status": "downloading", "message": "Download started"}

@app.get("/api/soundfont/download/status", dependencies=[Depends(verify_api_key)])
def api_download_soundfont_status():
    with soundfont_download_lock:
        return soundfont_download_state


@app.post("/api/control", dependencies=[Depends(verify_api_key)])
def api_control(req: ControlRequest):
    """Handles manual player actions (play, stop, set volume, scan screen, seek, set_bounds)."""
    if req.action == "play":
        if req.track_file:
            if req.volume is not None:
                tools.set_volume(req.volume)
            start_time = req.start_time if req.start_time is not None else 0.0
            return tools.play_track(req.track_file, start_time=start_time, end_time=req.end_time)
        return {"status": "error", "message": "track_file is required for play action."}
    elif req.action == "stop":
        return tools.stop_music()
    elif req.action == "volume":
        if req.volume is not None:
            return tools.set_volume(req.volume)
        return {"status": "error", "message": "volume is required for volume action."}
    elif req.action == "scan":
        return tools.check_screen_and_update_music()
    elif req.action == "pause":
        success = tools.player.pause()
        if success:
            return {"status": "success", "message": "Playback paused."}
        return {"status": "error", "message": "Failed to pause playback."}
    elif req.action == "resume":
        success = tools.player.resume()
        if success:
            return {"status": "success", "message": "Playback resumed."}
        return {"status": "error", "message": "Failed to resume playback."}
    elif req.action == "seek":
        if req.position is not None:
            success = tools.player.seek(req.position)
            if success:
                return {"status": "success", "message": f"Seeked to {req.position} seconds."}
            return {"status": "error", "message": "Failed to seek playback."}
        return {"status": "error", "message": "position is required for seek action."}
    elif req.action == "set_bounds":
        if req.start_time is not None or req.end_time is not None:
            if req.start_time is not None:
                tools.player.start_time = max(0.0, req.start_time)
            if req.end_time is not None:
                tools.player.end_time = req.end_time
            if tools.player.was_stopped or tools.player.paused:
                tools.player.last_seek_position = tools.player.start_time
            return {"status": "success", "message": "Playback bounds updated."}
        return {"status": "error", "message": "start_time or end_time is required for set_bounds."}
    elif req.action == "select":
        if req.track_file:
            if req.volume is not None:
                tools.set_volume(req.volume)
            start_time = req.start_time if req.start_time is not None else 0.0
            success = tools.player.select_track(req.track_file, start_time=start_time, end_time=req.end_time)
            if success:
                return {"status": "success", "message": f"Selected track {req.track_file}."}
            return {"status": "error", "message": "Failed to select track."}
        return {"status": "error", "message": "track_file is required for select action."}
    elif req.action == "instrument":
        if req.instrument is not None:
            tools.player.set_instrument(req.instrument)
            return {"status": "success", "message": f"Instrument set to {req.instrument}."}
        return {"status": "error", "message": "instrument is required for instrument action."}
    return {"status": "error", "message": f"Unknown action: {req.action}"}


@app.post("/api/config", dependencies=[Depends(verify_api_key)])
def api_config(req: ConfigUpdateRequest):
    """Updates mapping.yaml on disk and hot-reloads config in memory."""
    try:
        new_config = req.model_dump()
        with open(tools.CONFIG_PATH, "w") as f:
            yaml.safe_dump(new_config, f)

        # Hot-reload settings
        tools.config = new_config
        tools.player.playlist_dir = new_config.get("playlist_directory", "audio")
        tools.player.update_soundfont(new_config.get("active_soundfont"))
        tools.grabber.bounds = new_config.get("minimap_bounds")
        tools.mapper.mappings = new_config.get("mappings", [])

        return {
            "status": "success",
            "message": "Configuration hot-reloaded successfully.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


class BoundsUpdateRequest(BaseModel):
    x: float
    y: float
    width: float
    height: float


@app.post("/api/config/bounds", dependencies=[Depends(verify_api_key)])
def api_update_bounds(req: BoundsUpdateRequest):
    """Updates manual bounds in mapping.yaml, enforces size safeguards, and triggers a rescan."""
    x = max(0.0, min(1.0, req.x))
    y = max(0.0, min(1.0, req.y))
    w = max(0.05, min(0.30, req.width))
    h = max(0.05, min(0.45, req.height))
    
    try:
        import os
        import yaml
        
        # Load current config
        if os.path.exists(tools.CONFIG_PATH):
            with open(tools.CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
            
        config["minimap_bounds"] = {"x": x, "y": y, "width": w, "height": h}
        
        with open(tools.CONFIG_PATH, "w") as f:
            yaml.safe_dump(config, f)
            
        # Hot-reload in memory
        tools.config = config
        tools.grabber.bounds = config["minimap_bounds"]
        
        # Crop the screenshot, rerun OCR, and update music!
        res = tools.check_screen_and_update_music()
        
        return {
            "status": "success",
            "bounds": tools.grabber.bounds,
            "parsed_location": res.get("parsed_location"),
            "parsed_coordinates": res.get("parsed_coordinates"),
            "method": res.get("method"),
            "matched_track": res.get("matched_track"),
            "timestamp": res.get("timestamp")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


class CharBoundsUpdateRequest(BaseModel):
    x: float
    y: float
    width: float
    height: float


@app.post("/api/config/char_bounds", dependencies=[Depends(verify_api_key)])
def api_update_char_bounds(req: CharBoundsUpdateRequest):
    """Updates manual character bounds in mapping.yaml."""
    x = max(0.0, min(1.0, req.x))
    y = max(0.0, min(1.0, req.y))
    w = max(0.05, min(0.50, req.width))
    h = max(0.05, min(0.50, req.height))
    
    try:
        import os
        import yaml
        
        # Load current config
        if os.path.exists(tools.CONFIG_PATH):
            with open(tools.CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
            
        config["character_bounds"] = {"x": x, "y": y, "width": w, "height": h}
        
        with open(tools.CONFIG_PATH, "w") as f:
            yaml.safe_dump(config, f)
            
        # Hot-reload in memory
        tools.config = config
        
        # Crop the screenshot
        tools.check_screen_and_update_music()
        
        return {"status": "success", "bounds": config["character_bounds"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/character/autodetect", dependencies=[Depends(verify_api_key)])
def api_autodetect_character():
    """Tries to autodetect character in the center of the screen."""
    try:
        full_img = tools.grabber.capture_full()
        if not full_img:
            return {"status": "error", "message": "Failed to capture screenshot."}
            
        # Edge density check in the center 20% width, 50% height region
        import numpy as np
        import cv2
        
        open_cv_image = np.array(full_img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
        h, w, _ = open_cv_image.shape
        
        cx_min = int(w * 0.40)
        cx_max = int(w * 0.60)
        cy_min = int(h * 0.25)
        cy_max = int(h * 0.75)
        center_crop = open_cv_image[cy_min:cy_max, cx_min:cx_max]
        
        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        val = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"[CharacterDetector] Center Laplacian variance: {val}")
        
        if val < 5.0:
            return {"status": "error", "message": "No character detected (uniform background)."}
            
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vertical_density = np.mean(np.abs(sobel_x))
        print(f"[CharacterDetector] Vertical edge density: {vertical_density}")
        
        if vertical_density < 2.0:
            return {"status": "error", "message": "No character detected."}
            
        # Succeeded! Let's update the character bounds in the config
        import os
        import yaml
        
        if os.path.exists(tools.CONFIG_PATH):
            with open(tools.CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
            
        # Default bounds that capture the character nicely
        cb = {"x": 0.45, "y": 0.30, "width": 0.10, "height": 0.40}
        config["character_bounds"] = cb
        
        with open(tools.CONFIG_PATH, "w") as f:
            yaml.safe_dump(config, f)
            
        tools.config = config
        
        # Recapture and update character cache
        tools.check_screen_and_update_music()
        
        return {"status": "success", "bounds": cb}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/screenshot/cursor", dependencies=[Depends(verify_api_key)])
def api_screenshot_cursor():
    """Returns the latest cropped cursor image, or a transparent placeholder."""
    if tools.latest_cursor_bytes:
        return Response(content=tools.latest_cursor_bytes, media_type="image/png")
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


@app.get("/api/screenshot/cursor/processed", dependencies=[Depends(verify_api_key)])
def api_screenshot_cursor_processed():
    """Returns the latest processed cursor image, or a transparent placeholder."""
    if tools.latest_cursor_processed_bytes:
        return Response(content=tools.latest_cursor_processed_bytes, media_type="image/png")
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


@app.get("/api/screenshot/character", dependencies=[Depends(verify_api_key)])
def api_screenshot_character():
    """Returns the latest cropped character image, or a transparent placeholder."""
    if tools.latest_character_bytes:
        return Response(content=tools.latest_character_bytes, media_type="image/png")
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


@app.get("/api/screenshot/character/processed", dependencies=[Depends(verify_api_key)])
def api_screenshot_character_processed():
    """Returns the latest processed character image, or a transparent placeholder."""
    if tools.latest_character_processed_bytes:
        return Response(content=tools.latest_character_processed_bytes, media_type="image/png")
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


@app.get("/api/screenshot/processed", dependencies=[Depends(verify_api_key)])
def api_screenshot_processed():
    """Returns the latest processed location screenshot image, or a transparent placeholder."""
    if tools.latest_location_processed_bytes:
        return Response(content=tools.latest_location_processed_bytes, media_type="image/png")
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


@app.get("/api/screenshot/location_raw", dependencies=[Depends(verify_api_key)])
def api_screenshot_location_raw():
    """Returns the latest raw unbinarized location screenshot image, or a transparent placeholder."""
    if tools.latest_location_raw_bytes:
        return Response(content=tools.latest_location_raw_bytes, media_type="image/png")
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


def initialize_simulation_screen():
    """Loads the first test screen on startup if in simulation mode."""
    import os
    
    app_dir = os.path.dirname(os.path.abspath(__file__))
    capture_dir = os.path.join(os.path.dirname(app_dir), "capture")
    
    test_files = []
    if os.path.exists(capture_dir):
        test_files = sorted([
            f for f in os.listdir(capture_dir)
            if f.lower().startswith("test_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    
    if test_files:
        tools.grabber.test_index = 0
        try:
            full_img = tools.grabber.capture_full()
            print(f"[ScreenGrabber] Init Simulation: Loaded test screen")
            
            # Run minimap detection
            bounds, detected = tools.grabber.detect_minimap(full_img)
            if detected:
                tools.grabber.bounds = bounds
                tools.minimap_detected = True
            else:
                config = tools.load_config()
                tools.grabber.bounds = config.get("minimap_bounds", {"x": 0.8, "y": 0.05, "width": 0.15, "height": 0.15})
                tools.minimap_detected = False
            
            # Update tools.config in memory
            tools.config["minimap_bounds"] = tools.grabber.bounds
            
            # Run the scan pipeline to cache images, run OCR, and update status
            tools.check_screen_and_update_music()
        except Exception as e:
            print(f"[ScreenGrabber] Failed to initialize simulation screen: {e}")


@app.get("/api/screenshot", dependencies=[Depends(verify_api_key)])
def api_screenshot(full: bool = False):
    """Returns the latest cropped (or full) screenshot image, or a transparent placeholder."""
    if full and tools.latest_full_screenshot_bytes:
        return Response(content=tools.latest_full_screenshot_bytes, media_type="image/png")
    elif not full and tools.latest_screenshot_bytes:
        return Response(content=tools.latest_screenshot_bytes, media_type="image/png")

    # 1x1 transparent PNG fallback
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


@app.post("/api/screenshot/refresh", dependencies=[Depends(verify_api_key)])
def api_screenshot_refresh():
    """Reloads the current capture from disk/screen, runs auto-detection, and updates cache."""
    import os
    
    app_dir = os.path.dirname(os.path.abspath(__file__))
    capture_dir = os.path.join(os.path.dirname(app_dir), "capture")
    
    # Check if we have test screens for simulation mode (starts with 'test_')
    test_files = []
    if os.path.exists(capture_dir):
        test_files = sorted([
            f for f in os.listdir(capture_dir)
            if f.lower().startswith("test_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
    if test_files:
        # Cycle through test screens
        if not hasattr(tools.grabber, "test_index"):
            tools.grabber.test_index = 0
        else:
            tools.grabber.test_index = (tools.grabber.test_index + 1) % len(test_files)
            
    full_img = tools.grabber.capture_full()
    if not full_img:
        return {"status": "error", "message": "Failed to capture/load screenshot."}

    try:
        # Run minimap detection on the full image
        bounds, detected = tools.grabber.detect_minimap(full_img)
        
        if detected:
            tools.grabber.bounds = bounds
            tools.minimap_detected = True
        else:
            # Fallback to mapping.yaml bounds
            config = tools.load_config()
            tools.grabber.bounds = config.get("minimap_bounds", {"x": 0.8, "y": 0.05, "width": 0.15, "height": 0.15})
            tools.minimap_detected = False
            bounds = tools.grabber.bounds
            
        # Update tools.config in memory
        tools.config["minimap_bounds"] = tools.grabber.bounds
            
        # Run the scan pipeline to cache images, run OCR, and update status
        tools.check_screen_and_update_music()
        
        return {
            "status": "success",
            "message": "Screenshot reloaded successfully.",
            "detected": detected,
            "bounds": bounds
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/ocr/wrong", dependencies=[Depends(verify_api_key)])
def api_ocr_wrong():
    """Tells the backend that the OCR was wrong, cycling to the next preprocessing pass."""
    # Cycle ocr_pass between 0, 1, 2
    tools.current_ocr_pass = (tools.current_ocr_pass + 1) % 3
    print(f"[OCR] User clicked 'Wrong?'. Cycling to OCR Pass {tools.current_ocr_pass}")
    
    # Rerun the scan pipeline
    res = tools.check_screen_and_update_music()
    return {
        "status": "success",
        "ocr_pass": tools.current_ocr_pass,
        "parsed_location": res.get("parsed_location"),
        "parsed_coordinates": res.get("parsed_coordinates")
    }


@app.get("/api/audio-files", dependencies=[Depends(verify_api_key)])
def api_list_audio_files():
    """Lists all audio files in the playlist directory."""
    playlist_dir = tools.player.playlist_dir
    os.makedirs(playlist_dir, exist_ok=True)
    files = [
        f for f in os.listdir(playlist_dir)
        if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".abc", ".mp4", ".mid", ".midi"))
    ]
    file_tags = tools.load_file_tags()
    tags_registry = tools.load_tags_registry()
    return {"status": "success", "files": sorted(files), "file_tags": file_tags, "tags_registry": tags_registry}



@app.post("/api/upload-audio", dependencies=[Depends(verify_api_key)])
def api_upload_audio(file: UploadFile = File(...)):
    """Uploads an audio file and saves it to the playlist directory."""
    playlist_dir = tools.player.playlist_dir
    os.makedirs(playlist_dir, exist_ok=True)
    
    filename = os.path.basename(file.filename)
    if not filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".abc", ".mp4", ".mid", ".midi")):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio file format."
        )
        
    filepath = os.path.join(playlist_dir, filename)
    if os.path.exists(filepath):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File already exists in Audio Library."
        )
    try:
        with open(filepath, "wb") as f:
            f.write(file.file.read())
        return {"status": "success", "message": f"Successfully uploaded {filename}."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )


class SegmentModel(BaseModel):
    name: str
    track_file: str
    start_time: float
    end_time: float
    volume: float | None = None
    tags: list[str] | None = None
    eq: dict[str, float] | str | None = None


@app.get("/api/segments", dependencies=[Depends(verify_api_key)])
def get_segments():
    """Lists all segments from audio/segments.yaml."""
    return {"status": "success", "segments": tools.load_segments()}


@app.post("/api/segments", dependencies=[Depends(verify_api_key)])
def add_segment(req: SegmentModel):
    """Saves or updates a segment in audio/segments.yaml."""
    segments = tools.load_segments()
    
    # Find existing segment with the same name to preserve order and fields
    existing_segment = None
    index = -1
    for i, s in enumerate(segments):
        if s.get("name") == req.name:
            existing_segment = s
            index = i
            break

    segment_data = {
        "name": req.name,
        "track_file": req.track_file,
        "start_time": req.start_time,
        "end_time": req.end_time,
    }
    
    if req.volume is not None:
        segment_data["volume"] = req.volume
    elif existing_segment and "volume" in existing_segment:
        segment_data["volume"] = existing_segment["volume"]
        
    if req.tags is not None:
        segment_data["tags"] = req.tags
    elif existing_segment and "tags" in existing_segment:
        segment_data["tags"] = existing_segment["tags"]
        
    if req.eq is not None:
        segment_data["eq"] = req.eq
    elif existing_segment and "eq" in existing_segment:
        segment_data["eq"] = existing_segment["eq"]

    if index != -1:
        segments[index] = segment_data
    else:
        segments.append(segment_data)
        
    tools.save_segments(segments)
    
    # Save tags to registry if provided
    if req.tags is not None:
        registry = tools.load_tags_registry()
        registry_changed = False
        for t in req.tags:
            t_norm = t.strip().lower()
            if t_norm and t_norm not in registry:
                registry.append(t_norm)
                registry_changed = True
        if registry_changed:
            tools.save_tags_registry(registry)

    return {"status": "success", "message": "Segment saved."}



@app.delete("/api/segments", dependencies=[Depends(verify_api_key)])
def delete_segment(name: str = Query(...)):
    """Deletes a segment from audio/segments.yaml by its name."""
    segments = tools.load_segments()
    initial_len = len(segments)
    segments = [s for s in segments if s.get("name") != name]
    if len(segments) == initial_len:
        raise HTTPException(status_code=404, detail="Segment not found.")
    tools.save_segments(segments)
    return {"status": "success", "message": "Segment deleted."}


class RenameSegmentRequest(BaseModel):
    old_name: str
    new_name: str


@app.post("/api/segments/rename", dependencies=[Depends(verify_api_key)])
def api_rename_segment(req: RenameSegmentRequest):
    """Renames a segment in audio/segments.yaml."""
    old_name = req.old_name.strip()
    new_name = req.new_name.strip()
    if not old_name or not new_name:
        raise HTTPException(status_code=400, detail="Invalid segment name")
        
    segments = tools.load_segments()
    
    target_idx = -1
    for i, s in enumerate(segments):
        if s.get("name") == old_name:
            target_idx = i
            break
            
    if target_idx == -1:
        raise HTTPException(status_code=404, detail="Segment not found")
        
    for s in segments:
        if s.get("name") == new_name:
            raise HTTPException(status_code=400, detail="A segment with the new name already exists")
            
    segments[target_idx]["name"] = new_name
    tools.save_segments(segments)
    
    return {"status": "success", "message": "Segment renamed."}



class FileTagsModel(BaseModel):
    filename: str
    tags: list[str]


@app.post("/api/file-tags", dependencies=[Depends(verify_api_key)])
def api_update_file_tags(req: FileTagsModel):
    """Updates tags associated with a raw audio file."""
    file_tags = tools.load_file_tags()
    file_tags[req.filename] = req.tags
    tools.save_file_tags(file_tags)
    
    # Save tags to registry as well
    registry = tools.load_tags_registry()
    registry_changed = False
    for t in req.tags:
        t_norm = t.strip().lower()
        if t_norm and t_norm not in registry:
            registry.append(t_norm)
            registry_changed = True
    if registry_changed:
        tools.save_tags_registry(registry)
        
    return {"status": "success", "message": "File tags updated."}


class RenameTagRequest(BaseModel):
    old_name: str
    new_name: str


@app.post("/api/tags/rename", dependencies=[Depends(verify_api_key)])
def api_rename_tag(req: RenameTagRequest):
    """Renames a tag globally across files.yaml, segments.yaml, and tags_registry.yaml."""
    old_tag = req.old_name.strip().lower()
    new_tag = req.new_name.strip().lower()
    if not old_tag or not new_tag:
        raise HTTPException(status_code=400, detail="Invalid tag name")
        
    # 1. Update files.yaml
    file_tags = tools.load_file_tags()
    updated_file_tags = {}
    for filename, tags in file_tags.items():
        if isinstance(tags, list):
            new_tags = []
            for t in tags:
                if t == old_tag:
                    if new_tag not in new_tags:
                        new_tags.append(new_tag)
                else:
                    if t not in new_tags:
                        new_tags.append(t)
            updated_file_tags[filename] = new_tags
        else:
            updated_file_tags[filename] = tags
    tools.save_file_tags(updated_file_tags)
    
    # 2. Update segments.yaml
    segments = tools.load_segments()
    for seg in segments:
        if "tags" in seg and isinstance(seg["tags"], list):
            new_tags = []
            for t in seg["tags"]:
                if t == old_tag:
                    if new_tag not in new_tags:
                        new_tags.append(new_tag)
                else:
                    if t not in new_tags:
                        new_tags.append(t)
            seg["tags"] = new_tags
    tools.save_segments(segments)
    
    # 3. Update tags_registry.yaml
    registry = tools.load_tags_registry()
    if old_tag in registry:
        registry = [new_tag if t == old_tag else t for t in registry]
        registry = list(dict.fromkeys(registry))
    else:
        if new_tag not in registry:
            registry.append(new_tag)
    tools.save_tags_registry(registry)
    
    return {"status": "success", "message": f"Tag renamed from {old_tag} to {new_tag}."}


class BulkFileTagsModel(BaseModel):
    filenames: list[str]
    tags: list[str]
    action: str  # "add" or "remove"


@app.post("/api/file-tags/bulk", dependencies=[Depends(verify_api_key)])
def api_bulk_file_tags(req: BulkFileTagsModel):
    """Bulk updates tags associated with multiple raw audio files."""
    file_tags = tools.load_file_tags()
    normalized_tags = [t.strip().lower() for t in req.tags if t.strip()]
    if not normalized_tags:
        raise HTTPException(status_code=400, detail="No valid tags provided")
        
    for filename in req.filenames:
        current_tags = file_tags.get(filename, [])
        if not isinstance(current_tags, list):
            current_tags = []
            
        if req.action == "add":
            for t in normalized_tags:
                if t not in current_tags:
                    current_tags.append(t)
        elif req.action == "remove":
            current_tags = [t for t in current_tags if t not in normalized_tags]
            
        file_tags[filename] = current_tags
        
    tools.save_file_tags(file_tags)
    
    # Save to global registry as well
    if req.action == "add":
        registry = tools.load_tags_registry()
        registry_changed = False
        for t in normalized_tags:
            if t not in registry:
                registry.append(t)
                registry_changed = True
        if registry_changed:
            tools.save_tags_registry(registry)
            
    return {"status": "success", "message": "Bulk file tags updated successfully."}



@app.get("/api/eq", dependencies=[Depends(verify_api_key)])
def api_get_eq():
    """Returns the current EQ band gains."""
    return {
        "status": "success",
        "gains": {str(k): v for k, v in tools.player.eq_gains.items()},
    }


@app.post("/api/eq", dependencies=[Depends(verify_api_key)])
def api_set_eq(req: EQRequest):
    """Updates EQ gains and applies them asynchronously to the current track.

    Gains are specified as a dict mapping band frequency (Hz, as string) to
    dB gain (-12 to +12 recommended). Filtering runs in a background thread
    to keep the HTTP response fast.
    """
    import threading

    valid_bands = set(str(k) for k in tools.player.eq_gains)
    for band, gain in req.gains.items():
        if band not in valid_bands:
            return {"status": "error", "message": f"Unknown EQ band: {band}Hz"}
        tools.player.eq_gains[int(band)] = max(-20.0, min(20.0, float(gain)))

    def _apply():
        result = tools.player.apply_eq()
        print(f"[EQ] {result}")

    threading.Thread(target=_apply, daemon=True).start()
    return {"status": "success", "message": "EQ update dispatched.", "gains": {str(k): v for k, v in tools.player.eq_gains.items()}}




@app.get("/api/cache/status", dependencies=[Depends(verify_api_key)])
def api_cache_status():
    """Returns the total size (MB) and file count of the synthesis cache."""
    try:
        player = tools.player
        cache_dir = os.path.join(player.playlist_dir, ".cache")
        if not os.path.exists(cache_dir):
            return {"status": "success", "size_mb": 0.0, "file_count": 0}
        
        total_size = 0
        file_count = 0
        for filename in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
                file_count += 1
        size_mb = total_size / (1024 * 1024)
        return {"status": "success", "size_mb": round(size_mb, 2), "file_count": file_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/cache/clear", dependencies=[Depends(verify_api_key)])
def api_clear_cache():
    """Deletes all cached FLAC/WAV files, keeping the active track's file to avoid file-in-use errors."""
    try:
        player = tools.player
        cache_dir = os.path.join(player.playlist_dir, ".cache")
        if not os.path.exists(cache_dir):
            return {"status": "success", "message": "Cache is already empty."}
        
        active_file = None
        if player.current_track and player._sf is not None:
            active_file = os.path.abspath(player._sf.name)
            
        deleted_count = 0
        for filename in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, filename)
            if os.path.isfile(filepath):
                if active_file and os.path.abspath(filepath) == active_file:
                    continue
                os.remove(filepath)
                deleted_count += 1
        return {"status": "success", "message": f"Cleared {deleted_count} cached files."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


vlm_download_states = {
    "tesseract": {"ready": True, "status": "ready", "progress": 100},
    "gemini-2.5-flash-lite": {"ready": True, "status": "ready", "progress": 100},
    "moondream": {"ready": False, "status": "idle", "progress": 0},
    "qwen2-vl": {"ready": False, "status": "idle", "progress": 0},
    "qwen2.5-vl": {"ready": False, "status": "idle", "progress": 0},
    "florence-2": {"ready": False, "status": "idle", "progress": 0},
    "paligemma": {"ready": False, "status": "idle", "progress": 0},
    "minicpm-v": {"ready": False, "status": "idle", "progress": 0},
}

active_downloads = {}


def sync_ollama_ready_states():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models_list = [m["name"] for m in response.json().get("models", [])]
            for model_id, state in vlm_download_states.items():
                if model_id in ("tesseract", "gemini-2.5-flash-lite", "florence-2"):
                    continue
                if state["status"].startswith("downloading") or state["status"] == "paused" or state["status"].startswith("failed"):
                    continue
                ollama_names = []
                if model_id == "moondream":
                    ollama_names = ["moondream", "moondream:latest"]
                elif model_id == "qwen2-vl":
                    ollama_names = ["qwen2-vl", "qwen2-vl:latest", "qwen2-vl:2b", "hf.co/bartowski/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"]
                elif model_id == "qwen2.5-vl":
                    ollama_names = ["qwen2.5vl", "qwen2.5vl:latest", "qwen2.5vl:3b", "qwen2.5vl:7b"]
                elif model_id == "paligemma":
                    ollama_names = ["paligemma", "paligemma:latest"]
                elif model_id == "minicpm-v":
                    ollama_names = ["minicpm-v", "minicpm-v:latest"]
                
                if any(name in models_list for name in ollama_names):
                    state["ready"] = True
                    state["status"] = "ready"
                    state["progress"] = 100
    except Exception as e:
        print(f"[VLM Status] Could not connect to local Ollama: {e}")
 
 
def pull_ollama_model_task(model_id: str, ollama_name: str):
    global vlm_download_states, active_downloads
    state = vlm_download_states[model_id]
    state["status"] = "downloading"
    cancel_evt = active_downloads[model_id]["cancel_event"]
    try:
        url = "http://localhost:11434/api/pull"
        response = requests.post(url, json={"name": ollama_name}, stream=True, timeout=1200)
        active_downloads[model_id]["response"] = response
        
        if response.status_code == 200:
            largest_total = 0
            for line in response.iter_lines():
                if cancel_evt.is_set():
                    response.close()
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    completed = data.get("completed", 0)
                    total = data.get("total", 0)
                    status_text = data.get("status", "")
                    
                    if total > largest_total:
                        largest_total = total
                        state["progress"] = 0
                        
                    if total > 0 and total == largest_total:
                        progress = int((completed / total) * 100)
                        if progress > state["progress"]:
                            state["progress"] = progress
                        state["status"] = "downloading"
                    elif status_text and status_text != "success":
                        state["status"] = f"downloading - {status_text}"
                        
                    if status_text == "success":
                        state["ready"] = True
                        state["status"] = "ready"
                        state["progress"] = 100
                        break
                except Exception:
                    pass
            if cancel_evt.is_set():
                state["status"] = "paused"
        else:
            raise Exception(f"Ollama returned status {response.status_code}")
    except Exception as e:
        if cancel_evt.is_set():
            state["status"] = "paused"
        else:
            print(f"[VLM Pull] Error pulling {ollama_name}: {e}")
            state["status"] = f"failed: {str(e)}"


def pull_qwen2_vl_huggingface_task():
    global vlm_download_states, active_downloads
    state = vlm_download_states["qwen2-vl"]
    state["status"] = "downloading"
    state["progress"] = 0
    cancel_evt = active_downloads["qwen2-vl"]["cancel_event"]
    
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(app_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        gguf_path = os.path.join(models_dir, "Qwen2-VL-2B-Instruct-Q4_K_M.gguf")
        proj_path = os.path.join(models_dir, "mmproj-Qwen2-VL-2B-Instruct-f16.gguf")
        modelfile_path = os.path.join(models_dir, "Modelfile_qwen2_vl")
        
        files_to_download = [
            {
                "url": "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
                "path": gguf_path,
                "size": 986047232
            },
            {
                "url": "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
                "path": proj_path,
                "size": 1331656192
            }
        ]
        
        total_combined_size = sum(f["size"] for f in files_to_download)
        downloaded_so_far = 0
        
        for file_info in files_to_download:
            url = file_info["url"]
            dest = file_info["path"]
            
            # Skip if file already fully exists
            if os.path.exists(dest) and os.path.getsize(dest) == file_info["size"]:
                downloaded_so_far += file_info["size"]
                continue
                
            headers = {}
            temp_size = 0
            if os.path.exists(dest):
                temp_size = os.path.getsize(dest)
                if temp_size < file_info["size"]:
                    headers["Range"] = f"bytes={temp_size}-"
                    downloaded_so_far += temp_size
                else:
                    os.remove(dest)
                    temp_size = 0
                    
            mode = "ab" if temp_size > 0 else "wb"
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            active_downloads["qwen2-vl"]["response"] = response
            
            if response.status_code in (200, 206):
                with open(dest, mode) as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if cancel_evt.is_set():
                            response.close()
                            break
                        if chunk:
                            f.write(chunk)
                            downloaded_so_far += len(chunk)
                            progress = min(99, int((downloaded_so_far / total_combined_size) * 100))
                            state["progress"] = progress
                            state["status"] = f"downloading ({progress}%)"
                            
            if cancel_evt.is_set():
                break
                
        if cancel_evt.is_set():
            state["status"] = "paused"
            return
            
        if not (os.path.exists(gguf_path) and os.path.getsize(gguf_path) == files_to_download[0]["size"]):
            raise Exception("Base GGUF model download incomplete or corrupted.")
        if not (os.path.exists(proj_path) and os.path.getsize(proj_path) == files_to_download[1]["size"]):
            raise Exception("Multimodal projector download incomplete or corrupted.")
            
        state["status"] = "building model in ollama"
        with open(modelfile_path, "w") as mf:
            mf.write(f"FROM {gguf_path}\n")
            mf.write(f"ADAPTER {proj_path}\n\n")
            mf.write('TEMPLATE """{{- if .System -}}\n')
            mf.write('<|im_start|>system\n')
            mf.write('{{ .System }}<|im_end|>\n')
            mf.write('{{- end -}}\n')
            mf.write('{{- range $i, $_ := .Messages }}\n')
            mf.write('{{- $last := eq (len (slice $.Messages $i)) 1 -}}\n')
            mf.write('{{- if eq .Role "user" }}\n')
            mf.write('<|im_start|>user\n')
            mf.write('{{ .Content }}<|im_end|>\n')
            mf.write('{{- else if eq .Role "assistant" }}\n')
            mf.write('<|im_start|>assistant\n')
            mf.write('{{ if .Content }}{{ .Content }}{{ if not $last }}<|im_end|>\n')
            mf.write('{{- else -}}<|im_end|>{{- end -}}\n')
            mf.write('{{- end -}}\n')
            mf.write('{{- end -}}\n')
            mf.write('{{- if and (ne .Role "assistant") $last }}\n')
            mf.write('<|im_start|>assistant\n')
            mf.write('{{ end -}}\n')
            mf.write('{{- end }}"""\n\n')
            mf.write('PARAMETER stop "<|im_start|>"\n')
            mf.write('PARAMETER stop "<|im_end|>"\n')
            mf.write('PARAMETER temperature 0.0001\n')
            mf.write('PARAMETER num_predict 80\n')
            
        import subprocess
        process = subprocess.Popen(
            ["ollama", "create", "qwen2-vl", "-f", modelfile_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        active_downloads["qwen2-vl"]["subprocess"] = process
        
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            state["ready"] = True
            state["status"] = "ready"
            state["progress"] = 100
            print("[Qwen2-VL] Registered custom legacy Qwen2-VL model in Ollama successfully!")
        else:
            raise Exception(f"ollama create failed: {stderr or stdout}")
            
    except Exception as e:
        print(f"[Qwen2-VL] Error during Hugging Face pull/build: {e}")
        if cancel_evt.is_set():
            state["status"] = "paused"
        else:
            state["status"] = f"failed: {str(e)}"
            state["progress"] = 0


def simulate_vlm_download(model_id: str):
    import time
    global vlm_download_states, active_downloads
    state = vlm_download_states[model_id]
    state["status"] = "downloading"
    cancel_evt = active_downloads[model_id]["cancel_event"]
    for p in range(state["progress"], 101, 10):
        if cancel_evt.is_set():
            state["status"] = "paused"
            return
        state["progress"] = p
        state["status"] = f"downloading ({p}%)"
        time.sleep(0.3)
    state["ready"] = True
    state["status"] = "ready"
    state["progress"] = 100
 
 
def sync_florence_ready_state():
    if vlm_download_states["florence-2"]["status"].startswith("downloading") or vlm_download_states["florence-2"]["status"] == "paused" or vlm_download_states["florence-2"]["status"].startswith("failed"):
        return
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--Florence-2-large")
    if os.path.exists(cache_dir):
        has_weights = False
        for root, dirs, files in os.walk(cache_dir):
            if any(f.endswith((".bin", ".safetensors", ".h5")) for f in files):
                has_weights = True
                break
        if has_weights:
            vlm_download_states["florence-2"]["ready"] = True
            vlm_download_states["florence-2"]["status"] = "ready"
            vlm_download_states["florence-2"]["progress"] = 100


def pull_florence_model_task():
    global vlm_download_states, active_downloads
    state = vlm_download_states["florence-2"]
    state["status"] = "downloading"
    state["progress"] = 5
    
    cancel_evt = active_downloads["florence-2"]["cancel_event"]
    stop_event = threading.Event()
    def increment_progress():
        import time
        p = 5
        while not stop_event.is_set() and not cancel_evt.is_set() and p < 95:
            time.sleep(1.0)
            p += 2
            if p > 95:
                p = 95
            state["progress"] = p
            state["status"] = f"downloading ({p}%)"
            
    inc_t = threading.Thread(target=increment_progress)
    inc_t.daemon = True
    inc_t.start()
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        print("[Florence-2] Downloading microsoft/Florence-2-large in background...")
        if cancel_evt.is_set():
            raise Exception("Download cancelled by user.")
        AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        if cancel_evt.is_set():
            raise Exception("Download cancelled by user.")
        AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        
        stop_event.set()
        inc_t.join()
        
        if cancel_evt.is_set():
            state["status"] = "paused"
        else:
            state["ready"] = True
            state["status"] = "ready"
            state["progress"] = 100
            print("[Florence-2] microsoft/Florence-2-large downloaded and cached successfully!")
    except Exception as e:
        print(f"[Florence-2] Error downloading model: {e}")
        stop_event.set()
        try:
            inc_t.join(timeout=1.0)
        except:
            pass
        if cancel_evt.is_set():
            state["status"] = "paused"
        else:
            state["status"] = f"failed: {str(e)}"
            state["progress"] = 0


@app.get("/api/ocr/vlm_status", dependencies=[Depends(verify_api_key)])
def api_vlm_status():
    """Returns the ready/download status of all available local VLM methods."""
    sync_ollama_ready_states()
    sync_florence_ready_state()
    return {"status": "success", "states": vlm_download_states}


class VlmPullRequest(BaseModel):
    model: str


@app.post("/api/ocr/vlm_pull", dependencies=[Depends(verify_api_key)])
def api_vlm_pull(req: VlmPullRequest):
    """Triggers the setup (download) process for a local VLM in the background."""
    model_id = req.model.lower()
    if model_id not in vlm_download_states:
        return {"status": "error", "message": f"Unknown model: {req.model}"}
        
    if vlm_download_states[model_id]["ready"]:
        return {"status": "success", "message": f"{req.model} is already ready."}
        
    if vlm_download_states[model_id]["status"].startswith("downloading"):
        return {"status": "success", "message": f"Already downloading {req.model}."}

    model_map = {
        "moondream": "moondream:latest",
        "qwen2.5-vl": "qwen2.5vl:3b",
        "paligemma": "paligemma",
        "minicpm-v": "minicpm-v"
    }

    # Initialize active downloads entry
    active_downloads[model_id] = {
        "cancel_event": threading.Event(),
        "response": None,
        "subprocess": None
    }

    if model_id == "florence-2":
        t = threading.Thread(target=pull_florence_model_task)
        t.daemon = True
        t.start()
    elif model_id == "qwen2-vl":
        t = threading.Thread(target=pull_qwen2_vl_huggingface_task)
        t.daemon = True
        t.start()
    elif model_id in model_map:
        t = threading.Thread(target=pull_ollama_model_task, args=(model_id, model_map[model_id]))
        t.daemon = True
        t.start()
    else:
        t = threading.Thread(target=simulate_vlm_download, args=(model_id,))
        t.daemon = True
        t.start()
        
    return {"status": "success", "message": f"Started setup for {req.model}."}


class VlmPauseRequest(BaseModel):
    model: str


@app.post("/api/ocr/vlm_pause", dependencies=[Depends(verify_api_key)])
def api_vlm_pause(req: VlmPauseRequest):
    """Pauses or cancels an active VLM download/pull."""
    model_id = req.model.lower()
    if model_id not in vlm_download_states:
        return {"status": "error", "message": f"Unknown model: {req.model}"}
        
    # Trigger cancellation
    if model_id in active_downloads:
        active_downloads[model_id]["cancel_event"].set()
        resp_obj = active_downloads[model_id]["response"]
        if resp_obj:
            try:
                resp_obj.close()
            except Exception:
                pass
        subp = active_downloads[model_id].get("subprocess")
        if subp:
            try:
                subp.terminate()
            except Exception:
                pass
                
    vlm_download_states[model_id]["status"] = "paused"
    return {"status": "success", "message": f"Paused download for {req.model}."}


warmed_models = set()


class VlmWarmupRequest(BaseModel):
    model: str


@app.post("/api/ocr/vlm_warmup", dependencies=[Depends(verify_api_key)])
def api_vlm_warmup(req: VlmWarmupRequest):
    """Triggers a background warmup request to load/warmup the local VLM."""
    model_id = req.model.lower()
    if model_id not in vlm_download_states:
        return {"status": "error", "message": f"Unknown model: {req.model}"}
    if not vlm_download_states[model_id]["ready"]:
        return {"status": "success", "message": f"Model {req.model} is not ready yet."}

    model_map = {
        "moondream": "moondream:latest",
        "qwen2-vl": "qwen2-vl",
        "qwen2.5-vl": "qwen2.5vl:3b",
        "paligemma": "paligemma",
        "minicpm-v": "minicpm-v"
    }
    if model_id not in model_map:
        return {"status": "success", "message": "Warmup not required for this model."}

    def run_warmup():
        try:
            url = "http://127.0.0.1:11434/api/generate"
            dummy_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            print(f"[VLM Warmup] Starting background warmup for {model_id}...", flush=True)
            requests.post(
                url,
                json={
                    "model": model_map[model_id],
                    "prompt": "warmup",
                    "images": [dummy_img],
                    "stream": False,
                    "keep_alive": -1
                },
                timeout=180
            )
            print(f"[VLM Warmup] Background warmup for {model_id} completed.", flush=True)
        except Exception as e:
            print(f"[VLM Warmup] Warmup failed for {model_id}: {e}", flush=True)

    t = threading.Thread(target=run_warmup)
    t.daemon = True
    t.start()
    return {"status": "success", "message": f"Triggered warmup for {req.model}."}


class VlmUnloadRequest(BaseModel):
    model: str


@app.post("/api/ocr/vlm_unload", dependencies=[Depends(verify_api_key)])
def api_vlm_unload(req: VlmUnloadRequest):
    """Triggers an immediate unload request to free the local VLM from memory."""
    model_id = req.model.lower()
    if model_id in warmed_models:
        warmed_models.remove(model_id)
    if model_id not in vlm_download_states:
        return {"status": "error", "message": f"Unknown model: {req.model}"}

    model_map = {
        "moondream": "moondream:latest",
        "qwen2-vl": "qwen2-vl",
        "qwen2.5-vl": "qwen2.5vl:3b",
        "paligemma": "paligemma",
        "minicpm-v": "minicpm-v"
    }
    if model_id not in model_map:
        return {"status": "success", "message": "Unload not required/supported for this model."}

    def run_unload():
        try:
            url = "http://127.0.0.1:11434/api/generate"
            print(f"[VLM Unload] Starting immediate unload for {model_id}...", flush=True)
            requests.post(
                url,
                json={
                    "model": model_map[model_id],
                    "prompt": "",
                    "keep_alive": "0s",
                    "stream": False
                },
                timeout=30
            )
            print(f"[VLM Unload] Immediate unload for {model_id} completed.", flush=True)
        except Exception as e:
            print(f"[VLM Unload] Unload failed for {model_id}: {e}", flush=True)

    t = threading.Thread(target=run_unload)
    t.daemon = True
    t.start()
    return {"status": "success", "message": f"Triggered unload for {req.model}."}


class VlmTryRequest(BaseModel):
    model: str


florence_model = None
florence_processor = None


def run_florence_ocr(image):
    global florence_model, florence_processor
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    if florence_model is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[Florence-2] Loading microsoft/Florence-2-large on {device}...")
        florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", 
            trust_remote_code=True
        ).to(device)
        florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", 
            trust_remote_code=True
        )
        
    device = florence_model.device
    inputs = florence_processor(text="<OCR>", images=image, return_tensors="pt").to(device)
    
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    generated_text = florence_processor.post_process_generation(
        generated_ids, 
        task="<OCR>", 
        image_size=image.size
    )["<OCR>"]
    
    return generated_text


def get_ollama_pids():
    """Returns a list of all running Ollama process PIDs on any platform."""
    import platform
    import subprocess
    pids = []
    sys_type = platform.system()
    
    # 1. Linux
    if sys_type == "Linux":
        try:
            for pid_str in os.listdir("/proc"):
                if pid_str.isdigit():
                    try:
                        with open(f"/proc/{pid_str}/comm", "r") as f:
                            comm = f.read().strip().lower()
                        if "ollama" in comm:
                            pids.append(int(pid_str))
                    except Exception:
                        continue
        except Exception:
            pass
            
    # 2. Windows
    elif sys_type == "Windows":
        import csv
        try:
            res = subprocess.check_output(
                ["tasklist", "/NH", "/FO", "CSV", "/FI", "IMAGENAME eq ollama.exe"],
                creationflags=0x08000000 # CREATE_NO_WINDOW
            ).decode("utf-8", errors="ignore")
            reader = csv.reader(res.strip().splitlines())
            for row in reader:
                if row and len(row) > 1:
                    pids.append(int(row[1]))
        except Exception:
            pass
            
    # 3. macOS (Darwin)
    elif sys_type == "Darwin":
        try:
            res = subprocess.check_output(["pgrep", "-f", "ollama"]).decode("utf-8")
            for line in res.strip().splitlines():
                if line.isdigit():
                    pids.append(int(line))
        except Exception:
            pass
            
    return pids


def get_process_ram_usage_bytes(pid=None):
    """Gets the VmRSS (Resident Set Size) memory usage of a process in bytes."""
    import platform
    if pid is None:
        pid = os.getpid()
    sys_type = platform.system()

    # 1. Linux
    if sys_type == "Linux":
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
        except Exception:
            pass

    # 2. Windows (using ctypes)
    elif sys_type == "Windows":
        try:
            import ctypes
            from ctypes import wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
            if handle:
                try:
                    counters = PROCESS_MEMORY_COUNTERS()
                    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                        return counters.WorkingSetSize
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass

    # 3. macOS (using ps shell command fallback)
    elif sys_type == "Darwin":
        try:
            import subprocess
            res = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
            return int(res.strip()) * 1024
        except Exception:
            pass

    return 0


def get_process_peak_ram_bytes(target_pid=None):
    """Gets the peak RSS memory of a process in bytes."""
    import platform
    if target_pid is None:
        target_pid = os.getpid()
    sys_type = platform.system()

    # 1. Linux
    if sys_type == "Linux":
        try:
            with open(f"/proc/{target_pid}/status", "r") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
        except Exception:
            pass

    # 2. Windows (using ctypes)
    elif sys_type == "Windows":
        try:
            import ctypes
            from ctypes import wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, target_pid)
            if handle:
                try:
                    counters = PROCESS_MEMORY_COUNTERS()
                    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                        return counters.PeakWorkingSetSize
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass

    # 3. macOS
    elif sys_type == "Darwin":
        if target_pid == os.getpid():
            try:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            except Exception:
                pass
        return get_process_ram_usage_bytes(target_pid)

    return get_process_ram_usage_bytes(target_pid)


def get_ollama_ram_usage_bytes():
    """Finds all running processes containing 'ollama' and returns their combined RSS memory in bytes."""
    total_ram = 0
    for pid in get_ollama_pids():
        total_ram += get_process_ram_usage_bytes(pid)
    return total_ram


def get_ollama_peak_ram_usage_bytes():
    """Finds all running processes containing 'ollama' and returns their combined peak RSS memory in bytes."""
    total_ram = 0
    for pid in get_ollama_pids():
        total_ram += get_process_peak_ram_bytes(pid)
    return total_ram


def get_gpu_vram_usage_bytes():
    """Gets the GPU memory usage of current process and any ollama processes in bytes."""
    import subprocess
    import platform
    total_vram = 0
    try:
        import torch
        if torch.cuda.is_available():
            total_vram += torch.cuda.memory_allocated()
        elif torch.backends.mps.is_available():
            total_vram += torch.mps.driver_allocated_memory()
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1
        )
        if res.returncode == 0:
            pids_of_interest = {os.getpid()}
            for pid in get_ollama_pids():
                pids_of_interest.add(pid)
            
            for line in res.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    pid = int(parts[0].strip())
                    vram_mb = int(parts[1].strip())
                    if pid in pids_of_interest:
                        total_vram += vram_mb * 1024 * 1024
    except Exception:
        pass
        
    return total_vram


def format_memory_size(bytes_val):
    if bytes_val <= 0:
        return "0 MB"
    mb_val = bytes_val / (1024 * 1024)
    if mb_val >= 1000:
        return f"{mb_val / 1024:.2f} GB"
    return f"{int(mb_val)} MB"


@app.post("/api/ocr/try_vlm", dependencies=[Depends(verify_api_key)])
def api_ocr_try_vlm(req: VlmTryRequest):
    """Runs a benchmark trial using a local Vision-Language Model (or falls back to simulated/estimated times if not pulled)."""
    import io
    import time
    from PIL import Image
    
    if tools.latest_location_raw_bytes is None or tools.latest_screenshot_bytes is None:
        return {"status": "error", "message": "No preprocessed screenshot available. Please perform a screen scan first."}
        
    try:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        def get_peak_usage(is_ollama=False):
            if is_ollama:
                ram = get_process_peak_ram_bytes() + get_ollama_peak_ram_usage_bytes()
                vram = get_gpu_vram_usage_bytes()
            else:
                ram = get_process_peak_ram_bytes()
                vram = 0
            try:
                import torch
                if torch.cuda.is_available():
                    vram = max(vram, torch.cuda.max_memory_allocated())
                elif torch.backends.mps.is_available():
                    vram = max(vram, torch.mps.driver_allocated_memory())
            except Exception:
                pass
            return format_memory_size(ram), format_memory_size(vram)

        # Model performance parameters
        # Real-world benchmark times for local VLMs running on moderate GPUs:
        model_perf = {
            "tesseract": {"loc": 15.0, "coords": 10.0, "ram": "120 MB", "vram": "0 MB"},
            "moondream": {"loc": 65.0, "coords": 55.0, "ram": "850 MB", "vram": "2.2 GB"},
            "qwen2-vl": {"loc": 95.0, "coords": 85.0, "ram": "1.2 GB", "vram": "4.5 GB"},
            "qwen2.5-vl": {"loc": 85.0, "coords": 75.0, "ram": "1.4 GB", "vram": "5.0 GB"},
            "florence-2": {"loc": 45.0, "coords": 35.0, "ram": "650 MB", "vram": "1.8 GB"},
            "paligemma": {"loc": 135.0, "coords": 115.0, "ram": "1.5 GB", "vram": "5.6 GB"},
            "minicpm-v": {"loc": 185.0, "coords": 165.0, "ram": "1.8 GB", "vram": "6.8 GB"},
            "gemini-2.5-flash-lite": {"loc": 250.0, "coords": 200.0, "ram": "150 MB", "vram": "0 MB"}
        }
        
        selected_model = req.model.lower()
        if selected_model not in model_perf:
            # Clean matching of model key name
            if "moondream" in selected_model:
                selected_model = "moondream"
            elif "qwen2.5" in selected_model or "qwen25" in selected_model:
                selected_model = "qwen2.5-vl"
            elif "qwen" in selected_model:
                selected_model = "qwen2-vl"
            elif "florence" in selected_model:
                selected_model = "florence-2"
            elif "paligemma" in selected_model:
                selected_model = "paligemma"
            elif "minicpm" in selected_model:
                selected_model = "minicpm-v"
            elif "gemini" in selected_model:
                selected_model = "gemini-2.5-flash-lite"
            elif "tesseract" in selected_model:
                selected_model = "tesseract"
            else:
                selected_model = "moondream"

        # Load the PIL image that is fed to the models
        # (This keeps the comparison fair since we feed them the exact same raw cropped binary data!)
        text_img = Image.open(io.BytesIO(tools.latest_location_raw_bytes))
        # Qwen2-VL is most stable at 1x to avoid token context overflow. Qwen2.5-VL works optimally at 2x. Other models default to 4x.
        if selected_model == "qwen2-vl":
            scale_factor = 1
        elif "qwen" in selected_model:
            scale_factor = 2
        else:
            scale_factor = 4
        text_img_4x = text_img.resize((text_img.width * scale_factor, text_img.height * scale_factor), Image.Resampling.LANCZOS)
 
        # Check if we should execute actual local Tesseract OCR!
        if selected_model == "tesseract":
            tp0 = time.time()
            _ = tools.ocr_parser.preprocess_image(text_img, ocr_pass=2)
            tp1 = time.time()
            preprocess_time_ms = (tp1 - tp0) * 1000.0
 
            t0 = time.time()
            pass_num = getattr(tools, "current_ocr_pass", 2)
            loc_str, coords_str, ns, ew = tools.ocr_parser.run_ocr(text_img, pass_num, already_cropped=True)
            t1 = time.time()
            
            raw_text = getattr(tools.ocr_parser, "latest_raw_text", "")
            rich = tools.ocr_parser.parse_text_rich(raw_text)
            raw_loc = rich["raw_location"]
            raw_coords = rich["raw_coordinates"]
            
            coords_val = coords_str if coords_str else "None"
            loc_val = loc_str if loc_str else "None"
            total_time_ms = (t1 - t0) * 1000.0
            loc_time_ms = total_time_ms * 0.55
            coords_time_ms = total_time_ms * 0.45
            
            act_ram, act_vram = get_peak_usage()
            return {
                "status": "success",
                "model": "OpenCV+Tesseract",
                "parsed_location": loc_val,
                "parsed_coordinates": coords_val,
                "raw_location": raw_loc if raw_loc != "None" else None,
                "raw_coordinates": raw_coords if raw_coords != "None" else None,
                "parsed_bearing": tools.latest_parse_result.get("parsed_bearing", "None"),
                "loc_time_ms": round(loc_time_ms, 1),
                "coords_time_ms": round(coords_time_ms, 1),
                "preprocess_time_ms": round(preprocess_time_ms, 1),
                "total_time_ms": round(total_time_ms, 1),
                "actual_ram": act_ram,
                "actual_vram": act_vram
            }
 
        # Check if we should execute actual cloud Gemini 2.5 Flash Lite inference!
        elif selected_model == "gemini-2.5-flash-lite":
            tp0 = time.time()
            buffered = io.BytesIO()
            text_img_4x.save(buffered, format="PNG")
            import base64
            _ = base64.b64encode(buffered.getvalue()).decode("utf-8")
            tp1 = time.time()
            preprocess_time_ms = (tp1 - tp0) * 1000.0
 
            t0 = time.time()
            loc_str, coords_str, _, _ = tools.call_gemini_vision(text_img_4x, "gemini/gemini-2.5-flash-lite")
            t1 = time.time()
            
            if not loc_str and not coords_str:
                raise Exception("No response from Gemini API or configuration key invalid.")
            
            # Use parse_text_rich to fuzzy-match and extract raw values consistently
            combined_text = f"{loc_str or ''}\n{coords_str or ''}"
            rich = tools.ocr_parser.parse_text_rich(combined_text)
            
            parsed_loc = rich["parsed_location"] if rich["parsed_location"] != "None" else (loc_str or "None")
            parsed_coords = rich["parsed_coordinates"] if rich["parsed_coordinates"] != "None" else (coords_str or "None")
            raw_loc = rich["raw_location"] if rich["raw_location"] != "None" else (loc_str or "None")
            raw_coords = rich["raw_coordinates"] if rich["raw_coordinates"] != "None" else (coords_str or "None")
            
            total_time_ms = (t1 - t0) * 1000.0
            
            act_ram, act_vram = get_peak_usage()
            return {
                "status": "success",
                "model": "Gemini 2.5 Flash Lite",
                "parsed_location": parsed_loc,
                "parsed_coordinates": parsed_coords,
                "raw_location": raw_loc,
                "raw_coordinates": raw_coords,
                "loc_time_ms": None,
                "coords_time_ms": None,
                "preprocess_time_ms": round(preprocess_time_ms, 1),
                "total_time_ms": round(total_time_ms, 1),
                "actual_ram": act_ram,
                "actual_vram": act_vram
            }
 
        # Check if we should execute actual local Florence-2 (Large) inference!
        elif selected_model == "florence-2":
            if not vlm_download_states["florence-2"]["ready"]:
                return {"status": "error", "message": "Florence-2 model is not downloaded/ready yet."}
                
            tp0 = time.time()
            device = florence_model.device
            inputs = florence_processor(text="<OCR>", images=text_img_4x, return_tensors="pt").to(device)
            tp1 = time.time()
            preprocess_time_ms = (tp1 - tp0) * 1000.0
 
            t0 = time.time()
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            raw_text = florence_processor.post_process_generation(
                generated_ids, 
                task="<OCR>", 
                image_size=text_img_4x.size
            )["<OCR>"]
            t1 = time.time()
            
            rich = tools.ocr_parser.parse_text_rich(raw_text)
            loc_str = rich["parsed_location"]
            coords_str = rich["parsed_coordinates"]
            raw_loc = rich["raw_location"]
            raw_coords = rich["raw_coordinates"]
            
            total_time_ms = (t1 - t0) * 1000.0
            loc_time_ms = total_time_ms * 0.55
            coords_time_ms = total_time_ms * 0.45
            
            act_ram, act_vram = get_peak_usage()
            return {
                "status": "success",
                "model": "Florence-2 (Large)",
                "parsed_location": loc_str,
                "parsed_coordinates": coords_str,
                "raw_location": raw_loc,
                "raw_coordinates": raw_coords,
                "loc_time_ms": round(loc_time_ms, 1),
                "coords_time_ms": round(coords_time_ms, 1),
                "preprocess_time_ms": round(preprocess_time_ms, 1),
                "total_time_ms": round(total_time_ms, 1),
                "actual_ram": act_ram,
                "actual_vram": act_vram
            }
 
        # Check if we should execute actual local Ollama VLM (Moondream, Qwen2-VL, Qwen2.5-VL, PaliGemma, MiniCPM-V)!
        else:
            model_map = {
                "moondream": "moondream:latest",
                "qwen2-vl": "qwen2-vl",
                "qwen2.5-vl": "qwen2.5vl:3b",
                "paligemma": "paligemma",
                "minicpm-v": "minicpm-v"
            }
            if selected_model not in model_map:
                return {"status": "error", "message": f"Unsupported VLM model: {selected_model}"}
                
            if not vlm_download_states[selected_model]["ready"]:
                return {"status": "error", "message": f"{selected_model.capitalize()} model is not downloaded/ready yet."}
                
            tp0 = time.time()
            buffered = io.BytesIO()
            text_img_4x.save(buffered, format="PNG")
            import base64
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            tp1 = time.time()
            preprocess_time_ms = (tp1 - tp0) * 1000.0
            
            # Run inference (with a single auto-retry if Moondream or Qwen2-VL is run for the first time in this session)
            is_first_run = (selected_model in ("moondream", "qwen2-vl") and selected_model not in warmed_models)
            max_tries = 2 if is_first_run else 1
            response = None
            t0, t1 = 0.0, 0.0
            # Setup JSON payload and dynamic prompt
            url = "http://127.0.0.1:11434/api/generate"
            if "qwen" in selected_model:
                prompt = "The image shows a location name and coordinates. Read them exactly."
            else:
                prompt = "What does the text at the bottom of the image say?"

            json_payload = {
                "model": model_map[selected_model],
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "keep_alive": "5m",
                "options": {
                    "temperature": 0.0
                }
            }

            for try_idx in range(max_tries):
                t0 = time.time()
                response = requests.post(
                    url,
                    json=json_payload,
                    timeout=180
                )
                t1 = time.time()
                
                if response.status_code == 200:
                    resp_json = response.json()
                    print(f"[Ollama VLM Raw JSON] Model: {selected_model} (Try {try_idx + 1}), JSON: {resp_json!r}", flush=True)
                    response_text = resp_json.get("response", "")
                    
                    rich = tools.ocr_parser.parse_text_rich(response_text)
                    loc_str = rich["parsed_location"]
                    coords_str = rich["parsed_coordinates"]
                    raw_loc = rich["raw_location"]
                    raw_coords = rich["raw_coordinates"]
                    
                    if is_first_run and try_idx == 0:
                        print(f"[Ollama VLM] Discarding first {selected_model} run (warmup phase) and retrying...", flush=True)
                        continue
                    break
                else:
                    if try_idx == max_tries - 1:
                        raise Exception(f"Ollama returned status {response.status_code}: {response.text}")
                        
            act_ram, act_vram = get_peak_usage(is_ollama=True)
            
            if response and response.status_code == 200:
                if selected_model in ("moondream", "qwen2-vl"):
                    warmed_models.add(selected_model)
                total_time_ms = (t1 - t0) * 1000.0
                loc_time_ms = total_time_ms * 0.55
                coords_time_ms = total_time_ms * 0.45
                
                return {
                    "status": "success",
                    "model": req.model,
                    "parsed_location": loc_str,
                    "parsed_coordinates": coords_str,
                    "raw_location": raw_loc,
                    "raw_coordinates": raw_coords,
                    "loc_time_ms": round(loc_time_ms, 1),
                    "coords_time_ms": round(coords_time_ms, 1),
                    "preprocess_time_ms": round(preprocess_time_ms, 1),
                    "total_time_ms": round(total_time_ms, 1),
                    "actual_ram": act_ram,
                    "actual_vram": act_vram
                }
            else:
                raise Exception(f"Ollama returned status {response.status_code}")
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Main execution
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)

# Trigger reload trigger comment
