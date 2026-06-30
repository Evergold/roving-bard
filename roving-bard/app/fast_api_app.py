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

from dotenv import load_dotenv

load_dotenv()

import google.auth
import yaml
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
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


class EQRequest(BaseModel):
    gains: dict[str, float]  # e.g. {"32": 3.0, "1000": -2.5, ...}


def verify_api_key(
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(default=None),
):
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
    }


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
        tools.grabber.bounds = new_config.get("minimap_bounds")
        tools.mapper.mappings = new_config.get("mappings", [])

        return {
            "status": "success",
            "message": "Configuration hot-reloaded successfully.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def initialize_simulation_screen():
    """Loads the first test screen on startup if in simulation mode."""
    import os
    from PIL import Image
    
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
        filename = test_files[0]
        filepath = os.path.join(capture_dir, filename)
        try:
            full_img = Image.open(filepath).convert("RGB")
            print(f"[ScreenGrabber] Init Simulation: Loaded {filename}")
            
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
    from PIL import Image
    
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
            
        filename = test_files[tools.grabber.test_index]
        filepath = os.path.join(capture_dir, filename)
        try:
            full_img = Image.open(filepath).convert("RGB")
            print(f"[ScreenGrabber] Simulation Mode: Loaded {filename} (Index: {tools.grabber.test_index})")
        except Exception as e:
            return {"status": "error", "message": f"Failed to load test screen: {e}"}
    else:
        # Fallback to live capture
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


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

# Trigger reload trigger comment
