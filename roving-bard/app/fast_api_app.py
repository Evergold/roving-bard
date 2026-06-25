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


@app.get("/gui", response_class=HTMLResponse)
def get_gui():
    """Serves the music player HTML GUI dashboard with embedded API key."""
    gui_file = os.path.join(AGENT_DIR, "app", "gui.html")
    if os.path.exists(gui_file):
        with open(gui_file) as f:
            content = f.read()
        api_key = os.getenv("AGENT_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        content = content.replace("{{API_KEY_PLACEHOLDER}}", api_key)
        return content
    return "<h3>Error: gui.html not found!</h3>"


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


@app.get("/api/screenshot", dependencies=[Depends(verify_api_key)])
def api_screenshot():
    """Returns the latest cropped screenshot image, or a transparent placeholder."""
    if tools.latest_screenshot_bytes:
        return Response(content=tools.latest_screenshot_bytes, media_type="image/png")

    # 1x1 transparent PNG fallback
    transparent_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    return Response(content=transparent_png, media_type="image/png")


@app.post("/api/screenshot/refresh", dependencies=[Depends(verify_api_key)])
def api_screenshot_refresh():
    """Reloads the current capture from disk/screen and updates latest_screenshot_bytes without running OCR."""
    full_img = tools.grabber.capture_full()
    if not full_img:
        return {"status": "error", "message": "Failed to capture/load screenshot."}
    try:
        from io import BytesIO
        buf = BytesIO()
        full_img.save(buf, format="PNG")
        tools.latest_screenshot_bytes = buf.getvalue()
        return {"status": "success", "message": "Screenshot reloaded successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/audio-files", dependencies=[Depends(verify_api_key)])
def api_list_audio_files():
    """Lists all audio files in the playlist directory."""
    playlist_dir = tools.player.playlist_dir
    os.makedirs(playlist_dir, exist_ok=True)
    files = [
        f for f in os.listdir(playlist_dir)
        if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".abc", ".mp4"))
    ]
    file_tags = tools.load_file_tags()
    return {"status": "success", "files": sorted(files), "file_tags": file_tags}


@app.post("/api/upload-audio", dependencies=[Depends(verify_api_key)])
def api_upload_audio(file: UploadFile = File(...)):
    """Uploads an audio file and saves it to the playlist directory."""
    playlist_dir = tools.player.playlist_dir
    os.makedirs(playlist_dir, exist_ok=True)
    
    filename = os.path.basename(file.filename)
    if not filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".abc", ".mp4")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio file format."
        )
        
    filepath = os.path.join(playlist_dir, filename)
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


@app.get("/api/segments", dependencies=[Depends(verify_api_key)])
def get_segments():
    """Lists all segments from music/segments.yaml."""
    return {"status": "success", "segments": tools.load_segments()}


@app.post("/api/segments", dependencies=[Depends(verify_api_key)])
def add_segment(req: SegmentModel):
    """Saves or updates a segment in music/segments.yaml."""
    segments = tools.load_segments()
    # Check if duplicate name, overwrite it
    segments = [s for s in segments if s.get("name") != req.name]
    segment_data = {
        "name": req.name,
        "track_file": req.track_file,
        "start_time": req.start_time,
        "end_time": req.end_time,
    }
    if req.volume is not None:
        segment_data["volume"] = req.volume
    if req.tags is not None:
        segment_data["tags"] = req.tags
    segments.append(segment_data)
    tools.save_segments(segments)
    return {"status": "success", "message": "Segment saved."}


@app.delete("/api/segments", dependencies=[Depends(verify_api_key)])
def delete_segment(name: str = Query(...)):
    """Deletes a segment from music/segments.yaml by its name."""
    segments = tools.load_segments()
    initial_len = len(segments)
    segments = [s for s in segments if s.get("name") != name]
    if len(segments) == initial_len:
        raise HTTPException(status_code=404, detail="Segment not found.")
    tools.save_segments(segments)
    return {"status": "success", "message": "Segment deleted."}


class FileTagsModel(BaseModel):
    filename: str
    tags: list[str]


@app.post("/api/file-tags", dependencies=[Depends(verify_api_key)])
def api_update_file_tags(req: FileTagsModel):
    """Updates tags associated with a raw audio file."""
    file_tags = tools.load_file_tags()
    file_tags[req.filename] = req.tags
    tools.save_file_tags(file_tags)
    return {"status": "success", "message": "File tags updated."}


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


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
