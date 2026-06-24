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
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
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

    litellm.completion = mock_complete
    litellm.acompletion = mock_acomplete

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
    action: str  # play, stop, volume, scan
    track_file: str | None = None
    volume: float | None = None


class ConfigUpdateRequest(BaseModel):
    minimap_bounds: dict
    transitions: dict
    playlist_directory: str
    model_name: str
    polling_interval: float
    mappings: list
    api_key: str | None = None


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
    """Serves the music player HTML GUI dashboard."""
    gui_file = os.path.join(AGENT_DIR, "app", "gui.html")
    if os.path.exists(gui_file):
        with open(gui_file) as f:
            return f.read()
    return "<h3>Error: gui.html not found!</h3>"


@app.get("/api/status", dependencies=[Depends(verify_api_key)])
def api_status():
    """Returns the current playback status and loaded configuration."""
    return {
        "current_track": tools.player.current_track,
        "volume": tools.player.volume,
        "simulated": tools.player.simulated,
        "config_data": tools.config,
        "latest_parse": tools.latest_parse_result,
    }


@app.post("/api/control", dependencies=[Depends(verify_api_key)])
def api_control(req: ControlRequest):
    """Handles manual player actions (play, stop, set volume, scan screen)."""
    if req.action == "play":
        if req.track_file:
            return tools.play_track(req.track_file)
        return {"status": "error", "message": "track_file is required for play action."}
    elif req.action == "stop":
        return tools.stop_music()
    elif req.action == "volume":
        if req.volume is not None:
            return tools.set_volume(req.volume)
        return {"status": "error", "message": "volume is required for volume action."}
    elif req.action == "scan":
        return tools.check_screen_and_update_music()
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
        tools.player.playlist_dir = new_config.get("playlist_directory", "music")
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


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
