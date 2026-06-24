# Copyright 2026 Google LLC
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

import os
import pytest
from fastapi.testclient import TestClient

from app.fast_api_app import app
from app import tools
from app.player import SafeMusicPlayer

client = TestClient(app)


def get_headers():
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("AGENT_API_KEY")
        or "test-mock-key"
    )
    # Ensure the key is allowed by tools.config
    tools.config["api_key"] = api_key
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def test_player_pause_resume() -> None:
    """Test pause and resume methods on SafeMusicPlayer."""
    player = SafeMusicPlayer(playlist_dir="music")
    player.simulated = True
    player.current_track = "mock_track.mp3"
    player.paused = False

    # Pause the player
    assert player.pause() is True
    assert player.paused is True

    # Resume the player
    assert player.resume() is True
    assert player.paused is False


def test_api_pause_resume_control() -> None:
    """Test POST /api/control with pause and resume actions, and status representation."""
    headers = get_headers()

    # Reset player state
    tools.player.simulated = True
    tools.player.current_track = "test_track.mp3"
    tools.player.paused = False

    # Check initial status
    status_response = client.get("/api/status", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["paused"] is False
    assert status_response.json()["current_track"] == "test_track.mp3"

    # Send pause command
    pause_response = client.post(
        "/api/control",
        headers=headers,
        json={"action": "pause"},
    )
    assert pause_response.status_code == 200
    assert pause_response.json() == {"status": "success", "message": "Playback paused."}

    # Verify status after pause
    status_response = client.get("/api/status", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["paused"] is True

    # Send resume command
    resume_response = client.post(
        "/api/control",
        headers=headers,
        json={"action": "resume"},
    )
    assert resume_response.status_code == 200
    assert resume_response.json() == {"status": "success", "message": "Playback resumed."}

    # Verify status after resume
    status_response = client.get("/api/status", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["paused"] is False
