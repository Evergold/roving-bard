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


def test_player_transition_delay(tmp_path) -> None:
    """Test that transitioning from an active track to a new track incurs the fadeout delay."""
    import time

    # Create two dummy track files
    track1 = tmp_path / "track1.mp3"
    track1.write_bytes(b"dummy")
    track2 = tmp_path / "track2.mp3"
    track2.write_bytes(b"dummy")

    player = SafeMusicPlayer(playlist_dir=str(tmp_path))
    player.simulated = True

    # Start play of track1 (no delay because no previous track)
    start_time = time.time()
    assert player.play_track("track1.mp3", fade_in_ms=100, fade_out_ms=100) is True
    duration_1 = time.time() - start_time
    assert duration_1 < 0.05  # should be virtually instant
    assert player.current_track == "track1.mp3"

    # Transition to track2 (should delay by fade_out_ms = 200ms)
    start_time = time.time()
    assert player.play_track("track2.mp3", fade_in_ms=100, fade_out_ms=200) is True
    duration_2 = time.time() - start_time
    assert duration_2 >= 0.18  # should sleep for approximately 200ms
    assert player.current_track == "track2.mp3"


def test_player_transition_no_delay_when_paused(tmp_path) -> None:
    """Test that transitioning from a paused active track to a new track is immediate."""
    import time

    # Create two dummy track files
    track1 = tmp_path / "track1.mp3"
    track1.write_bytes(b"dummy")
    track2 = tmp_path / "track2.mp3"
    track2.write_bytes(b"dummy")

    player = SafeMusicPlayer(playlist_dir=str(tmp_path))
    player.simulated = True

    # Start play of track1
    assert player.play_track("track1.mp3", fade_in_ms=100, fade_out_ms=100) is True
    assert player.current_track == "track1.mp3"

    # Pause the player
    assert player.pause() is True
    assert player.paused is True

    # Transition to track2 (should NOT delay despite fade_out_ms = 200ms)
    start_time = time.time()
    assert player.play_track("track2.mp3", fade_in_ms=100, fade_out_ms=200) is True
    duration = time.time() - start_time
    assert duration < 0.05  # should be virtually instant
    assert player.current_track == "track2.mp3"
    assert player.paused is False


def test_player_transition_no_delay_when_fadeout_zero(tmp_path) -> None:
    """Test that transitioning from a playing active track to a new track is immediate when fade_out_ms=0."""
    import time

    # Create two dummy track files
    track1 = tmp_path / "track1.mp3"
    track1.write_bytes(b"dummy")
    track2 = tmp_path / "track2.mp3"
    track2.write_bytes(b"dummy")

    player = SafeMusicPlayer(playlist_dir=str(tmp_path))
    player.simulated = True

    # Start play of track1
    assert player.play_track("track1.mp3", fade_in_ms=100, fade_out_ms=100) is True
    assert player.current_track == "track1.mp3"

    # Transition to track2 with fade_out_ms=0 (should NOT delay)
    start_time = time.time()
    assert player.play_track("track2.mp3", fade_in_ms=100, fade_out_ms=0) is True
    duration = time.time() - start_time
    assert duration < 0.05  # should be virtually instant
    assert player.current_track == "track2.mp3"
    assert player.paused is False


def test_player_stop_restarts_from_beginning() -> None:
    """Test that stop sets the was_stopped flag and resume resets it."""
    player = SafeMusicPlayer(playlist_dir="music")
    player.simulated = True
    player.current_track = "mock_track.mp3"
    player.paused = False

    # Stop the player (does not clear current_track, sets paused and was_stopped to True)
    player.stop()
    assert player.current_track == "mock_track.mp3"
    assert player.paused is True
    assert player.was_stopped is True

    # Resume the player (resets was_stopped, unpauses)
    assert player.resume() is True
    assert player.paused is False
    assert player.was_stopped is False


def test_api_stop_control() -> None:
    """Test POST /api/control with stop action, and status representation."""
    headers = get_headers()

    # Reset player state
    tools.player.simulated = True
    tools.player.current_track = "test_track.mp3"
    tools.player.paused = False
    tools.player.was_stopped = False

    # Send stop command
    stop_response = client.post(
        "/api/control",
        headers=headers,
        json={"action": "stop"},
    )
    assert stop_response.status_code == 200
    assert stop_response.json() == {"status": "success", "message": "Music stopped."}

    # Verify status after stop
    status_response = client.get("/api/status", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["paused"] is True
    assert status_response.json()["was_stopped"] is True

    # Send resume command
    resume_response = client.post(
        "/api/control",
        headers=headers,
        json={"action": "resume"},
    )
    assert resume_response.status_code == 200

    # Verify status after resume
    status_response = client.get("/api/status", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["paused"] is False
    assert status_response.json()["was_stopped"] is False


def test_player_seek() -> None:
    """Test seeking functionality on SafeMusicPlayer."""
    player = SafeMusicPlayer(playlist_dir="music")
    player.simulated = True
    player.current_track = "mock_track.mp3"
    player.track_duration = 180.0
    player.paused = False

    # Seek to 45.0s
    assert player.seek(45.0) is True
    assert player.paused is False
    assert player.was_stopped is False
    assert player.get_current_position() >= 45.0

    # Test seek bound limits
    assert player.seek(-10.0) is True
    assert player.last_seek_position == 0.0

    assert player.seek(300.0) is True
    assert player.last_seek_position == 180.0


def test_api_seek_control() -> None:
    """Test POST /api/control with seek action, and status representations."""
    headers = get_headers()

    # Reset player state
    tools.player.simulated = True
    tools.player.current_track = "test_track.mp3"
    tools.player.track_duration = 180.0
    tools.player.paused = False

    # Send seek command
    seek_response = client.post(
        "/api/control",
        headers=headers,
        json={"action": "seek", "position": 60.0},
    )
    assert seek_response.status_code == 200
    assert seek_response.json()["status"] == "success"

    # Verify status includes correct position and duration
    status_response = client.get("/api/status", headers=headers)
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert "duration" in status_data
    assert "current_position" in status_data
    assert status_data["duration"] == 180.0
    assert status_data["current_position"] >= 60.0




