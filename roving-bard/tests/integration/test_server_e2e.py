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

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8002"
STREAM_URL = BASE_URL + "/run_sse"
FEEDBACK_URL = BASE_URL + "/feedback"

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("AGENT_API_KEY") or ""
HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}


def log_output(pipe: Any, log_func: Any) -> None:
    """Log the output from the given pipe."""
    import sys

    for line in iter(pipe.readline, ""):
        log_func(line.strip())
        sys.stdout.write(f"[Server Log] {line}")
        sys.stdout.flush()


def start_server() -> subprocess.Popen[str]:
    """Start the FastAPI server using subprocess and log its output."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.fast_api_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8002",
    ]
    env = os.environ.copy()
    env["INTEGRATION_TEST"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    # Start threads to log stdout and stderr in real-time
    threading.Thread(
        target=log_output, args=(process.stdout, logger.info), daemon=True
    ).start()
    threading.Thread(
        target=log_output, args=(process.stderr, logger.error), daemon=True
    ).start()

    return process


def wait_for_server(timeout: int = 90, interval: int = 1) -> bool:
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/docs", timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready")
                return True
        except RequestException:
            pass
        time.sleep(interval)
    logger.error(f"Server did not become ready within {timeout} seconds")
    return False


@pytest.fixture(scope="session")
def server_fixture(request: Any) -> Iterator[subprocess.Popen[str]]:
    """Pytest fixture to start and stop the server for testing."""
    if not API_KEY:
        pytest.skip("Skipping integration tests: no API key (GEMINI_API_KEY, GOOGLE_API_KEY, or AGENT_API_KEY) found in the environment.")
    logger.info("Starting server process")
    server_process = start_server()
    if not wait_for_server():
        pytest.fail("Server failed to start")
    logger.info("Server process started")

    def stop_server() -> None:
        logger.info("Stopping server process")
        server_process.terminate()
        server_process.wait()
        logger.info("Server process stopped")

    request.addfinalizer(stop_server)
    yield server_process


def test_chat_stream(server_fixture: subprocess.Popen[str]) -> None:
    """Test the chat stream functionality."""
    logger.info("Starting chat stream test")
    # Create session first
    user_id = "test_user_123"
    session_data = {"state": {"preferred_language": "English", "visit_count": 1}}

    session_url = f"{BASE_URL}/apps/app/users/{user_id}/sessions"
    session_response = requests.post(
        session_url,
        headers=HEADERS,
        json=session_data,
        timeout=60,
    )
    assert session_response.status_code == 200
    logger.info(f"Session creation response: {session_response.json()}")
    session_id = session_response.json()["id"]

    # Then send chat message
    data = {
        "app_name": "app",
        "user_id": user_id,
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": "Hi!"}],
        },
        "streaming": True,
    }
    response = requests.post(
        STREAM_URL, headers=HEADERS, json=data, stream=True, timeout=60
    )
    assert response.status_code == 200

    # Parse SSE events from response
    events = []
    for line in response.iter_lines():
        if line:
            # SSE format is "data: {json}"
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                event_json = line_str[6:]  # Remove "data: " prefix
                event = json.loads(event_json)
                events.append(event)

    print("DEBUG EVENTS:", events)
    assert events, "No events received from stream"
    # Check for valid content in the response
    has_text_content = False
    for event in events:
        content = event.get("content")
        if (
            content is not None
            and content.get("parts")
            and any(part.get("text") for part in content["parts"])
        ):
            has_text_content = True
            break

    assert has_text_content, "Expected at least one event with text content"


def test_chat_stream_error_handling(server_fixture: subprocess.Popen[str]) -> None:
    """Test the chat stream error handling."""
    logger.info("Starting chat stream error handling test")
    data = {
        "input": {"messages": [{"type": "invalid_type", "content": "Cause an error"}]}
    }
    response = requests.post(
        STREAM_URL, headers=HEADERS, json=data, stream=True, timeout=10
    )

    assert response.status_code == 422, (
        f"Expected status code 422, got {response.status_code}"
    )
    logger.info("Error handling test completed successfully")


def test_collect_feedback(server_fixture: subprocess.Popen[str]) -> None:
    """
    Test the feedback collection endpoint (/feedback) to ensure it properly
    logs the received feedback.
    """
    # Create sample feedback data
    feedback_data = {
        "score": 4,
        "user_id": "test-user-456",
        "session_id": "test-session-456",
        "text": "Great response!",
    }

    response = requests.post(
        FEEDBACK_URL, json=feedback_data, headers=HEADERS, timeout=10
    )
    assert response.status_code == 200


def test_vlm_endpoints(server_fixture: subprocess.Popen[str]) -> None:
    """Test the VLM benchmarking and download status endpoints."""
    logger.info("Starting VLM endpoints test")
    
    # 1. Check vlm status initially
    status_url = BASE_URL + "/api/ocr/vlm_status"
    response = requests.get(status_url, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["states"]["tesseract"]["ready"] is True
    
    # 2. Trigger screen scan to cache raw screenshot bytes
    action_url = BASE_URL + "/api/control"
    response = requests.post(action_url, json={"action": "scan"}, headers=HEADERS, timeout=15)
    assert response.status_code == 200
    
    # 3. Test try_vlm with tesseract (since it is ready)
    try_url = BASE_URL + "/api/ocr/try_vlm"
    response = requests.post(try_url, json={"model": "tesseract"}, headers=HEADERS, timeout=15)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["model"] == "OpenCV+Tesseract"
    assert res["parsed_location"] != "Unknown"
    
    # 4. Trigger download/pull for a local VLM (minicpm-v)
    pull_url = BASE_URL + "/api/ocr/vlm_pull"
    response = requests.post(pull_url, json={"model": "minicpm-v"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    response = requests.get(status_url, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    status = response.json()["states"]["minicpm-v"]["status"]
    assert status.startswith("downloading") or "pulling" in status
    
    # 5. Pause the download
    pause_url = BASE_URL + "/api/ocr/vlm_pause"
    response = requests.post(pause_url, json={"model": "minicpm-v"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Check status changed to paused
    response = requests.get(status_url, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["states"]["minicpm-v"]["status"] == "paused"
    
    # 6. Resume/pull again
    response = requests.post(pull_url, json={"model": "minicpm-v"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    response = requests.get(status_url, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    status = response.json()["states"]["minicpm-v"]["status"]
    assert status.startswith("downloading") or "pulling" in status
    
    # Clean up by pausing
    requests.post(pause_url, json={"model": "minicpm-v"}, headers=HEADERS, timeout=10)
    
    # 7. Trigger pull for qwen2.5-vl
    response = requests.post(pull_url, json={"model": "qwen2.5-vl"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    response = requests.get(status_url, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    status = response.json()["states"]["qwen2.5-vl"]["status"]
    assert status.startswith("downloading") or "pulling" in status or status == "ready"
    
    # Pause qwen2.5-vl
    response = requests.post(pause_url, json={"model": "qwen2.5-vl"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify qwen2.5-vl is paused
    response = requests.get(status_url, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["states"]["qwen2.5-vl"]["status"] == "paused"
    
    # 8. Test warmup endpoint
    warmup_url = BASE_URL + "/api/ocr/vlm_warmup"
    response = requests.post(warmup_url, json={"model": "tesseract"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "Warmup not required" in response.json()["message"]
    
    response = requests.post(warmup_url, json={"model": "moondream"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # 9. Test unload endpoint
    unload_url = BASE_URL + "/api/ocr/vlm_unload"
    response = requests.post(unload_url, json={"model": "moondream"}, headers=HEADERS, timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "Triggered unload" in response.json()["message"]
    
    logger.info("VLM endpoints test completed successfully")
