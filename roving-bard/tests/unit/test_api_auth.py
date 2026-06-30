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

import os
import pytest
from fastapi.testclient import TestClient

from app.fast_api_app import app

client = TestClient(app)


def test_api_status_unauthorized() -> None:
    """Ensure status endpoint requires an API key and returns 401 on failure."""
    response = client.get("/api/status")
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["detail"]


def test_api_status_authorized() -> None:
    """Ensure status endpoint succeeds when correct API key is passed."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("AGENT_API_KEY")
    if not api_key:
        pytest.skip("No API key (GEMINI_API_KEY, GOOGLE_API_KEY, or AGENT_API_KEY) found in environment.")
    response = client.get("/api/status", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert "current_track" in response.json()


def test_verify_api_key_localhost_bypass() -> None:
    """Ensure verify_api_key bypasses verification for local client hosts."""
    from unittest.mock import MagicMock
    from fastapi import Request
    from app.fast_api_app import verify_api_key

    mock_request = MagicMock(spec=Request)
    mock_request.client = MagicMock()

    for local_host in ("127.0.0.1", "::1", "localhost"):
        mock_request.client.host = local_host
        assert verify_api_key(request=mock_request) == "localhost"

