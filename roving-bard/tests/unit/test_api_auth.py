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
    response = client.get("/api/status", headers={"X-API-Key": "dev-api-key-12345"})
    assert response.status_code == 200
    assert "current_track" in response.json()
