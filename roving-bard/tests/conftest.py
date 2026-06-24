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

os.environ.setdefault("AGENT_API_KEY", "dev-api-key-12345")

import litellm
import pytest
from litellm.utils import ModelResponse, ModelResponseStream


@pytest.fixture(autouse=True)
def mock_litellm_completion(monkeypatch):
    """Automatically mocks all litellm.completion and acompletion calls for safety and offline testing."""

    def mock_complete(*args, **kwargs):
        # Determine content to return based on requested format
        response_format = kwargs.get("response_format")
        if response_format and response_format.get("type") == "json_object":
            content = '{"location": "Town", "coordinates": "19.3N, 70.9W"}'
        else:
            content = "This is a mock assistant response from the Game-Aware Music Player Agent."

        # Return an actual litellm ModelResponse object
        return ModelResponse(
            choices=[
                {
                    "message": {"content": content, "role": "assistant"},
                    "finish_reason": "stop",
                }
            ]
        )

    async def mock_acomplete(*args, **kwargs):
        # Return an async generator of ModelResponseStream objects to simulate streaming
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

    monkeypatch.setattr(litellm, "completion", mock_complete)
    monkeypatch.setattr(litellm, "acompletion", mock_acomplete)
