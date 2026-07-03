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

import warnings
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")

import os

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm

from app.tools import (
    check_screen_and_update_music,
    config,
    get_playback_status,
    play_track,
    set_volume,
    stop_music,
)

# Local setup - avoiding GCP authentication and Vertex AI method
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "mock-project-id")
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
os.environ["ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS"] = "true"

# Resolve model from configuration (mapping.yaml)
# LiteLLM format: e.g. "gemini/gemini-2.5-flash-lite", "openai/gpt-4o", etc.
model_name = config.get("model_name", "gemini/gemini-2.5-flash-lite")
if "gemini-1.5-flash" in model_name or "gemini-2.5-flash" in model_name:
    if "lite" not in model_name:
        model_name = "gemini/gemini-2.5-flash-lite"

root_agent = Agent(
    name="roving_bard",
    model=LiteLlm(model=model_name),
    instruction=(
        "You are the Roving Bard, a smart assistant that controls "
        "background music playback for a video game running in the foreground.\n\n"
        "You have access to tools that can:\n"
        "- Capture and parse the game screen (Tesseract OCR + Gemini Vision fallback) to "
        "update music dynamically according to mapping.yaml (use `check_screen_and_update_music`).\n"
        "- Play specific tracks directly (use `play_track`).\n"
        "- Stop playback (use `stop_music`).\n"
        "- Set player volume (use `set_volume`).\n"
        "- Retrieve current playback status and active track (use `get_playback_status`).\n\n"
        "Politely answer questions about the music player, help the user play/stop music, "
        "or trigger automatic updates from the screen."
    ),
    tools=[
        check_screen_and_update_music,
        play_track,
        stop_music,
        set_volume,
        get_playback_status,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
