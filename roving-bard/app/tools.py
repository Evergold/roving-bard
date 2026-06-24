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

import base64
import json
import os
from io import BytesIO

import litellm
import yaml

from app.player import LocalOCRParser, SafeMusicPlayer, ScreenGrabber, TrackMapper

# Load configuration
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mapping.yaml"
)


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {
        "minimap_bounds": {"x": 0.8, "y": 0.05, "width": 0.15, "height": 0.15},
        "transitions": {"fade_out_ms": 1500, "fade_in_ms": 1500},
        "playlist_directory": "music",
        "model_name": "gemini/gemini-1.5-flash",
        "polling_interval": 2.0,
        "mappings": [],
        "api_key": None,
    }


config = load_config()

# Initialize core player elements
player = SafeMusicPlayer(playlist_dir=config.get("playlist_directory", "music"))
grabber = ScreenGrabber(bounds_config=config.get("minimap_bounds"))
ocr_parser = LocalOCRParser()
mapper = TrackMapper(mappings=config.get("mappings", []))
# Shared caching for GUI visualization
latest_screenshot_bytes = None
latest_parse_result = {
    "parsed_location": None,
    "parsed_coordinates": None,
    "matched_track": None,
    "action": "stopped",
    "method": "None",
    "timestamp": None,
}


def call_gemini_vision(img, model_name):
    """Fallback vision call using LiteLLM to extract coordinates and location."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = (
        "Analyze this screenshot cropped from a video game's mini-map widget. "
        "Extract the location name (if visible) and the coordinate string (e.g. '19.3N, 70.9W' or '14.9S, 103.1E'). "
        "Return a JSON object with keys:\n"
        "- 'location': string containing the name of the place, or null if not found\n"
        "- 'coordinates': string of coordinates (e.g. '19.3N, 70.9W'), or null if not found\n"
        "Do not include any markdown formatting or extra text outside the JSON object."
    )

    try:
        response = litellm.completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).replace("```", "", 1).strip()
        elif content.startswith("```"):
            content = content.replace("```", "", 1).replace("```", "", 1).strip()

        parsed = json.loads(content)
        # Convert location and coordinates string to matching values
        location = parsed.get("location")
        coordinates = parsed.get("coordinates")

        ns_val, ew_val = None, None
        if coordinates:
            _, _, ns_val, ew_val = ocr_parser.parse_text(f"dummy\n{coordinates}")

        return location, coordinates, ns_val, ew_val
    except Exception as e:
        print(f"Error executing Gemini Vision fallback: {e}")
        return None, None, None, None


# Agent Tools


def check_screen_and_update_music() -> dict:
    """Captures the foreground game screen, parses the location and coordinates,
    matches them against mapping.yaml, and plays the appropriate music track.

    Returns:
        dict containing the extraction result (location, coordinates) and action taken.
    """
    global latest_screenshot_bytes, latest_parse_result
    img = grabber.capture_and_crop()
    if not img:
        return {"status": "error", "message": "Failed to capture screenshot."}

    # Cache image as bytes for GUI
    try:
        buf = BytesIO()
        img.save(buf, format="PNG")
        latest_screenshot_bytes = buf.getvalue()
    except Exception as e:
        print(f"Error caching screenshot: {e}")

    # Step 1: Attempt local OCR
    print("[Pipeline] Attempting local Tesseract OCR...")
    location, coordinates, ns, ew = ocr_parser.run_ocr(img)
    method = "Local OCR"

    # Step 2: Fallback to Gemini Multimodal if OCR failed to get location or coordinates
    if not coordinates or not location:
        print(
            "[Pipeline] Local OCR was inconclusive. Falling back to Gemini Multimodal Vision..."
        )
        model_name = config.get("model_name", "gemini/gemini-1.5-flash")
        gemini_loc, gemini_coord, gemini_ns, gemini_ew = call_gemini_vision(
            img, model_name
        )

        if gemini_coord or gemini_loc:
            location = gemini_loc or location
            coordinates = gemini_coord or coordinates
            ns = gemini_ns if gemini_ns is not None else ns
            ew = gemini_ew if gemini_ew is not None else ew
            method = f"Gemini Vision fallback ({model_name})"

    print(
        f"[Pipeline] Result: Location='{location}', Coordinates='{coordinates}' (via {method})"
    )

    # Step 3: Match against mappings
    track_file = mapper.get_track_for_state(location, ns, ew)

    transitions = config.get("transitions", {"fade_out_ms": 1500, "fade_in_ms": 1500})
    fade_out = transitions.get("fade_out_ms", 1500)
    fade_in = transitions.get("fade_in_ms", 1500)

    # Step 4: Play track
    playback_action = "no_change"
    if track_file:
        success = player.play_track(
            track_file, fade_in_ms=fade_in, fade_out_ms=fade_out
        )
        if success:
            playback_action = f"playing_{track_file}"
        else:
            playback_action = "playback_failed"
    else:
        # No match found - stop playback or keep playing? Let's stop to be safe
        player.stop(fade_out_ms=fade_out)
        playback_action = "stopped"

    import datetime

    latest_parse_result = {
        "parsed_location": location,
        "parsed_coordinates": coordinates,
        "matched_track": track_file,
        "action": playback_action,
        "method": method,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
    }

    return {
        "status": "success",
        "method": method,
        "parsed_location": location,
        "parsed_coordinates": coordinates,
        "matched_track": track_file,
        "action": playback_action,
    }


def play_track(track_file: str) -> dict:
    """Manually plays a specific track from the playlist directory.

    Args:
        track_file: Filename of the track to play (e.g. 'town.wav').

    Returns:
        dict containing status of playback.
    """
    transitions = config.get("transitions", {"fade_out_ms": 1500, "fade_in_ms": 1500})
    fade_out = transitions.get("fade_out_ms", 1500)
    fade_in = transitions.get("fade_in_ms", 1500)

    success = player.play_track(track_file, fade_in_ms=fade_in, fade_out_ms=fade_out)
    if success:
        return {"status": "success", "message": f"Now playing {track_file}"}
    return {"status": "error", "message": f"Failed to play track {track_file}"}


def stop_music() -> dict:
    """Stops the current music playback with a smooth fade-out.

    Returns:
        dict containing status of playback.
    """
    transitions = config.get("transitions", {"fade_out_ms": 1500, "fade_in_ms": 1500})
    fade_out = transitions.get("fade_out_ms", 1500)
    player.stop(fade_out_ms=fade_out)
    return {"status": "success", "message": "Music stopped."}


def set_volume(volume: float) -> dict:
    """Sets the player volume.

    Args:
        volume: Volume level as a float between 0.0 (silent) and 1.0 (maximum).

    Returns:
        dict containing status.
    """
    player.set_volume(volume)
    return {"status": "success", "message": f"Volume set to {int(volume * 100)}%"}


def get_playback_status() -> dict:
    """Returns the current playback status, active track, volume level, and configuration.

    Returns:
        dict with playback status details.
    """
    return {
        "status": "success",
        "current_track": player.current_track,
        "volume": player.volume,
        "simulated": player.simulated,
        "config": {
            "playlist_directory": config.get("playlist_directory"),
            "polling_interval": config.get("polling_interval"),
            "model_name": config.get("model_name"),
            "minimap_bounds": config.get("minimap_bounds"),
        },
    }
