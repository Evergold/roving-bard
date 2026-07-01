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
import time
from io import BytesIO

import litellm
import yaml

from app.player import LocalOCRParser, SafeMusicPlayer, ScreenGrabber, TrackMapper

# Load configuration
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "mapping.yaml"
)
SEGMENTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "segments.yaml"
)
FILES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "files.yaml"
)
TAGS_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "tags_registry.yaml"
)



def load_segments() -> list:
    if os.path.exists(SEGMENTS_PATH):
        try:
            with open(SEGMENTS_PATH, "r") as f:
                data = yaml.safe_load(f)
                segments = []
                if isinstance(data, dict) and "segments" in data:
                    segments = data["segments"] or []
                elif isinstance(data, list):
                    segments = data
                
                # Check for missing EQ configurations and default to 'flat'
                modified = False
                for s in segments:
                    if s.get("eq") is None:
                        s["eq"] = "flat"
                        modified = True
                
                if modified:
                    save_segments(segments)
                
                return segments
        except Exception as e:
            print(f"Error loading segments.yaml: {e}")
    return []


def save_segments(segments: list):
    try:
        with open(SEGMENTS_PATH, "w") as f:
            yaml.safe_dump({"segments": segments}, f)
    except Exception as e:
        print(f"Error saving segments.yaml: {e}")


def load_file_tags() -> dict:
    if os.path.exists(FILES_PATH):
        try:
            with open(FILES_PATH, "r") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"Error loading files.yaml: {e}")
    return {}


def save_file_tags(file_tags: dict):
    try:
        with open(FILES_PATH, "w") as f:
            yaml.safe_dump(file_tags, f)
    except Exception as e:
        print(f"Error saving files.yaml: {e}")


def load_tags_registry() -> list:
    if not os.path.exists(TAGS_REGISTRY_PATH):
        in_use_tags = set()
        
        # 1. Load tags from files.yaml
        file_tags = load_file_tags()
        for tags in file_tags.values():
            if isinstance(tags, list):
                for t in tags:
                    in_use_tags.add(t.strip().lower())
                    
        # 2. Load tags from segments.yaml
        segs = load_segments()
        for seg in segs:
            tags = seg.get("tags", [])
            if isinstance(tags, list):
                for t in tags:
                    in_use_tags.add(t.strip().lower())
                    
        sorted_tags = sorted(list(in_use_tags))
        save_tags_registry(sorted_tags)
        return sorted_tags
    try:
        with open(TAGS_REGISTRY_PATH, "r") as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "tags" in data:
                return data["tags"] or []
    except Exception as e:
        print(f"Error loading tags_registry.yaml: {e}")
    return []


def save_tags_registry(tags: list):
    try:
        with open(TAGS_REGISTRY_PATH, "w") as f:
            yaml.safe_dump(tags, f)
    except Exception as e:
        print(f"Error saving tags_registry.yaml: {e}")




def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {
        "minimap_bounds": {"x": 0.8, "y": 0.05, "width": 0.15, "height": 0.15},
        "transitions": {"fade_out_ms": 1500, "fade_in_ms": 1500},
        "playlist_directory": "audio",
        "model_name": "gemini/gemini-2.5-flash-lite",
        "polling_interval": 2.0,
        "mappings": [],
        "api_key": None,
    }


config = load_config()

# Initialize core player elements
player = SafeMusicPlayer(playlist_dir=config.get("playlist_directory", "audio"))
player.update_soundfont(config.get("active_soundfont"))
grabber = ScreenGrabber(bounds_config=config.get("minimap_bounds"))
ocr_parser = LocalOCRParser()
mapper = TrackMapper(mappings=config.get("mappings", []))
# Shared caching for GUI visualization
latest_screenshot_bytes = None
latest_full_screenshot_bytes = None
latest_cursor_bytes = None
latest_character_bytes = None
latest_cursor_processed_bytes = None
latest_location_processed_bytes = None
latest_location_raw_bytes = None
latest_character_processed_bytes = None
minimap_detected = False
current_ocr_pass = 2
latest_parse_result = {
    "parsed_location": None,
    "parsed_coordinates": None,
    "parsed_bearing": None,
    "matched_track": None,
    "action": "stopped",
    "method": "None",
    "timestamp": None,
    "loc_time_ms": 0.0,
    "coords_time_ms": 0.0,
    "preprocess_time_ms": 0.0,
    "total_time_ms": 0.0,
    "actual_ram": None,
    "actual_vram": None
}


def call_gemini_vision(img, model_name):
    """Fallback vision call using LiteLLM to extract coordinates and location."""
    if "gemini-1.5-flash" in model_name or "gemini-2.5-flash" in model_name:
        if "lite" not in model_name:
            model_name = "gemini/gemini-2.5-flash-lite"
        
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
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]

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


def get_actual_usage():
    """Get actual RAM and VRAM usage on Linux systems."""
    try:
        import os
        pid = os.getpid()
        ram_mb = 0
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ram_mb = int(parts[1]) // 1024
                    break
        
        vram_mb = 0
        import subprocess
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=1.0
            )
            vram_mb = int(res.stdout.strip())
        except Exception:
            pass
            
        return f"{ram_mb} MB", f"{vram_mb} MB"
    except Exception as e:
        print(f"Error getting actual memory usage: {e}")
        return None, None


# Agent Tools


def check_screen_and_update_music() -> dict:
    """Captures the foreground game screen, parses the location and coordinates,
    matches them against mapping.yaml, and plays the appropriate music track.

    Returns:
        dict containing the extraction result (location, coordinates) and action taken.
    """
    global latest_screenshot_bytes, latest_full_screenshot_bytes, latest_cursor_bytes, latest_character_bytes, latest_cursor_processed_bytes, latest_location_processed_bytes, latest_location_raw_bytes, latest_character_processed_bytes, latest_parse_result, current_ocr_pass
    full_img = grabber.capture_full()
    if not full_img:
        return {"status": "error", "message": "Failed to capture screenshot."}

    # Crop to bounds for OCR processing
    img = grabber.crop_image(full_img)

    # Crop to location and coordinates (bottom 42%) at 1x for OCR
    w, h = img.size
    y_start = int(h * 0.58)
    text_img_1x = img.crop((0, y_start, w, h))

    # Parse bearing from red cursor at the center of the minimap and crop it
    bearing_deg = None
    bearing_str = "None"
    try:
        import numpy as np
        import cv2
        from PIL import Image
        
        # Center of the radar circle (mw x mw square at the top)
        cx = w // 2
        cy = w // 2
        r = int(w * 0.12)
        cursor_crop = img.crop((cx - r, cy - r, cx + r, cy + r))
        
        # Enlarge 3x for preview and high-precision detection
        cursor_crop_3x = cursor_crop.resize((r * 6, r * 6), Image.Resampling.LANCZOS)
        buf_cursor = BytesIO()
        cursor_crop_3x.save(buf_cursor, format="PNG")
        latest_cursor_bytes = buf_cursor.getvalue()
        
        # Parse bearing angle using the scaled up 3x image
        cv_crop = np.array(cursor_crop_3x)
        cv_crop = cv_crop[:, :, ::-1].copy()
        hsv = cv2.cvtColor(cv_crop, cv2.COLOR_BGR2HSV)
        
        # Red HSV ranges
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Save cursor processed crop (the red mask)
        try:
            cursor_proc_pil = Image.fromarray(mask)
            buf_cursor_proc = BytesIO()
            cursor_proc_pil.save(buf_cursor_proc, format="PNG")
            latest_cursor_processed_bytes = buf_cursor_proc.getvalue()
        except Exception as e_mask:
            print(f"Error caching processed cursor: {e_mask}")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= 5:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    furthest_pt = None
                    max_dist = -1
                    for pt in largest_contour:
                        px, py = pt[0][0], pt[0][1]
                        dist = (px - cX)**2 + (py - cY)**2
                        if dist > max_dist:
                            max_dist = dist
                            furthest_pt = (px, py)
                            
                    dx = furthest_pt[0] - cX
                    dy = furthest_pt[1] - cY
                    angle_rad = np.arctan2(dx, -dy)
                    bearing_deg = float(np.degrees(angle_rad) % 360)
                    
                    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                    idx = int((bearing_deg + 11.25) / 22.5) % 16
                    bearing_str = f"{bearing_deg:.1f}° ({directions[idx]})"
    except Exception as e:
        print(f"Error parsing bearing: {e}")

    # Crop character view
    try:
        from PIL import Image
        char_bounds = config.get("character_bounds", {"x": 0.45, "y": 0.30, "width": 0.10, "height": 0.40})
        full_w, full_h = full_img.size
        cx_min = int(char_bounds["x"] * full_w)
        cy_min = int(char_bounds["y"] * full_h)
        cw = int(char_bounds["width"] * full_w)
        ch = int(char_bounds["height"] * full_h)
        char_img = full_img.crop((cx_min, cy_min, cx_min + cw, cy_min + ch))
        
        buf_char = BytesIO()
        char_img.save(buf_char, format="PNG")
        latest_character_bytes = buf_char.getvalue()
        
        # Save character processed crop (Canny edges)
        try:
            char_cv = np.array(char_img)
            char_cv = char_cv[:, :, ::-1].copy()
            char_gray = cv2.cvtColor(char_cv, cv2.COLOR_BGR2GRAY)
            char_edges = cv2.Canny(char_gray, 50, 150)
            
            char_edges_pil = Image.fromarray(char_edges)
            buf_char_proc = BytesIO()
            char_edges_pil.save(buf_char_proc, format="PNG")
            latest_character_processed_bytes = buf_char_proc.getvalue()
        except Exception as e_char_proc:
            print(f"Error processing character crop: {e_char_proc}")
    except Exception as e:
        print(f"Error cropping character: {e}")

    # Generate location processed crop (OCR preprocess)
    t_prep_start = time.time()
    try:
        from PIL import Image
        loc_proc_np = ocr_parser.preprocess_image(text_img_1x, current_ocr_pass)
        loc_proc_pil = Image.fromarray(loc_proc_np)
        buf_loc_proc = BytesIO()
        loc_proc_pil.save(buf_loc_proc, format="PNG")
        latest_location_processed_bytes = buf_loc_proc.getvalue()
        
        # Save raw unbinarized location crop
        buf_loc_raw = BytesIO()
        text_img_1x.save(buf_loc_raw, format="PNG")
        latest_location_raw_bytes = buf_loc_raw.getvalue()
    except Exception as e_loc_proc:
        print(f"Error caching location images: {e_loc_proc}")
    t_prep_end = time.time()
    preprocess_time_ms = (t_prep_end - t_prep_start) * 1000.0

    # Create the 2x enlarged preview for the GUI
    try:
        from PIL import Image
        img_2x = text_img_1x.resize((w * 2, int((h - y_start) * 2)), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error resizing minimap text area: {e}")
        img_2x = text_img_1x

    # Cache full and cropped images as bytes for GUI
    try:
        # Save full screenshot
        buf_full = BytesIO()
        full_img.save(buf_full, format="PNG")
        latest_full_screenshot_bytes = buf_full.getvalue()
        
        # Save cropped and 2x enlarged text-only screenshot for GUI
        buf_crop = BytesIO()
        img_2x.save(buf_crop, format="PNG")
        latest_screenshot_bytes = buf_crop.getvalue()
    except Exception as e:
        print(f"Error caching screenshot: {e}")

    # Step 1: Attempt local OCR on the 1x text-only image
    print(f"[Pipeline] Attempting local Tesseract OCR (Pass {current_ocr_pass})...")
    t_start = time.time()
    location, coordinates, ns, ew = ocr_parser.run_ocr(text_img_1x, current_ocr_pass, already_cropped=True)
    t_end = time.time()
    tesseract_total_ms = (t_end - t_start) * 1000.0
    method = f"Local OCR (Pass {current_ocr_pass})"

    # Extract raw unfuzzy location/coordinates from Tesseract output
    raw_text = getattr(ocr_parser, "latest_raw_text", "")
    rich = ocr_parser.parse_text_rich(raw_text)
    raw_location = rich["raw_location"]
    raw_coordinates = rich["raw_coordinates"]

    # Step 2: Fallback to Gemini Multimodal if OCR failed (DISABLED BY USER)
    if not coordinates or not location:
        print(
            "[Pipeline] Local OCR was inconclusive. Gemini Vision fallback is currently disabled."
        )

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
    act_ram, act_vram = get_actual_usage()

    latest_parse_result = {
        "parsed_location": location,
        "parsed_coordinates": coordinates,
        "raw_location": raw_location if raw_location != "None" else None,
        "raw_coordinates": raw_coordinates if raw_coordinates != "None" else None,
        "parsed_bearing": bearing_str,
        "matched_track": track_file,
        "action": playback_action,
        "method": method,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "loc_time_ms": round(tesseract_total_ms * 0.55, 1),
        "coords_time_ms": round(tesseract_total_ms * 0.45, 1),
        "preprocess_time_ms": round(preprocess_time_ms, 1),
        "total_time_ms": round(tesseract_total_ms, 1),
        "actual_ram": act_ram,
        "actual_vram": act_vram
    }

    return {
        "status": "success",
        "method": method,
        "parsed_location": location,
        "parsed_coordinates": coordinates,
        "parsed_bearing": bearing_str,
        "matched_track": track_file,
        "action": playback_action,
    }


def play_track(track_file: str, start_time: float = 0.0, end_time: float | None = None) -> dict:
    """Manually plays a specific track from the playlist directory.

    Args:
        track_file: Filename of the track to play (e.g. 'town.wav').
        start_time: Start position in seconds.
        end_time: End position in seconds.

    Returns:
        dict containing status of playback.
    """
    # Manual play transitions stop the currently playing track immediately with no fade-in or fade-out delays
    success = player.play_track(track_file, fade_in_ms=0, fade_out_ms=0, start_time=start_time, end_time=end_time)
    if success:
        return {"status": "success", "message": f"Now playing {track_file}"}
    return {"status": "error", "message": f"Failed to play track {track_file}"}


def stop_music() -> dict:
    """Stops the current music playback immediately.

    Returns:
        dict containing status of playback.
    """
    player.stop(fade_out_ms=0)
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
