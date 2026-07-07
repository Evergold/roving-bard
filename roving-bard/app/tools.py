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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
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
    cfg = {}
    if not os.path.exists(CONFIG_PATH):
        # Auto-create config.yaml from config.yaml.template if it exists in the agent root
        template_path = os.path.join(os.path.dirname(CONFIG_PATH), "config.yaml.template")
        if os.path.exists(template_path):
            try:
                import shutil
                shutil.copy(template_path, CONFIG_PATH)
                print(f"[Config] Created {CONFIG_PATH} from template.")
            except Exception as e:
                print(f"[Config] Error copying template to config.yaml: {e}")
                
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    
    defaults = {
        "minimap_bounds": {"x": 0.8, "y": 0.05, "width": 0.15, "height": 0.15},
        "force_manual_bounds": False,
        "transitions": {"fade_out_ms": 1500, "fade_in_ms": 1500},
        "playlist_directory": "audio",
        "model_name": "gemini/gemini-2.5-flash-lite",
        "polling_interval": 2.0,
        "mappings": [],
        "api_key": None,
        "ui_lang": "en-US",
        "lotro_locale": "en"
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


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
minimap_detecting = False
server_baseline_ram = 1100 * 1024 * 1024  # Default fallback 1.1 GB
server_baseline_vram = 0
current_ocr_pass = "auto"
detection_generation = 0
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
    "postprocess_time_ms": 0.0,
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
    vlm_format = config.get("vlm_image_format", "JPEG").upper()
    if vlm_format == "PNG":
        img.save(buffered, format="PNG", compress_level=1)
    else:
        if img.mode != "RGB":
            img.convert("RGB").save(buffered, format="JPEG", quality=95)
        else:
            img.save(buffered, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Constrain to expected LOTRO locale based on user selection
    lotro_lang = config.get("lotro_locale", "en")
    if lotro_lang not in ("en", "fr", "de"):
        lotro_lang = "en"
    
    lotro_lang_name = "English"
    if lotro_lang == "fr":
        lotro_lang_name = "French"
    elif lotro_lang == "de":
        lotro_lang_name = "German"

    prompt = f"Extract both the location name and the coordinates in {lotro_lang_name} without translation. Output format: Location Name, Coordinates. Do not include any other text."

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
            temperature=0.01,
        )
        content = response.choices[0].message.content.strip()
        
        # Check if response is JSON formatted (fallback check for robustness/testing)
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            try:
                json_str = content[start_idx:end_idx+1]
                parsed = json.loads(json_str)
                location_val = parsed.get("location")
                coordinates_val = parsed.get("coordinates")
                ns_val, ew_val = None, None
                if coordinates_val:
                    _, _, ns_val, ew_val = ocr_parser.parse_text(f"dummy\n{coordinates_val}")
                if location_val:
                    # Fuzzy match location using standard dictionary parser
                    location_val, _, _, _ = ocr_parser.parse_text(location_val)
                return location_val, coordinates_val, ns_val, ew_val
            except Exception:
                pass
                
        location, coordinates, ns_val, ew_val = ocr_parser.parse_text(content)
        return location, coordinates, ns_val, ew_val
    except Exception as e:
        print(f"Error executing Gemini Vision fallback: {e}")
        return None, None, None, None


def get_system_vram_bytes():
    import subprocess
    import os
    
    # 1. Try Nvidia
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=1.0
        )
        return int(res.stdout.strip()) * 1024 * 1024
    except Exception:
        pass
        
    # 2. Try AMD ROCm
    try:
        res = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, check=True, timeout=1.0
        )
        for line in res.stdout.splitlines():
            if "Used Memory" in line and "(B)" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    return int(parts[1].strip())
    except Exception:
        pass
        
    # 3. Try Intel Linux sysfs
    try:
        for card in ["card0", "card1"]:
            path = f"/sys/class/drm/{card}/device/mem_info_vram_used"
            if os.path.exists(path):
                with open(path, "r") as f:
                    return int(f.read().strip())
    except Exception:
        pass
        
    return None


def get_actual_usage():
    """Get actual RAM and VRAM usage on Linux systems, omitting the baseline footprint."""
    try:
        import os
        pid = os.getpid()
        ram_bytes = 0
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ram_bytes = int(parts[1]) * 1024
                    break
        
        vram_bytes = get_system_vram_bytes() or 0
            
        incremental_ram = max(0, ram_bytes - server_baseline_ram)
        incremental_vram = max(0, vram_bytes - server_baseline_vram)
        
        def fmt_mem(bytes_val):
            if bytes_val <= 0:
                return "0 MB"
            mb_val = bytes_val / (1024 * 1024)
            if mb_val >= 1000:
                return f"{mb_val / 1024:.2f} GB"
            return f"{int(mb_val)} MB"
            
        return fmt_mem(incremental_ram), fmt_mem(incremental_vram)
    except Exception as e:
        print(f"Error getting actual memory usage: {e}")
        return None, None


# Agent Tools


def check_screen_and_update_music(ignore_detecting: bool = False, skip_ocr: bool = False) -> dict:
    """Captures the foreground game screen, parses the location and coordinates,
    matches them against config.yaml, and plays the appropriate music track.

    Returns:
        dict containing the extraction result (location, coordinates) and action taken.
    """
    global latest_screenshot_bytes, latest_full_screenshot_bytes, latest_cursor_bytes, latest_character_bytes, latest_cursor_processed_bytes, latest_location_processed_bytes, latest_location_raw_bytes, latest_character_processed_bytes, latest_parse_result, current_ocr_pass
    
    bearing_prep_ms = 0.0
    import app.tools as tools_mod
    if not ignore_detecting and getattr(tools_mod, "minimap_detecting", False):
        print("[ScanPipeline] Skipping scan: bounds detection is in progress.")
        return {"status": "skipped", "message": "Minimap bounds detection in progress."}
        
    full_img = grabber.capture_full()
    if not full_img:
        return {"status": "error", "message": "Failed to capture screenshot."}

    # Crop to bounds for OCR processing
    img = grabber.crop_image(full_img)

    # Crop to location and coordinates (snug centered text area, consistent across all auto-detected screens)
    w, h = img.size
    y_start = int(h * 0.772)
    y_end = int(h * 0.925)
    x_min = int(w * 0.096)
    x_max = int(w * 0.950)
    text_img_1x = img.crop((x_min, y_start, x_max, y_end))

    # Parse bearing from red cursor at the center of the minimap and crop it
    bearing_deg = None
    bearing_str = "None"
    try:
        import numpy as np
        import cv2
        from PIL import Image
        
        # Center of the radar circle (vertical center is slightly higher in the widget)
        cx = int(w * 0.512)
        cy = int(h * 0.367)
        r = int(w * 0.10)
        cursor_crop = img.crop((cx - r, cy - r, cx + r, cy + r))
        
        # Enlarge 3x for preview and high-precision detection
        cursor_crop_3x = cursor_crop.resize((r * 6, r * 6), Image.Resampling.LANCZOS)
        buf_cursor = BytesIO()
        cursor_crop_3x.save(buf_cursor, format="PNG")
        latest_cursor_bytes = buf_cursor.getvalue()
        
        # Parse bearing angle using the scaled up 3x image
        cv_crop = np.array(cursor_crop_3x)
        cv_crop = cv_crop[:, :, ::-1].copy()
        
        # Pre-process the BGR crop to fix the orange corners of the chevron to red before final HSV conversion
        t_bearing_prep_start = time.time()
        hsv_temp = cv2.cvtColor(cv_crop, cv2.COLOR_BGR2HSV)
        
        # 1. Find red core of chevron
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv_temp, lower_red1, upper_red1) | cv2.inRange(hsv_temp, lower_red2, upper_red2)
        
        # 2. Calculate red centroid (fall back to center of image if no red found)
        cX_red, cY_red = cv_crop.shape[1] // 2, cv_crop.shape[0] // 2
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_red:
            largest_contour_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_contour_red) >= 5:
                M_red = cv2.moments(largest_contour_red)
                if M_red["m00"] > 0:
                    cX_red = int(M_red["m10"] / M_red["m00"])
                    cY_red = int(M_red["m01"] / M_red["m00"])
                    
        # 3. Dilate red mask to cover adjacent orange tips (3x3 kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_red = cv2.dilate(red_mask, kernel)
        
        # 4. Spatial mask centered on the detected red centroid (radius 24)
        H, W = cv_crop.shape[:2]
        y_indices, x_indices = np.ogrid[:H, :W]
        dist_from_centroid = np.sqrt((x_indices - cX_red)**2 + (y_indices - cY_red)**2)
        centroid_mask = dist_from_centroid <= 24
        
        # 5. Find only bright, saturated orange/yellow/peach pixels (excluding dark background paths/terrain)
        lower_orange = np.array([11, 80, 200])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv_temp, lower_orange, upper_orange)
        
        # 6. Combined mask: orange AND adjacent to red AND close to centroid
        valid_orange_mask = orange_mask & dilated_red & centroid_mask
        cv_crop[valid_orange_mask > 0] = [0, 0, 255] # make them BGR red
        t_bearing_prep_end = time.time()
        bearing_prep_ms = (t_bearing_prep_end - t_bearing_prep_start) * 1000.0
        
        hsv = cv2.cvtColor(cv_crop, cv2.COLOR_BGR2HSV)
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
        preview_pass = 2 if current_ocr_pass == "auto" else current_ocr_pass
        loc_proc_np = ocr_parser.preprocess_image(text_img_1x, preview_pass)
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
    preprocess_time_ms = (t_prep_end - t_prep_start) * 1000.0 + bearing_prep_ms

    # Create the 2x enlarged preview for the GUI
    try:
        from PIL import Image
        img_2x = text_img_1x.resize((text_img_1x.width * 2, text_img_1x.height * 2), Image.Resampling.LANCZOS)
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
    if skip_ocr:
        import datetime
        act_ram, act_vram = get_actual_usage()
        latest_parse_result = {
            "parsed_location": "",
            "parsed_coordinates": "",
            "raw_location": None,
            "raw_coordinates": None,
            "parsed_bearing": bearing_str,
            "matched_track": "None",
            "action": "stopped",
            "method": "None",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "loc_time_ms": 0.0,
            "coords_time_ms": 0.0,
            "preprocess_time_ms": round(preprocess_time_ms, 1),
            "postprocess_time_ms": 0.0,
            "total_time_ms": 0.0,
            "actual_ram": act_ram,
            "actual_vram": act_vram,
            "current_ocr_pass": current_ocr_pass
        }
        return {
            "status": "success",
            "method": "None",
            "parsed_location": "",
            "parsed_coordinates": "",
            "parsed_bearing": bearing_str,
            "matched_track": "None",
            "action": "stopped",
        }

    t_start = time.time()
    location, coordinates, ns, ew = ocr_parser.run_ocr(text_img_1x, current_ocr_pass, already_cropped=True)
    t_end = time.time()
    tesseract_total_ms = (t_end - t_start) * 1000.0
    actual_pass = getattr(ocr_parser, "latest_successful_pass", 2)
    method = f"Local OCR (Pass {actual_pass})"
    
    # Regenerate preview image using the actual successful pass if we were in auto mode
    if current_ocr_pass == "auto":
        try:
            loc_proc_np = ocr_parser.preprocess_image(text_img_1x, actual_pass)
            loc_proc_pil = Image.fromarray(loc_proc_np)
            buf_loc_proc = BytesIO()
            loc_proc_pil.save(buf_loc_proc, format="PNG")
            latest_location_processed_bytes = buf_loc_proc.getvalue()
        except Exception:
            pass

    # Extract raw unfuzzy location/coordinates from Tesseract output
    raw_text = getattr(ocr_parser, "latest_raw_text", "")
    t_post0 = time.time()
    rich = ocr_parser.parse_text_rich(raw_text)
    t_post1 = time.time()
    postprocess_time_ms = (t_post1 - t_post0) * 1000.0
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
        "postprocess_time_ms": round(postprocess_time_ms, 2),
        "total_time_ms": round(tesseract_total_ms, 1),
        "actual_ram": act_ram,
        "actual_vram": act_vram,
        "current_ocr_pass": current_ocr_pass
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
