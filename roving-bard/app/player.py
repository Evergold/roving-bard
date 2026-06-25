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
import re

import cv2
import mss
import numpy as np
import pygame
import pytesseract
from PIL import Image
from tinytag import TinyTag


# Safe pygame mixer initialization
class SafeMusicPlayer:
    def __init__(self, playlist_dir="music"):
        self.playlist_dir = playlist_dir
        self.current_track = None
        self.volume = 1.0
        self.mixer_initialized = False
        self.simulated = False
        self.paused = False
        self.was_stopped = False
        self.track_duration = 0.0
        self.last_seek_position = 0.0
        self.last_play_time = None
        self.start_time = 0.0
        self.end_time = None

        try:
            pygame.mixer.init()
            self.mixer_initialized = True
            print("Successfully initialized Pygame mixer.")
        except Exception as e:
            self.simulated = True
            print(
                f"Warning: Could not initialize Pygame mixer (running in simulated mode): {e}"
            )

    def play_track(self, track_file, fade_in_ms=1500, fade_out_ms=1500):
        if not track_file:
            self.stop(fade_out_ms)
            return True

        track_path = os.path.join(self.playlist_dir, track_file)
        if not os.path.exists(track_path):
            print(f"Warning: Track file not found: {track_path}")
            return False

        if self.current_track == track_file:
            # Track is already playing
            if self.paused:
                return self.resume()
            return True

        print(
            f"[Playback] Transitioning to track '{track_file}' (fadeout: {fade_out_ms}ms, fadein: {fade_in_ms}ms)"
        )

        # If a track is currently playing, fade it out first and wait for it to end (only if not paused)
        if self.current_track:
            if not self.paused and fade_out_ms > 0:
                if not self.simulated and self.mixer_initialized:
                    try:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.fadeout(fade_out_ms)
                    except Exception as e:
                        print(f"Error during fadeout: {e}")
                import time
                time.sleep(fade_out_ms / 1000.0)
            else:
                if not self.simulated and self.mixer_initialized:
                    try:
                        pygame.mixer.music.stop()
                    except Exception as e:
                        print(f"Error stopping music: {e}")

        self.current_track = track_file
        self.paused = False
        self.was_stopped = False

        # Load track duration
        self.track_duration = 0.0
        try:
            tag = TinyTag.get(track_path)
            self.track_duration = tag.duration
        except Exception as e:
            print(f"Error loading track duration with TinyTag: {e}")
        if self.track_duration == 0.0:
            self.track_duration = 180.0

        self.start_time = 0.0
        self.end_time = None
        self.last_seek_position = self.start_time
        import time
        self.last_play_time = time.time()

        if self.simulated:
            print(f"[Playback SIMULATED] Playing: {track_file}")
            return True

        try:
            pygame.mixer.music.load(track_path)
            pygame.mixer.music.play(loops=-1, start=self.start_time, fade_ms=fade_in_ms)
            pygame.mixer.music.set_volume(self.volume)
            return True
        except Exception as e:
            print(f"Error during Pygame playback of {track_file}: {e}")
            return False

    def stop(self, fade_out_ms=1500):
        if not self.current_track:
            return

        print(f"[Playback] Stopping playback (fadeout: {fade_out_ms}ms)")
        self.paused = True
        self.was_stopped = True
        self.last_seek_position = 0.0
        self.last_play_time = None

        if self.simulated:
            return

        try:
            if pygame.mixer.music.get_busy():
                if fade_out_ms > 0:
                    pygame.mixer.music.fadeout(fade_out_ms)
                else:
                    pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error stopping playback: {e}")

    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))
        if self.mixer_initialized and not self.simulated:
            try:
                pygame.mixer.music.set_volume(self.volume)
            except Exception as e:
                print(f"Error setting volume: {e}")
        print(f"[Playback] Volume set to {int(self.volume * 100)}%")

    def pause(self):
        if not self.current_track:
            return False
        print("[Playback] Pausing music.")
        
        # Capture current progress before pausing
        import time
        if not self.paused and self.last_play_time is not None:
            self.last_seek_position += time.time() - self.last_play_time
            self.last_play_time = None

        self.paused = True
        if self.simulated:
            return True
        try:
            pygame.mixer.music.pause()
            return True
        except Exception as e:
            print(f"Error pausing music: {e}")
            return False

    def resume(self):
        if not self.current_track:
            return False
        print("[Playback] Resuming music.")
        self.paused = False

        import time
        self.last_play_time = time.time()

        if self.simulated:
            if self.was_stopped:
                if self.last_seek_position < self.start_time:
                    self.last_seek_position = self.start_time
            self.was_stopped = False
            return True
        try:
            if self.was_stopped:
                start_pos = self.last_seek_position
                if start_pos < self.start_time:
                    start_pos = self.start_time
                track_path = os.path.join(self.playlist_dir, self.current_track)
                pygame.mixer.music.load(track_path)
                pygame.mixer.music.play(loops=-1, start=start_pos, fade_ms=1500)
                pygame.mixer.music.set_volume(self.volume)
                self.was_stopped = False
            else:
                pygame.mixer.music.unpause()
            return True
        except Exception as e:
            print(f"Error resuming music: {e}")
            return False

    def seek(self, position):
        if not self.current_track:
            return False

        position = max(0.0, min(self.track_duration, position))
        print(f"[Playback] Seeking to {position}s (was_stopped={self.was_stopped})")

        self.last_seek_position = position
        import time

        if self.was_stopped:
            # When stopped, seeking only updates the current position marker without playing
            self.last_play_time = None
            return True

        # Otherwise continue current behavior (resume/start playback)
        self.last_play_time = time.time()
        self.paused = False
        self.was_stopped = False

        if self.simulated:
            return True

        try:
            track_path = os.path.join(self.playlist_dir, self.current_track)
            pygame.mixer.music.load(track_path)
            pygame.mixer.music.play(loops=-1, start=position)
            pygame.mixer.music.set_volume(self.volume)
            return True
        except Exception as e:
            print(f"Error seeking: {e}")
            return False

    def get_current_position(self):
        if not self.current_track:
            return 0.0
        import time
        pos = self.last_seek_position
        if not self.paused and self.last_play_time is not None:
            pos += time.time() - self.last_play_time

        duration = self.track_duration
        start = self.start_time
        end = self.end_time if self.end_time is not None else duration

        if start < 0:
            start = 0.0
        if end > duration:
            end = duration

        range_len = end - start
        if range_len > 0:
            if pos > end:
                # Loop back to start
                loops = int((pos - start) // range_len)
                pos = start + ((pos - start) % range_len)
                
                # Update Pygame playback position
                if not self.simulated and self.mixer_initialized:
                    try:
                        pygame.mixer.music.play(loops=-1, start=pos)
                    except Exception as e:
                        print(f"Error during loop seek: {e}")
                self.last_seek_position = pos
                self.last_play_time = time.time()
        else:
            if duration > 0:
                if pos > duration:
                    pos = pos % duration
        return max(0.0, pos)


CAPTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "capture"
)


# Minimap screen grabbing and cropping
class ScreenGrabber:
    def __init__(self, bounds_config):
        self.bounds = bounds_config

    def capture_full(self):
        """Captures the primary monitor screen or loads from capture directory, returning the full uncropped image."""
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        # Check manual screen captures first
        if os.path.exists(CAPTURE_DIR):
            files = [f for f in os.listdir(CAPTURE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                filepath = os.path.join(CAPTURE_DIR, files[0])
                try:
                    img = Image.open(filepath).convert("RGB")
                    print(f"[ScreenGrabber] Loaded manual capture: {filepath}")
                    return img
                except Exception as e:
                    print(f"Error loading manual capture {filepath}: {e}")

        # Fallback to mss capture
        try:
            with mss.mss() as sct:
                # Primary monitor is 1 (0 is virtual screen of all monitors combined)
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                # Convert to PIL Image immediately
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                return img
        except Exception as e:
            print(f"Error capturing full screenshot: {e}")
            return None

    def crop_image(self, img):
        """Crops a full image to the minimap bounds."""
        if not img or not self.bounds:
            return img
        try:
            width, height = img.size
            left = int(self.bounds["x"] * width)
            top = int(self.bounds["y"] * height)
            crop_width = int(self.bounds["width"] * width)
            crop_height = int(self.bounds["height"] * height)
            return img.crop((left, top, left + crop_width, top + crop_height))
        except Exception as e:
            print(f"Error cropping image: {e}")
            return img

    def capture_and_crop(self):
        """Maintains backward compatibility by capturing and cropping immediately."""
        full_img = self.capture_full()
        return self.crop_image(full_img)


# Local OCR and parsing
class LocalOCRParser:
    @staticmethod
    def preprocess_image(pil_img):
        """Applies grayscale, 2x resizing, and Otsu thresholding for OCR optimization."""
        # Convert PIL to open-cv format
        cv_img = np.array(pil_img)
        cv_img = cv_img[:, :, ::-1].copy()  # Convert RGB to BGR

        # Grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 2x Resize
        resized = cv2.resize(
            gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
        )

        # Thresholding (Otsu binarization)
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def parse_text(text):
        """Extracts coordinate floats (signed) and potential location names from OCR text."""
        # Coordinate pattern: e.g. 19.3N, 70.9W or 14.9S, 103.1E
        # Lat/NS: N is positive, S is negative. Long/EW: E is positive, W is negative.
        coord_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*([NS])[\s,\-]+(\d+(?:\.\d+)?)\s*([EW])", re.IGNORECASE
        )

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        location = None
        coordinates = None
        ns_val, ew_val = None, None

        # Search for coordinates in text
        for line in lines:
            match = coord_pattern.search(line)
            if match:
                ns_raw = float(match.group(1))
                ns_dir = match.group(2).upper()
                ew_raw = float(match.group(3))
                ew_dir = match.group(4).upper()

                ns_val = ns_raw if ns_dir == "N" else -ns_raw
                ew_val = ew_raw if ew_dir == "E" else -ew_raw
                coordinates = f"{ns_raw}{ns_dir}, {ew_raw}{ew_dir}"
                break

        # Extract location: find lines that are not coordinates and contain alphabetical characters
        for line in lines:
            if coord_pattern.search(line):
                continue
            # Remove symbols/noise, check if it looks like a location name
            cleaned = re.sub(r"[^a-zA-Z\s]", "", line).strip()
            if len(cleaned) > 2:  # At least 3 chars
                location = cleaned
                break

        return location, coordinates, ns_val, ew_val

    def run_ocr(self, pil_img):
        try:
            processed = self.preprocess_image(pil_img)
            # Run pytesseract OCR
            raw_text = pytesseract.image_to_string(processed)
            return self.parse_text(raw_text)
        except Exception as e:
            print(f"Local OCR engine execution error: {e}")
            return None, None, None, None


# Coordinates and Location Mapper
class TrackMapper:
    def __init__(self, mappings):
        self.mappings = mappings

    def get_track_for_state(self, location, ns, ew):
        """Matches current location/coordinates against configured mappings.

        Matches location names first (fuzzy substring), then matches coordinate ranges.
        """
        # 1. Match by Location Name if available
        if location:
            for mapping in self.mappings:
                loc_name = mapping.get("location_name")
                if loc_name and loc_name.lower() in location.lower():
                    print(
                        f"[Mapper] Matched location name: '{loc_name}' -> '{mapping['track_file']}'"
                    )
                    return mapping["track_file"]

        # 2. Match by Coordinate Ranges if coordinates are available
        if ns is not None and ew is not None:
            for mapping in self.mappings:
                # Range fields: ns_min, ns_max, ew_min, ew_max
                if all(k in mapping for k in ["ns_min", "ns_max", "ew_min", "ew_max"]):
                    if (
                        mapping["ns_min"] <= ns <= mapping["ns_max"]
                        and mapping["ew_min"] <= ew <= mapping["ew_max"]
                    ):
                        print(
                            f"[Mapper] Matched coordinate range (NS: {ns}, EW: {ew}) -> '{mapping['track_file']}'"
                        )
                        return mapping["track_file"]

        return None
