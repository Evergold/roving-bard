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

import math
import os
import struct
import time
import wave

from dotenv import load_dotenv

# Load local environment variables (.env) if present
load_dotenv()


# Helper function to generate mock audio files for testing out-of-the-box
def generate_mock_tracks(playlist_dir):
    os.makedirs(playlist_dir, exist_ok=True)

    # 4 sample frequencies for our regions
    tracks = {
        "town.wav": 440,  # A4 (Town theme)
        "forest.wav": 554,  # C#5 (Forest theme)
        "boss.wav": 659,  # E5 (Boss arena theme)
        "cave.wav": 330,  # E4 (Cave theme)
    }

    sample_rate = 22050
    duration = 3.0  # 3 seconds is plenty for a loop test
    num_samples = int(duration * sample_rate)

    print("Generating mock audio files for testing...")
    for filename, freq in tracks.items():
        filepath = os.path.join(playlist_dir, filename)
        if os.path.exists(filepath):
            continue

        print(f"Creating mock track: {filepath} ({freq}Hz tone)")
        # Generate simple sine wave tone
        with wave.open(filepath, "w") as wav_file:
            # Mono, 2 bytes per sample, sample_rate
            wav_file.setparams(
                (1, 2, sample_rate, num_samples, "NONE", "not compressed")
            )
            for i in range(num_samples):
                t = float(i) / sample_rate
                # Generate sine wave
                value = int(32767.0 * math.sin(2.0 * math.pi * freq * t))
                data = struct.pack("<h", value)
                wav_file.writeframesraw(data)
    print("Mock tracks successfully generated.\n")


def main():
    print("=" * 60)
    print("         GAME-AWARE MUSIC PLAYER AGENT RUNNER")
    print("=" * 60)

    # Import tools (this will parse mapping.yaml)
    from app.tools import check_screen_and_update_music, config, stop_music, player
    player.initialize_backend(verbose=True)

    # Generate mock tracks if they don't exist
    playlist_dir = config.get("playlist_directory", "audio")
    generate_mock_tracks(playlist_dir)

    polling_interval = config.get("polling_interval", 2.0)
    model_name = config.get("model_name", "gemini/gemini-2.5-flash-lite")
    bounds = config.get("minimap_bounds", {})

    print("Configuration Loaded:")
    print(f" - Polling Interval: {polling_interval} seconds")
    print(f" - Fallback Model:   {model_name}")
    print(f" - Playlist Folder:  {playlist_dir}/")
    print(
        f" - Minimap Bounds:   x={bounds.get('x')}, y={bounds.get('y')}, w={bounds.get('width')}, h={bounds.get('height')}"
    )
    print("\nMonitoring game screen... Press Ctrl+C to stop.")
    print("-" * 60)

    try:
        while True:
            start_time = time.time()

            # Execute screen capture, parsing, matching, and playback update
            res = check_screen_and_update_music()

            # Print feedback
            if res.get("status") == "success":
                loc = res.get("parsed_location") or "Unknown"
                coord = res.get("parsed_coordinates") or "Unknown"
                track = res.get("matched_track") or "None"
                action = res.get("action")
                method = res.get("method")

                print(
                    f"[Status] Location: {loc:<15} Coordinates: {coord:<18} Track: {track:<12} Action: {action:<18} ({method})"
                )
            else:
                print(f"[Error] Failed step: {res.get('message')}")

            # Throttle based on config
            elapsed = time.time() - start_time
            sleep_time = max(0.1, polling_interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nExiting and stopping playback...")
        stop_music()
        print("Goodbye!")


if __name__ == "__main__":
    main()
