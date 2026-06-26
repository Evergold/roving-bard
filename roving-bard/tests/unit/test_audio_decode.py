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
import shutil
import subprocess
import pytest
import tinytag
import pygame
import soundfile as sf
from app.player import SafeMusicPlayer

@pytest.fixture(scope="module")
def audio_fixtures():
    """Generates valid 1-second silent audio test files for all supported formats."""
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixtures_dir = os.path.join(workspace_root, "tests", "temp_decode_fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    
    formats = {
        ".wav": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", "temp.wav"],
        ".mp3": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", "temp.mp3"],
        ".ogg": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", "temp.ogg"],
        ".flac": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", "temp.flac"],
        ".mp4": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", "-c:a", "aac", "temp.mp4"],
    }
    
    paths = {}
    
    # Check if ffmpeg is available
    ffmpeg_available = shutil.which("ffmpeg") is not None
    
    if ffmpeg_available:
        for ext, cmd in formats.items():
            out_path = os.path.join(fixtures_dir, f"test{ext}")
            cmd[-1] = out_path
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            paths[ext] = out_path
    else:
        # Fallback dummy file creation if ffmpeg is missing
        for ext in formats:
            out_path = os.path.join(fixtures_dir, f"test{ext}")
            with open(out_path, "wb") as f:
                f.write(b"dummy")
            paths[ext] = out_path
            
    # Generate .abc file (always plain text notation)
    abc_path = os.path.join(fixtures_dir, "test.abc")
    with open(abc_path, "w") as f:
        f.write("X:1\nT:Test Tune\nM:4/4\nK:C\nC D E F\n")
    paths[".abc"] = abc_path
    
    yield paths
    
    # Cleanup fixtures directory
    if os.path.exists(fixtures_dir):
        shutil.rmtree(fixtures_dir)


def test_metadata_decoding(audio_fixtures):
    """Verify that TinyTag can decode metadata for formats it supports, and player defaults correctly for others."""
    # TinyTag should succeed on .wav, .mp3, .ogg, .flac, .mp4
    for ext in [".wav", ".mp3", ".ogg", ".flac", ".mp4"]:
        path = audio_fixtures[ext]
        with open(path, "rb") as f:
            header = f.read(10)
        if header == b"dummy":
            continue
            
        tag = tinytag.TinyTag.get(path)
        assert tag.duration is not None
        assert tag.duration > 0
        
    # TinyTag should fail on .abc (plain text, unsupported tag format)
    abc_path = audio_fixtures[".abc"]
    with pytest.raises(tinytag.UnsupportedFormatError):
        tinytag.TinyTag.get(abc_path)


def test_player_playback_decoding(audio_fixtures):
    """Verify that player can load the audio formats. Tests non-simulated mode if mixer is available."""
    path_sample = audio_fixtures[".wav"]
    fixtures_dir = os.path.dirname(path_sample)
    
    player = SafeMusicPlayer(playlist_dir=fixtures_dir)
    
    # If mixer is initialized, test real pygame load/decode
    if not player.simulated:
        for ext in [".wav", ".mp3", ".ogg", ".flac", ".abc"]:
            path = audio_fixtures[ext]
            with open(path, "rb") as f:
                if f.read(10) == b"dummy":
                    continue
            
            # play_track/select_track returns True if loading/playing succeeds
            filename = os.path.basename(path)
            assert player.select_track(filename) is True
            if ext == ".abc":
                assert player.track_duration == 1.0
            
        # MP4 is not natively supported by standard pygame mixer backend, so it should fail to load/decode
        mp4_filename = os.path.basename(audio_fixtures[".mp4"])
        with open(audio_fixtures[".mp4"], "rb") as f:
            if f.read(10) != b"dummy":
                assert player.select_track(mp4_filename) is False
    else:
        # If simulated, check that selection executes cleanly
        for ext in audio_fixtures:
            filename = os.path.basename(audio_fixtures[ext])
            assert player.select_track(filename) is True
            if ext == ".abc":
                assert player.track_duration == 1.0


def test_eq_audio_decoding(audio_fixtures):
    """Verify that the EQ engine (via soundfile) can decode the supported audio formats."""
    # soundfile should decode .wav, .mp3, .ogg, .flac
    for ext in [".wav", ".mp3", ".ogg", ".flac"]:
        path = audio_fixtures[ext]
        with open(path, "rb") as f:
            if f.read(10) == b"dummy":
                continue
                
        info = sf.info(path)
        assert info.samplerate > 0
        assert info.frames > 0
        
        audio, sr = sf.read(path)
        assert sr == info.samplerate
        assert len(audio) > 0

    # soundfile should fail for .mp4 and .abc
    for ext in [".mp4", ".abc"]:
        path = audio_fixtures[ext]
        with open(path, "rb") as f:
            if f.read(10) == b"dummy":
                continue
        with pytest.raises(sf.LibsndfileError):
            sf.read(path)


def test_player_apply_eq_error_handling(audio_fixtures):
    """Verify that SafeMusicPlayer.apply_eq handles decoding errors gracefully."""
    path_sample = audio_fixtures[".wav"]
    fixtures_dir = os.path.dirname(path_sample)
    
    player = SafeMusicPlayer(playlist_dir=fixtures_dir)
    player.eq_gains = {freq: 0.0 for freq in player.eq_gains}
    
    # 1. Apply EQ to wav (should succeed)
    player.current_track = os.path.basename(audio_fixtures[".wav"])
    res = player.apply_eq()
    assert res["status"] == "success"
    
    # 2. Apply EQ to unsupported file formats (abc or mp4) - should fail gracefully with status 'error'
    player.current_track = os.path.basename(audio_fixtures[".abc"])
    res = player.apply_eq()
    assert res["status"] == "error"
    assert "Format not recognised" in res["message"] or "Error opening" in res["message"]
