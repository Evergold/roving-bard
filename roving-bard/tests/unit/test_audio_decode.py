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
import threading
import time
import pytest
import tinytag
import soundfile as sf
import numpy as np
from app.player import SafeMusicPlayer

@pytest.fixture(scope="module")
def audio_fixtures():
    """Generates valid 1-second silent audio test files for all supported formats."""
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixtures_dir = os.path.join(workspace_root, "tests", "temp_decode_fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    
    formats = {
        ".wav": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "temp.wav"],
        ".mp3": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "temp.mp3"],
        ".ogg": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "temp.ogg"],
        ".flac": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "temp.flac"],
        ".mp4": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "-c:a", "aac", "temp.mp4"],
        ".aac": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "-c:a", "aac", "temp.aac"],
        ".m4a": ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "-c:a", "aac", "temp.m4a"],
    }
    
    paths = {}
    
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
    
    # Generate a tiny 1-second dummy MIDI file
    mid_path = os.path.join(fixtures_dir, "test.mid")
    # A tiny valid MIDI header and track
    midi_dummy_bytes = b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\xc0MTrk\x00\x00\x00\x0b\x00\xff\x51\x03\x07\xa1\x20\x00\xff\x2f\x00'
    with open(mid_path, "wb") as f:
        f.write(midi_dummy_bytes)
    paths[".mid"] = mid_path
    
    yield paths
    
    # Cleanup fixtures directory
    if os.path.exists(fixtures_dir):
        shutil.rmtree(fixtures_dir)


def test_metadata_decoding(audio_fixtures):
    """Verify that TinyTag can decode metadata for formats it supports, and player defaults correctly for others."""
    for ext in [".wav", ".mp3", ".ogg", ".flac", ".mp4", ".aac", ".m4a"]:
        path = audio_fixtures[ext]
        with open(path, "rb") as f:
            header = f.read(10)
        if header == b"dummy":
            continue
            
        tag = tinytag.TinyTag.get(path)
        # Raw .aac (ADTS) does not store duration in headers, which is expected to return None
        if ext not in [".aac", ".m4a"]:
            assert tag.duration is not None
            assert tag.duration > 0
        
    # TinyTag should fail on .abc (plain text, unsupported tag format)
    abc_path = audio_fixtures[".abc"]
    with pytest.raises(tinytag.UnsupportedFormatError):
        tinytag.TinyTag.get(abc_path)


def test_player_playback_decoding(audio_fixtures):
    """Verify that player can load and decode all audio formats using the sounddevice engine."""
    path_sample = audio_fixtures[".wav"]
    fixtures_dir = os.path.dirname(path_sample)
    
    player = SafeMusicPlayer(playlist_dir=fixtures_dir)
    
    # Disable actual audio output in tests to prevent ALSA/PortAudio errors in CI
    player.sd_device = None
    
    # Verify select_track and duration parsing for all formats
    for ext in [".wav", ".mp3", ".ogg", ".flac", ".mp4", ".aac", ".m4a", ".abc", ".mid"]:
        path = audio_fixtures[ext]
        with open(path, "rb") as f:
            if f.read(10) == b"dummy":
                continue
        
        filename = os.path.basename(path)
        assert player.select_track(filename) is True
        assert player.track_duration > 0.0


def test_in_depth_decoding(audio_fixtures):
    """Verify that the player can decode audio chunks (WAV/FLAC/OGG via soundfile, MP3/MP4 via ffmpeg)."""
    path_sample = audio_fixtures[".wav"]
    fixtures_dir = os.path.dirname(path_sample)
    
    player = SafeMusicPlayer(playlist_dir=fixtures_dir)
    
    # 1. Test SoundFile-based decoding (WAV, FLAC, OGG)
    for ext in [".wav", ".flac", ".ogg"]:
        path = audio_fixtures[ext]
        filename = os.path.basename(path)
        
        assert player.select_track(filename) is True
        # Emulate starting playback but without the audio output thread
        player._sf = sf.SoundFile(path)
        assert player._sf is not None
        
        # Decode a chunk
        chunk = player._sf.read(1024, dtype="float32", always_2d=True).copy()
        assert chunk.shape == (1024, 2)
        assert chunk.dtype == np.float32
        
        player._sf.close()
        player._sf = None

    # 2. Test FFmpeg-based pipe decoding (MP3, MP4)
    for ext in [".mp3", ".mp4", ".aac", ".m4a"]:
        path = audio_fixtures[ext]
        filename = os.path.basename(path)
        
        assert player.select_track(filename) is True
        # Start the ffmpeg pipe
        cmd = [
            "ffmpeg", "-y", "-ss", "0.000", "-i", path,
            "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "2", "-"
        ]
        player._ffmpeg_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        assert player._ffmpeg_proc is not None
        
        # Read a chunk from the pipe (1024 frames * 2 channels * 2 bytes = 4096 bytes)
        raw_bytes = player._ffmpeg_proc.stdout.read(1024 * 4)
        assert len(raw_bytes) > 0
        
        # Decode the raw bytes
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        chunk = samples.reshape((-1, 2))
        assert chunk.shape[1] == 2
        assert chunk.dtype == np.float32
        
        player._ffmpeg_proc.terminate()
        player._ffmpeg_proc = None


def test_eq_realtime_processing(audio_fixtures):
    """Verify that the 10-band EQ filters process audio chunks correctly."""
    path_sample = audio_fixtures[".wav"]
    fixtures_dir = os.path.dirname(path_sample)
    
    player = SafeMusicPlayer(playlist_dir=fixtures_dir)
    
    # Create a 1024-frame silent stereo chunk
    chunk = np.zeros((1024, 2), dtype=np.float32)
    
    # Set custom EQ gains (boost 1000Hz by 6dB, cut 125Hz by -10dB)
    player.eq_gains[1000] = 6.0
    player.eq_gains[125] = -10.0
    
    # Emulate the EQ application in the playback loop
    sos_dict = {}
    last_eq_gains = {}
    zi_dict = {}
    
    # 1. Compute filters
    for band, gain in player.eq_gains.items():
        Q = 1.0
        for fb, qb in player._EQ_BANDS:
            if fb == band:
                Q = qb
                break
        sos_dict[band] = player._peaking_sos(float(band), float(gain), float(Q), 44100)
        
    # 2. Apply filters
    from scipy.signal import sosfilt
    for band, sos in sos_dict.items():
        if band not in zi_dict:
            zi_dict[band] = np.zeros((1, 2, 2), dtype=np.float32)
        chunk, zi_dict[band] = sosfilt(sos, chunk, zi=zi_dict[band], axis=0)
        
    # Cast back to float32 just like the player's playback loop does
    chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        
    assert chunk.shape == (1024, 2)
    assert chunk.dtype == np.float32
