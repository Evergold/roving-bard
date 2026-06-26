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
import tempfile
import threading

import cv2
import mss
import numpy as np
import pygame
import pytesseract
from PIL import Image
from tinytag import TinyTag




def abc_to_midi_bytes(abc_text: str) -> bytes:
    meter = 1.0
    unit_note_len = None
    program = 0  # Default to Piano
    
    headers_done = False
    notes_parts = []
    
    header_pattern = re.compile(r'^([A-Z]):\s*(.*)$')
    midi_program_pattern = re.compile(r'(?:%%MIDI\s+program|I:MIDI\s+program)\s+(\d+)', re.IGNORECASE)
    
    for line in abc_text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Parse %%MIDI program from comment lines
        if line.startswith('%') or line.startswith('I:'):
            m_prog = midi_program_pattern.search(line)
            if m_prog:
                program = int(m_prog.group(1))
            if line.startswith('%'):
                continue
                
        match = header_pattern.match(line)
        if match and not headers_done:
            key, val = match.group(1), match.group(2).strip()
            if key == 'M':
                if val.lower() in ('c', '4/4'):
                    meter = 1.0
                elif val.lower() == 'c|':
                    meter = 1.0
                else:
                    m = re.match(r'(\d+)/(\d+)', val)
                    if m:
                        meter = float(m.group(1)) / float(m.group(2))
            elif key == 'L':
                m = re.match(r'(\d+)/(\d+)', val)
                if m:
                    unit_note_len = float(m.group(1)) / float(m.group(2))
            elif key == 'K':
                headers_done = True
        else:
            cleaned_line = line.split('%')[0]
            cleaned_line = re.sub(r'"[^"]*"', '', cleaned_line)
            cleaned_line = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', cleaned_line)
            notes_parts.append(cleaned_line)
            
    if unit_note_len is None:
        unit_note_len = 0.0625 if meter < 0.75 else 0.125
        
    ticks_per_quarter = 480
    unit_ticks = int(1920 * unit_note_len)
    
    # MIDI track events bytes
    track_events = bytearray()
    
    # 1. Delta-time 0, Program Change (C0 <program>)
    track_events.append(0x00)
    track_events.extend([0xC0, program])
    
    # Parse notes and chords
    pattern = re.compile(r'\[([^\]]+)\]|([_^^=]*)([A-Ga-gxzXZ])([,\']*)(\d*(?:/+\d*)*)')
    note_pattern = re.compile(r'([_^^=]*)([A-Ga-gxzXZ])([,\']*)(\d*(?:/+\d*)*)')
    
    PITCH_MAP = {
        'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
        'c': 72, 'd': 74, 'e': 76, 'f': 77, 'g': 79, 'a': 81, 'b': 83
    }
    
    def parse_multiplier(num_str, slash_str):
        num = float(num_str) if num_str else 1.0
        if not slash_str:
            return num
        slash_count = slash_str.count('/')
        m = re.search(r'(\d+)$', slash_str)
        if m:
            denom = float(m.group(1))
            return num / denom
        else:
            return num / (2 ** slash_count)
            
    def get_midi_note(acc, pitch, octaves):
        if pitch in ('z', 'x', 'Z', 'X'):
            return None
        note = PITCH_MAP.get(pitch, 60)
        note += acc.count('^')
        note += 2 * acc.count('^^')
        note -= acc.count('_')
        note -= 2 * acc.count('__')
        note -= 12 * octaves.count(',')
        note += 12 * octaves.count("'")
        return max(0, min(127, note))

    def to_vlq(n: int) -> bytes:
        out = bytearray()
        while True:
            out.append((n & 0x7f) | (0x80 if out else 0))
            n >>= 7
            if n == 0:
                break
        return bytes(reversed(out))

    accumulated_delta = 0
    
    for part in notes_parts:
        for m in pattern.finditer(part):
            chord_content = m.group(1)
            if chord_content:
                chord_notes = []
                max_mult = 0.0
                for cn in note_pattern.finditer(chord_content):
                    acc = cn.group(1) or ""
                    pitch = cn.group(2)
                    octaves = cn.group(3) or ""
                    suffix = cn.group(4) or ""
                    
                    midi_note = get_midi_note(acc, pitch, octaves)
                    suffix_match = re.match(r'^(\d+)?((?:/+\d*)*)$', suffix)
                    mult = 1.0
                    if suffix_match:
                        mult = parse_multiplier(suffix_match.group(1), suffix_match.group(2))
                    if mult > max_mult:
                        max_mult = mult
                    if midi_note is not None:
                        chord_notes.append(midi_note)
                        
                chord_ticks = int(max_mult * unit_ticks)
                
                if chord_notes:
                    for idx, note in enumerate(chord_notes):
                        delta = accumulated_delta if idx == 0 else 0
                        track_events.extend(to_vlq(delta))
                        track_events.extend([0x90, note, 96])
                    accumulated_delta = 0
                    
                    for idx, note in enumerate(chord_notes):
                        delta = chord_ticks if idx == 0 else 0
                        track_events.extend(to_vlq(delta))
                        track_events.extend([0x80, note, 0])
                else:
                    accumulated_delta += chord_ticks
            else:
                acc = m.group(2) or ""
                pitch = m.group(3)
                octaves = m.group(4) or ""
                suffix = m.group(5) or ""
                
                midi_note = get_midi_note(acc, pitch, octaves)
                suffix_match = re.match(r'^(\d+)?((?:/+\d*)*)$', suffix)
                mult = 1.0
                if suffix_match:
                    mult = parse_multiplier(suffix_match.group(1), suffix_match.group(2))
                note_ticks = int(mult * unit_ticks)
                
                if midi_note is not None:
                    track_events.extend(to_vlq(accumulated_delta))
                    track_events.extend([0x90, midi_note, 96])
                    accumulated_delta = 0
                    
                    track_events.extend(to_vlq(note_ticks))
                    track_events.extend([0x80, midi_note, 0])
                else:
                    accumulated_delta += note_ticks
                    
    track_events.extend(to_vlq(accumulated_delta))
    track_events.extend([0xFF, 0x2F, 0x00])
    
    midi_file = bytearray()
    midi_file.extend(b'MThd')
    midi_file.extend((6).to_bytes(4, byteorder='big'))
    midi_file.extend((0).to_bytes(2, byteorder='big'))
    midi_file.extend((1).to_bytes(2, byteorder='big'))
    midi_file.extend(ticks_per_quarter.to_bytes(2, byteorder='big'))
    
    midi_file.extend(b'MTrk')
    midi_file.extend(len(track_events).to_bytes(4, byteorder='big'))
    midi_file.extend(track_events)
    
    return bytes(midi_file)


def get_abc_duration(filepath: str) -> float:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            abc_text = f.read()
    except Exception as e:
        print(f"Error reading ABC file {filepath}: {e}")
        return 180.0

    meter = 1.0  # Default: 4/4
    unit_note_len = None
    bpm = 120.0
    beat_fraction = None
    
    headers_done = False
    notes_parts = []
    
    header_pattern = re.compile(r'^([A-Z]):\s*(.*)$')
    
    for line in abc_text.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
            
        match = header_pattern.match(line)
        if match and not headers_done:
            key, val = match.group(1), match.group(2).strip()
            if key == 'M':
                if val.lower() in ('c', '4/4'):
                    meter = 1.0
                elif val.lower() == 'c|':
                    meter = 1.0
                else:
                    m = re.match(r'(\d+)/(\d+)', val)
                    if m:
                        meter = float(m.group(1)) / float(m.group(2))
            elif key == 'L':
                m = re.match(r'(\d+)/(\d+)', val)
                if m:
                    unit_note_len = float(m.group(1)) / float(m.group(2))
            elif key == 'Q':
                bpm_match = re.search(r'(\d+)\s*$', val)
                if bpm_match:
                    bpm = float(bpm_match.group(1))
                frac_match = re.search(r'(\d+)/(\d+)\s*=', val)
                if frac_match:
                    beat_fraction = float(frac_match.group(1)) / float(frac_match.group(2))
            elif key == 'K':
                headers_done = True
        else:
            cleaned_line = line.split('%')[0]
            cleaned_line = re.sub(r'"[^"]*"', '', cleaned_line)
            cleaned_line = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', cleaned_line)
            notes_parts.append(cleaned_line)
            
    if unit_note_len is None:
        unit_note_len = 0.0625 if meter < 0.75 else 0.125
            
    if beat_fraction is None:
        beat_fraction = unit_note_len
        
    total_multipliers = 0.0
    
    pattern = re.compile(r'\[([^\]]+)\]|([A-Ga-gxzXZ])([,\']*)(\d*(?:/+\d*)*)')
    note_pattern = re.compile(r'([A-Ga-gxzXZ])([,\']*)(\d*(?:/+\d*)*)')
    
    def parse_multiplier(num_str, slash_str):
        num = float(num_str) if num_str else 1.0
        if not slash_str:
            return num
        slash_count = slash_str.count('/')
        m = re.search(r'(\d+)$', slash_str)
        if m:
            denom = float(m.group(1))
            return num / denom
        else:
            return num / (2 ** slash_count)

    for part in notes_parts:
        for m in pattern.finditer(part):
            chord_content = m.group(1)
            if chord_content:
                max_mult = 0.0
                chord_notes = note_pattern.findall(chord_content)
                for cn in chord_notes:
                    suffix = cn[2]
                    suffix_match = re.match(r'^(\d+)?((?:/+\d*)*)$', suffix)
                    if suffix_match:
                        mult = parse_multiplier(suffix_match.group(1), suffix_match.group(2))
                        if mult > max_mult:
                            max_mult = mult
                total_multipliers += max_mult
            else:
                suffix = m.group(4)
                suffix_match = re.match(r'^(\d+)?((?:/+\d*)*)$', suffix)
                if suffix_match:
                    mult = parse_multiplier(suffix_match.group(1), suffix_match.group(2))
                    total_multipliers += mult
                    
    duration = (total_multipliers * unit_note_len) * 60.0 / (bpm * beat_fraction)
    return duration


# Safe pygame mixer initialization
class SafeMusicPlayer:
    def __init__(self, playlist_dir="audio"):
        self.playlist_dir = playlist_dir
        self.current_track = None
        self.volume = 1.0
        self.mixer_initialized = False
        self.simulated = False
        self.paused = False
        self.was_stopped = False
        self.seeked_while_paused = False
        self.track_duration = 0.0
        self.last_seek_position = 0.0
        self.last_play_time = None
        self.start_time = 0.0
        self.end_time = None

        # EQ: 10-band gains in dB, keyed by centre frequency (Hz)
        self.eq_gains: dict[int, float] = {
            32: 0.0, 64: 0.0, 125: 0.0, 250: 0.0, 500: 0.0,
            1000: 0.0, 2000: 0.0, 4000: 0.0, 8000: 0.0, 16000: 0.0,
        }
        self._eq_tmp_path: str | None = None  # path to the currently loaded temp EQ file
        self._eq_lock = threading.Lock()
        self._abc_tmp_path: str | None = None  # path to the currently compiled MIDI file

        # Set SDL_SOUNDFONTS to enable correct MIDI instrument synthesis on Linux
        soundfont_paths = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2",
            "/usr/share/sounds/sf2/TimGM6mb.sf2",
            "/usr/share/sounds/sf3/FluidR3_GM.sf3",
            "/usr/share/sounds/sf3/default.sf3",
            "/usr/share/midi/soundfont/FluidR3_GM.sf2",
            "/usr/share/midi/soundfont/default.sf2",
        ]
        for path in soundfont_paths:
            if os.path.exists(path):
                os.environ["SDL_SOUNDFONTS"] = path
                print(f"Set SDL_SOUNDFONTS environment variable to {path}")
                break

        try:
            pygame.mixer.init()
            self.mixer_initialized = True
            print("Successfully initialized Pygame mixer.")
        except Exception as e:
            self.simulated = True
            print(
                f"Warning: Could not initialize Pygame mixer (running in simulated mode): {e}"
            )

    def __del__(self):
        # Clean up any leftover temporary files when player is destroyed
        for path in (getattr(self, "_eq_tmp_path", None), getattr(self, "_abc_tmp_path", None)):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    @property
    def playlist_dir(self):
        return self._playlist_dir

    @playlist_dir.setter
    def playlist_dir(self, value):
        if value and not os.path.isabs(value):
            self._playlist_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), value
            )
        else:
            self._playlist_dir = value

    def play_track(self, track_file, fade_in_ms=1500, fade_out_ms=1500, start_time=0.0, end_time=None):
        if not track_file:
            self.stop(fade_out_ms)
            return True

        track_path = os.path.join(self.playlist_dir, track_file)
        if not os.path.exists(track_path):
            print(f"Warning: Track file not found: {track_path}")
            return False

        if self.current_track == track_file and getattr(self, "start_time", 0.0) == start_time and getattr(self, "end_time", None) == end_time:
            # Track is already playing with same bounds
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
        self.seeked_while_paused = False

        # Load track duration
        self.track_duration = 0.0
        if track_file.lower().endswith(".abc"):
            self.track_duration = get_abc_duration(track_path)
            try:
                with open(track_path, "r", encoding="utf-8") as f:
                    abc_text = f.read()
                midi_bytes = abc_to_midi_bytes(abc_text)
                tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False, prefix="abc_tmp_")
                tmp_path = tmp.name
                tmp.close()
                with open(tmp_path, "wb") as f_midi:
                    f_midi.write(midi_bytes)
                if self._abc_tmp_path and os.path.exists(self._abc_tmp_path):
                    try:
                        os.unlink(self._abc_tmp_path)
                    except OSError:
                        pass
                self._abc_tmp_path = tmp_path
            except Exception as e:
                print(f"Error compiling ABC to MIDI: {e}")
        else:
            try:
                tag = TinyTag.get(track_path)
                self.track_duration = tag.duration
            except Exception as e:
                print(f"Error loading track duration with TinyTag: {e}")
        if self.track_duration is None or self.track_duration == 0.0:
            self.track_duration = 180.0

        self.start_time = start_time
        self.end_time = end_time
        self.last_seek_position = self.start_time
        import time
        self.last_play_time = time.time()

        if self.simulated:
            print(f"[Playback SIMULATED] Playing: {track_file}")
            return True

        try:
            load_path = self._abc_tmp_path if track_file.lower().endswith(".abc") else track_path
            pygame.mixer.music.load(load_path)
            pygame.mixer.music.play(loops=-1, start=self.start_time, fade_ms=fade_in_ms)
            pygame.mixer.music.set_volume(self.volume)
            return True
        except Exception as e:
            print(f"Error during Pygame playback of {track_file}: {e}")
            return False

    def select_track(self, track_file, start_time=0.0, end_time=None):
        if not track_file:
            return False

        track_path = os.path.join(self.playlist_dir, track_file)
        if not os.path.exists(track_path):
            print(f"Warning: Track file not found: {track_path}")
            return False

        # If there is currently a track playing or paused, stop it
        if self.current_track:
            if not self.simulated and self.mixer_initialized:
                try:
                    pygame.mixer.music.stop()
                except Exception as e:
                    print(f"Error stopping music: {e}")

        self.current_track = track_file
        self.paused = True
        self.was_stopped = True
        self.seeked_while_paused = False

        # Load track duration
        self.track_duration = 0.0
        if track_file.lower().endswith(".abc"):
            self.track_duration = get_abc_duration(track_path)
            try:
                with open(track_path, "r", encoding="utf-8") as f:
                    abc_text = f.read()
                midi_bytes = abc_to_midi_bytes(abc_text)
                tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False, prefix="abc_tmp_")
                tmp_path = tmp.name
                tmp.close()
                with open(tmp_path, "wb") as f_midi:
                    f_midi.write(midi_bytes)
                if self._abc_tmp_path and os.path.exists(self._abc_tmp_path):
                    try:
                        os.unlink(self._abc_tmp_path)
                    except OSError:
                        pass
                self._abc_tmp_path = tmp_path
            except Exception as e:
                print(f"Error compiling ABC to MIDI: {e}")
        else:
            try:
                tag = TinyTag.get(track_path)
                self.track_duration = tag.duration
            except Exception as e:
                print(f"Error loading track duration with TinyTag: {e}")
        if self.track_duration is None or self.track_duration == 0.0:
            self.track_duration = 180.0

        self.start_time = start_time
        self.end_time = end_time
        self.last_seek_position = self.start_time
        self.last_play_time = None

        if self.simulated:
            print(f"[Playback SIMULATED] Selected track (stopped): {track_file}")
            return True

        try:
            load_path = self._abc_tmp_path if track_file.lower().endswith(".abc") else track_path
            pygame.mixer.music.load(load_path)
            return True
        except Exception as e:
            print(f"Error loading Pygame track {track_file}: {e}")
            return False

    def stop(self, fade_out_ms=1500):
        if not self.current_track:
            return

        print(f"[Playback] Stopping playback (fadeout: {fade_out_ms}ms)")
        self.paused = True
        self.was_stopped = True
        self.seeked_while_paused = False
        self.last_seek_position = self.start_time
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
        self.seeked_while_paused = False
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
            if self.was_stopped or getattr(self, "seeked_while_paused", False):
                if self.last_seek_position < self.start_time:
                    self.last_seek_position = self.start_time
            self.was_stopped = False
            self.seeked_while_paused = False
            return True
        try:
            if self.was_stopped or getattr(self, "seeked_while_paused", False):
                start_pos = self.last_seek_position
                if start_pos < self.start_time:
                    start_pos = self.start_time
                track_path = os.path.join(self.playlist_dir, self.current_track)
                load_path = self._abc_tmp_path if self.current_track.lower().endswith(".abc") else track_path
                pygame.mixer.music.load(load_path)
                pygame.mixer.music.play(loops=-1, start=start_pos, fade_ms=1500)
                pygame.mixer.music.set_volume(self.volume)
                self.was_stopped = False
                self.seeked_while_paused = False
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
        print(f"[Playback] Seeking to {position}s (was_stopped={self.was_stopped}, paused={self.paused})")

        self.last_seek_position = position
        import time

        if self.was_stopped or self.paused:
            # When stopped or paused, seeking only updates the current position marker without playing/unpausing
            self.last_play_time = None
            if self.paused and not self.was_stopped:
                self.seeked_while_paused = True
            return True

        # Otherwise continue current behavior (resume/start playback)
        self.last_play_time = time.time()
        self.paused = False
        self.was_stopped = False
        self.seeked_while_paused = False

        if self.simulated:
            return True

        try:
            track_path = os.path.join(self.playlist_dir, self.current_track)
            load_path = self._abc_tmp_path if self.current_track.lower().endswith(".abc") else track_path
            pygame.mixer.music.load(load_path)
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

    # ------------------------------------------------------------------
    # EQ  (10-band peaking IIR filters via scipy, applied to a 30s window)
    # ------------------------------------------------------------------

    # EQ_BANDS: (centre_hz, octave_width_Q)
    _EQ_BANDS: list[tuple[int, float]] = [
        (32, 1.0), (64, 1.0), (125, 1.0), (250, 1.0), (500, 1.0),
        (1000, 1.0), (2000, 1.0), (4000, 1.0), (8000, 1.0), (16000, 1.0),
    ]
    _WINDOW_SEC = 30  # seconds of audio to filter per EQ apply

    @staticmethod
    def _peaking_sos(fc: float, gain_db: float, Q: float, fs: int) -> np.ndarray:
        """Return a 2nd-order peaking EQ filter as a single SOS row."""
        # Build peaking coefficients manually (Audio EQ Cookbook).
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / (2 * Q)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        # Return as SOS row: [b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

    def apply_eq(self) -> dict:
        """Apply current eq_gains to the active track and hot-reload pygame.

        Reads a _WINDOW_SEC window centred on the current playback position,
        filters it with peaking IIR filters, writes a temp WAV, and reloads
        pygame playback from position 0 of the window (seeking back seamlessly).

        Returns a dict with 'status' and 'message'.
        """
        if not self.current_track:
            return {"status": "error", "message": "No track loaded."}

        # Check if all gains are 0 — if so, just reload the original file
        all_flat = all(abs(g) < 0.01 for g in self.eq_gains.values())

        track_path = os.path.join(self.playlist_dir, self.current_track)
        if not os.path.exists(track_path):
            return {"status": "error", "message": f"Track file not found: {track_path}"}

        capture_pos = self.get_current_position()

        with self._eq_lock:
            try:
                import soundfile as sf  # type: ignore[import]
                from scipy.signal import sosfilt  # type: ignore[import]

                # --- Read window ---
                info = sf.info(track_path)
                fs = info.samplerate
                total_frames = info.frames

                win_frames = int(self._WINDOW_SEC * fs)
                start_frame = max(0, int(capture_pos * fs) - win_frames // 4)
                end_frame = min(total_frames, start_frame + win_frames)
                start_frame = max(0, end_frame - win_frames)  # clamp

                audio, _ = sf.read(track_path, start=start_frame, stop=end_frame, dtype="float32", always_2d=True)

                if not all_flat:
                    # --- Build and apply filter chain ---
                    for (fc, Q) in self._EQ_BANDS:
                        gain_db = self.eq_gains.get(fc, 0.0)
                        if abs(gain_db) < 0.01:
                            continue
                        if fc >= fs / 2:  # skip bands above Nyquist
                            continue
                        sos = self._peaking_sos(float(fc), float(gain_db), float(Q), fs)
                        audio = sosfilt(sos, audio, axis=0).astype(np.float32)
                    # Clamp to prevent clipping
                    audio = np.clip(audio, -1.0, 1.0)

                # --- Write temp WAV ---
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="eq_tmp_")
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, audio, fs)

                # --- Reload pygame from the window start ---
                resume_offset = capture_pos - (start_frame / fs)
                resume_offset = max(0.0, min(resume_offset, self._WINDOW_SEC - 0.5))

                was_playing = not self.paused and not self.was_stopped

                if not self.simulated and self.mixer_initialized:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(tmp_path)
                    if was_playing:
                        pygame.mixer.music.play(loops=0, start=resume_offset, fade_ms=80)
                        pygame.mixer.music.set_volume(self.volume)

                # Track position so the seek timer stays accurate
                import time
                self.last_seek_position = capture_pos
                self.last_play_time = time.time() if was_playing else None

                # Clean up previous temp file
                if self._eq_tmp_path and self._eq_tmp_path != tmp_path:
                    try:
                        os.unlink(self._eq_tmp_path)
                    except OSError:
                        pass
                self._eq_tmp_path = tmp_path

                return {"status": "success", "message": "EQ applied."}

            except ImportError as e:
                return {"status": "error", "message": f"Missing dependency for EQ: {e}. Install soundfile."}
            except Exception as e:
                print(f"[EQ] Error applying EQ: {e}")
                return {"status": "error", "message": str(e)}


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
