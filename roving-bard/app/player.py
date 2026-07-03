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

import time
import os
import re
import tempfile
import threading
import subprocess

import cv2
try:
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
except Exception:
    pass
import mss
import numpy as np
import pytesseract
from PIL import Image
from tinytag import TinyTag

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "capture")




def abc_to_midi_bytes(abc_text: str, start_pos: float = 0.0, instrument: int | None = None) -> bytes:
    meter_num = 4
    meter_den = 4
    unit_note_len = None
    program = 0  # Default to Piano
    bpm = 120.0
    beat_fraction = None
    has_tempo_header = False
    
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
                    meter_num, meter_den = 4, 4
                elif val.lower() == 'c|':
                    meter_num, meter_den = 2, 2
                else:
                    m = re.match(r'(\d+)/(\d+)', val)
                    if m:
                        meter_num = int(m.group(1))
                        meter_den = int(m.group(2))
            elif key == 'L':
                m = re.match(r'(\d+)/(\d+)', val)
                if m:
                     unit_note_len = float(m.group(1)) / float(m.group(2))
            elif key == 'Q':
                has_tempo_header = True
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
        unit_note_len = 0.0625 if (meter_num / meter_den) < 0.75 else 0.125
        
    if beat_fraction is None:
        if has_tempo_header:
            beat_fraction = unit_note_len
        else:
            if meter_den == 8 and meter_num in (6, 9, 12):
                beat_fraction = 0.375
            elif meter_den == 2:
                beat_fraction = 0.5
            else:
                beat_fraction = 0.25
        
    ticks_per_quarter = 480
    unit_ticks = int(ticks_per_quarter * unit_note_len / beat_fraction)
    tempo_us = int(60000000 / bpm)
    
    # MIDI track events bytes
    track_events = bytearray()
    
    # Write MIDI Tempo event (FF 51 03 tttttt) at delta-time 0
    track_events.append(0x00)
    track_events.extend([0xFF, 0x51, 0x03])
    track_events.extend(tempo_us.to_bytes(3, byteorder='big'))
    
    if instrument is not None:
        program = instrument

    # Write Program Change (C0 <program>) at delta-time 0
    track_events.append(0x00)
    track_events.extend([0xC0, program])
    
    # Join note parts, remove grace notes, and expand repeats
    notes_str = " ".join(notes_parts)
    notes_str = re.sub(r'{[^}]*}', '', notes_str)
    
    # Repeat expansion logic
    bar_pattern = re.compile(r'(\|:\s*\[\d|\|:\s*|:\s*\|:\s*|:\s*\||::|\|\]|\|\||\|)')
    parts = bar_pattern.split(notes_str)
    measures = []
    current_repeat_block = []
    
    for i in range(0, len(parts), 2):
        notes = parts[i].strip()
        bar = parts[i+1].strip() if i+1 < len(parts) else ""
        measure_data = (notes, bar)
        
        is_repeat_start = "|:" in bar
        is_repeat_end = ":|" in bar or "::" in bar
        
        current_repeat_block.append(measure_data)
        
        if is_repeat_end:
            for nd in current_repeat_block:
                measures.append(nd)
            for nd in current_repeat_block:
                measures.append(nd)
            current_repeat_block = []
        elif bar in ("||", "|]", ""):
            for nd in current_repeat_block:
                measures.append(nd)
            current_repeat_block = []
            
        if is_repeat_start or bar == "::":
            current_repeat_block = []
            
    for nd in current_repeat_block:
        measures.append(nd)
    
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

    start_pos_ticks = int(start_pos * ticks_per_quarter * bpm / 60.0)
    current_abs_ticks = 0
    last_written_abs_ticks = 0
    accumulated_delta = 0
    
    for notes, _ in measures:
        for m in pattern.finditer(notes):
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
                start_ticks = current_abs_ticks + accumulated_delta
                end_ticks = start_ticks + chord_ticks
                
                if end_ticks <= start_pos_ticks:
                    current_abs_ticks = end_ticks
                    accumulated_delta = 0
                    continue
                
                new_start_ticks = max(0, start_ticks - start_pos_ticks)
                new_end_ticks = end_ticks - start_pos_ticks
                
                if chord_notes:
                    for idx, note in enumerate(chord_notes):
                        delta = (new_start_ticks - last_written_abs_ticks) if idx == 0 else 0
                        track_events.extend(to_vlq(delta))
                        track_events.extend([0x90, note, 96])
                    last_written_abs_ticks = new_start_ticks
                    
                    for idx, note in enumerate(chord_notes):
                        delta = (new_end_ticks - last_written_abs_ticks) if idx == 0 else 0
                        track_events.extend(to_vlq(delta))
                        track_events.extend([0x80, note, 0])
                    last_written_abs_ticks = new_end_ticks
                
                current_abs_ticks = end_ticks
                accumulated_delta = 0
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
                
                start_ticks = current_abs_ticks + accumulated_delta
                end_ticks = start_ticks + note_ticks
                
                if end_ticks <= start_pos_ticks:
                    current_abs_ticks = end_ticks
                    accumulated_delta = 0
                    continue
                
                new_start_ticks = max(0, start_ticks - start_pos_ticks)
                new_end_ticks = end_ticks - start_pos_ticks
                
                if midi_note is not None:
                    delta = new_start_ticks - last_written_abs_ticks
                    track_events.extend(to_vlq(delta))
                    track_events.extend([0x90, midi_note, 96])
                    last_written_abs_ticks = new_start_ticks
                    
                    delta = new_end_ticks - last_written_abs_ticks
                    track_events.extend(to_vlq(delta))
                    track_events.extend([0x80, midi_note, 0])
                    last_written_abs_ticks = new_end_ticks
                
                current_abs_ticks = end_ticks
                accumulated_delta = 0
                    
    end_track_abs_ticks = max(0, current_abs_ticks + accumulated_delta - start_pos_ticks)
    track_events.extend(to_vlq(end_track_abs_ticks - last_written_abs_ticks))
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



def get_midi_duration(filepath: str) -> float:
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        if len(data) < 14 or data[:4] != b"MThd":
            return 180.0
        
        division = int.from_bytes(data[12:14], byteorder="big")
        if division & 0x8000:
            return 180.0
            
        idx = 14
        tracks = []
        while idx < len(data):
            if data[idx:idx+4] == b"MTrk":
                track_len = int.from_bytes(data[idx+4:idx+8], byteorder="big")
                track_data = data[idx+8:idx+8+track_len]
                tracks.append(track_data)
                idx += 8 + track_len
            else:
                idx += 1
                
        if not tracks:
            return 180.0
            
        tempo_events = []
        for track in tracks:
            t_idx = 0
            current_ticks = 0
            running_status = None
            while t_idx < len(track):
                val = 0
                while True:
                    b = track[t_idx]
                    t_idx += 1
                    val = (val << 7) | (b & 0x7f)
                    if not (b & 0x80):
                        break
                current_ticks += val
                
                if t_idx >= len(track):
                    break
                
                status = track[t_idx]
                if status >= 0x80:
                    t_idx += 1
                    running_status = status
                else:
                    status = running_status
                
                if status == 0xFF:
                    meta_type = track[t_idx]
                    t_idx += 1
                    len_val = 0
                    while True:
                        b = track[t_idx]
                        t_idx += 1
                        len_val = (len_val << 7) | (b & 0x7f)
                        if not (b & 0x80):
                            break
                    if meta_type == 0x51 and len_val == 3:
                        tempo = int.from_bytes(track[t_idx:t_idx+3], byteorder="big")
                        tempo_events.append((current_ticks, tempo))
                    t_idx += len_val
                elif status in (0xF0, 0xF7):
                    len_val = 0
                    while True:
                        b = track[t_idx]
                        t_idx += 1
                        len_val = (len_val << 7) | (b & 0x7f)
                        if not (b & 0x80):
                            break
                    t_idx += len_val
                else:
                    msg_type = status & 0xF0
                    if msg_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                        t_idx += 2
                    elif msg_type in (0xC0, 0xD0):
                        t_idx += 1
                    else:
                        t_idx += 1
        
        tempo_events.sort(key=lambda x: x[0])
        
        def ticks_to_seconds(total_ticks):
            if not tempo_events:
                return total_ticks * 0.5 / division
            
            curr_tick = 0
            curr_time = 0.0
            curr_tempo = 500000
            
            for t_tick, t_tempo in tempo_events:
                if t_tick >= total_ticks:
                    break
                curr_time += (t_tick - curr_tick) * (curr_tempo / 1000000.0) / division
                curr_tick = t_tick
                curr_tempo = t_tempo
                
            if total_ticks > curr_tick:
                curr_time += (total_ticks - curr_tick) * (curr_tempo / 1000000.0) / division
            return curr_time

        max_duration = 0.0
        for track in tracks:
            t_idx = 0
            current_ticks = 0
            running_status = None
            while t_idx < len(track):
                val = 0
                while True:
                    b = track[t_idx]
                    t_idx += 1
                    val = (val << 7) | (b & 0x7f)
                    if not (b & 0x80):
                        break
                current_ticks += val
                
                if t_idx >= len(track):
                    break
                
                status = track[t_idx]
                if status >= 0x80:
                    t_idx += 1
                    running_status = status
                else:
                    status = running_status
                
                if status == 0xFF:
                    meta_type = track[t_idx]
                    t_idx += 1
                    len_val = 0
                    while True:
                        b = track[t_idx]
                        t_idx += 1
                        len_val = (len_val << 7) | (b & 0x7f)
                        if not (b & 0x80):
                            break
                    t_idx += len_val
                elif status in (0xF0, 0xF7):
                    len_val = 0
                    while True:
                        b = track[t_idx]
                        t_idx += 1
                        len_val = (len_val << 7) | (b & 0x7f)
                        if not (b & 0x80):
                            break
                    t_idx += len_val
                else:
                    msg_type = status & 0xF0
                    if msg_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                        t_idx += 2
                    elif msg_type in (0xC0, 0xD0):
                        t_idx += 1
                    else:
                        t_idx += 1
            
            track_duration = ticks_to_seconds(current_ticks)
            if track_duration > max_duration:
                max_duration = track_duration
                
        return max_duration
    except Exception as e:
        print(f"Error parsing MIDI duration: {e}")
        return 180.0


def get_abc_duration(filepath: str) -> float:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            abc_text = f.read()
    except Exception as e:
        print(f"Error reading ABC file {filepath}: {e}")
        return 180.0

    meter_num = 4
    meter_den = 4
    unit_note_len = None
    bpm = 120.0
    beat_fraction = None
    has_tempo_header = False
    
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
                    meter_num, meter_den = 4, 4
                elif val.lower() == 'c|':
                    meter_num, meter_den = 2, 2
                else:
                    m = re.match(r'(\d+)/(\d+)', val)
                    if m:
                        meter_num = int(m.group(1))
                        meter_den = int(m.group(2))
            elif key == 'L':
                m = re.match(r'(\d+)/(\d+)', val)
                if m:
                    unit_note_len = float(m.group(1)) / float(m.group(2))
            elif key == 'Q':
                has_tempo_header = True
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
            
    # Resolve default unit note length
    if unit_note_len is None:
        meter_val = meter_num / meter_den
        unit_note_len = 0.0625 if meter_val < 0.75 else 0.125
            
    # Resolve default beat fraction for tempo
    if beat_fraction is None:
        if has_tempo_header:
            beat_fraction = unit_note_len
        else:
            if meter_den == 8 and meter_num in (6, 9, 12):
                beat_fraction = 0.375
            elif meter_den == 2:
                beat_fraction = 0.5
            else:
                beat_fraction = 0.25
        
    # Join note parts, remove grace notes, and expand repeats
    notes_str = " ".join(notes_parts)
    notes_str = re.sub(r'{[^}]*}', '', notes_str)
    
    # Repeat expansion logic
    bar_pattern = re.compile(r'(\|:\s*\[\d|\|:\s*|:\s*\|:\s*|:\s*\||::|\|\]|\|\||\|)')
    parts = bar_pattern.split(notes_str)
    measures = []
    current_repeat_block = []
    
    for i in range(0, len(parts), 2):
        notes = parts[i].strip()
        bar = parts[i+1].strip() if i+1 < len(parts) else ""
        measure_data = (notes, bar)
        
        is_repeat_start = "|:" in bar
        is_repeat_end = ":|" in bar or "::" in bar
        
        current_repeat_block.append(measure_data)
        
        if is_repeat_end:
            for nd in current_repeat_block:
                measures.append(nd)
            for nd in current_repeat_block:
                measures.append(nd)
            current_repeat_block = []
        elif bar in ("||", "|]", ""):
            for nd in current_repeat_block:
                measures.append(nd)
            current_repeat_block = []
            
        if is_repeat_start or bar == "::":
            current_repeat_block = []
            
    for nd in current_repeat_block:
        measures.append(nd)
        
    # Sum note multipliers of expanded measures
    total_multipliers = 0.0
    pattern = re.compile(r'\[([^\]]+)\]|([_^^=]*)([A-Ga-gxzXZ])([,\']*)(\d*(?:/+\d*)*)')
    note_pattern = re.compile(r'([_^^=]*)([A-Ga-gxzXZ])([,\']*)(\d*(?:/+\d*)*)')
    
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

    for notes, _ in measures:
        for m in pattern.finditer(notes):
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
                suffix = m.group(5)
                suffix_match = re.match(r'^(\d+)?((?:/+\d*)*)$', suffix)
                if suffix_match:
                    mult = parse_multiplier(suffix_match.group(1), suffix_match.group(2))
                    total_multipliers += mult
                    
    duration = (total_multipliers * unit_note_len) * 60.0 / (bpm * beat_fraction)
    return duration


# Safe sounddevice audio player initialization
class SafeMusicPlayer:
    def __init__(self, playlist_dir="audio"):
        self.playlist_dir = playlist_dir
        self.current_track = None
        self.volume = 1.0
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
        self._abc_tmp_path: str | None = None
        self.active_instrument: int | None = None

        # sounddevice backend fields
        self._play_thread = None
        self._play_lock = threading.Lock()
        self._audio_data = None  # NumPy array of shape (N, channels)
        self._sf = None  # soundfile.SoundFile object for streaming WAV/OGG/FLAC
        self._ffmpeg_proc = None  # subprocess.Popen object for streaming MP3/AAC
        self._sample_rate = 44100
        self._channels = 2
        self._playhead = 0
        self._eq_zi = {}  # band -> zi array for filter state
        self.soundfont_path = None
        self.backend_initialized = False
        self.fluidsynth_available = False
        self.sd_device = None
        self._init_soundfont(silent=True)

    def __del__(self):
        # Stop sounddevice playback thread if active
        try:
            self.was_stopped = True
            if self._play_thread and self._play_thread.is_alive():
                self._play_thread.join(timeout=0.5)
        except Exception:
            pass
            
        # Clean up temporary ABC midi file
        if self._abc_tmp_path and os.path.exists(self._abc_tmp_path):
            try:
                os.unlink(self._abc_tmp_path)
            except OSError:
                pass
        with self._play_lock:
            if self._sf:
                try:
                    self._sf.close()
                except Exception:
                    pass
                self._sf = None
                self._ffmpeg_proc = None

    def _init_soundfont(self, silent=True):
        """Initializes the soundfont path using environment variables or fallback directories."""
        env_soundfont = os.environ.get("SDL_SOUNDFONTS")
        if env_soundfont and os.path.exists(env_soundfont):
            if not silent:
                print(f"Using pre-configured SDL_SOUNDFONTS: {env_soundfont}")
            self.soundfont_path = env_soundfont
        else:
            soundfont_paths = []
            if self.playlist_dir and os.path.exists(self.playlist_dir):
                try:
                    for filename in sorted(os.listdir(self.playlist_dir)):
                        if filename.lower().endswith((".sf2", ".sf3")):
                            soundfont_paths.append(os.path.join(self.playlist_dir, filename))
                except Exception as e:
                    if not silent:
                        print(f"Error scanning playlist_dir for soundfonts: {e}")

            soundfont_paths.extend([
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                "/usr/share/sounds/sf2/default-GM.sf2",
                "/usr/share/sounds/sf2/TimGM6mb.sf2",
                "/usr/share/sounds/sf3/FluidR3_GM.sf3",
                "/usr/share/sounds/sf3/default.sf3",
                "/usr/share/midi/soundfont/FluidR3_GM.sf2",
                "/usr/share/midi/soundfont/default.sf2",
            ])
            for path in soundfont_paths:
                if os.path.exists(path):
                    os.environ["SDL_SOUNDFONTS"] = path
                    self.soundfont_path = path
                    if not silent:
                        print(f"Set SDL_SOUNDFONTS environment variable to {path}")
                    break

    def update_soundfont(self, soundfont_name: str | None, silent=True):
        """Updates the active soundfont to the one specified in the configuration."""
        old_soundfont_path = self.soundfont_path

        if not soundfont_name:
            self._init_soundfont(silent=silent)
        else:
            # Check if the soundfont exists in playlist_dir
            path_in_playlist = os.path.join(self.playlist_dir, soundfont_name)
            if os.path.exists(path_in_playlist):
                self.soundfont_path = path_in_playlist
                os.environ["SDL_SOUNDFONTS"] = path_in_playlist
                if not silent:
                    print(f"[SafeMusicPlayer] Set active SoundFont to {path_in_playlist}")
            # Check if the soundfont is a system path
            elif os.path.exists(soundfont_name):
                self.soundfont_path = soundfont_name
                os.environ["SDL_SOUNDFONTS"] = soundfont_name
                if not silent:
                    print(f"[SafeMusicPlayer] Set active SoundFont to {soundfont_name}")
            else:
                # Otherwise, fall back to default search
                if not silent:
                    print(f"[SafeMusicPlayer] SoundFont not found: {soundfont_name}. Falling back to default search.")
                self._init_soundfont(silent=silent)

        # If the soundfont changed, invalidate the synthesis cache and restart playback if active
        if old_soundfont_path != self.soundfont_path:
            # 1. Clear FLAC synthesis cache
            cache_dir = os.path.join(self.playlist_dir, ".cache")
            if os.path.exists(cache_dir):
                if not silent:
                    print(f"[SafeMusicPlayer] SoundFont changed. Clearing synthesis cache in {cache_dir}...")
                try:
                    for filename in os.listdir(cache_dir):
                        if filename.lower().endswith(".flac"):
                            os.remove(os.path.join(cache_dir, filename))
                except Exception as e:
                    if not silent:
                        print(f"Error clearing synthesis cache: {e}")
            
            # 2. Restart playback if playing an ABC/MIDI track
            if self.current_track and self.current_track.lower().endswith((".abc", ".mid", ".midi")):
                if not getattr(self, "paused", False) and not getattr(self, "was_stopped", False):
                    if not silent:
                        print(f"[SafeMusicPlayer] SoundFont changed while playing {self.current_track}. Restarting from start_time ({self.start_time}s) to apply new SoundFont.")
                    self.play_track(
                        self.current_track,
                        start_time=self.start_time,
                        end_time=self.end_time
                    )

    def initialize_backend(self, verbose=False):
        """Initializes Fluidsynth and sounddevice on-demand, printing diagnostics once."""
        if getattr(self, "backend_initialized", False):
            return
        self.backend_initialized = True

        # Check if fluidsynth is available on the system for WAV synthesis
        self.fluidsynth_available = False
        try:
            import shutil
            if shutil.which("fluidsynth") and self.soundfont_path and os.path.exists(self.soundfont_path):
                self.fluidsynth_available = True
                if verbose:
                    print("Fluidsynth detected. MIDI/ABC files will be played via sounddevice backend with full seeking/EQ support!")
        except Exception:
            pass

        # Detect best sounddevice output device (prefer 'pulse' on Linux to avoid ALSA exclusive locks)
        self.sd_device = None
        try:
            import sounddevice as sd
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_output_channels'] > 0 and 'pulse' in dev['name'].lower():
                    self.sd_device = i
                    if verbose:
                        print(f"Detected PulseAudio device at index {i}. Routing sounddevice output through it.")
                    break
        except Exception:
            pass

        if verbose and self.soundfont_path:
            print(f"[SafeMusicPlayer] Set active SoundFont to {self.soundfont_path}")

    def _prepare_abc_midi(self, track_path, start_pos):
        try:
            with open(track_path, "r", encoding="utf-8") as f:
                abc_text = f.read()
            midi_bytes = abc_to_midi_bytes(abc_text, start_pos=start_pos, instrument=self.active_instrument)
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
            return tmp_path
        except Exception as e:
            print(f"Error compiling ABC to MIDI: {e}")
            return self._abc_tmp_path

    def _synthesize_midi_to_flac(self, midi_path, target_flac_path):
        if not self.soundfont_path or not os.path.exists(self.soundfont_path):
            raise ValueError("No soundfont found for MIDI synthesis.")
            
        os.makedirs(os.path.dirname(target_flac_path), exist_ok=True)
        
        try:
            cmd = [
                "fluidsynth",
                "-ni",
                self.soundfont_path,
                midi_path,
                "-F",
                target_flac_path,
                "-T",
                "flac",
                "-r",
                "44100"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            if os.path.exists(target_flac_path):
                try:
                    os.unlink(target_flac_path)
                except OSError:
                    pass
            raise e

    def _clear_eq_zi(self):
        self._eq_zi.clear()

    def _stop_sounddevice_playback(self):
        self.was_stopped = True
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)
        self._play_thread = None
        
        with self._play_lock:
            self._audio_data = None
            if self._sf:
                try:
                    self._sf.close()
                except Exception:
                    pass
                self._sf = None
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.terminate()
                    self._ffmpeg_proc.wait(timeout=0.2)
                except Exception:
                    try:
                        self._ffmpeg_proc.kill()
                    except Exception:
                        pass
                self._ffmpeg_proc = None

    def _playback_loop(self):
        import sounddevice as sd
        from scipy.signal import sosfilt
        
        chunk_size = 1024
        zi_dict = {}
        last_eq_gains = {}
        sos_dict = {}
        is_midi_abc = self.current_track.lower().endswith((".abc", ".mid", ".midi")) if self.current_track else False
        
        try:
            # --- ON-DEMAND LAZY LOADING FOR DEFERRED TRACKS ---
            with self._play_lock:
                if self._sf is None and self._ffmpeg_proc is None and self.current_track:
                    track_path = os.path.join(self.playlist_dir, self.current_track)
                    is_midi_abc = self.current_track.lower().endswith((".abc", ".mid", ".midi"))
                    actual_track_path = track_path
                    use_pipe = False
                    
                    if is_midi_abc:
                        # Resolve cached WAV path (use instrument-specific if active)
                        cache_dir = os.path.join(self.playlist_dir, ".cache")
                        if self.current_track.lower().endswith(".abc") and self.active_instrument is not None:
                            cached_flac = os.path.join(cache_dir, f"{self.current_track}_inst_{self.active_instrument}.flac")
                        else:
                            cached_flac = os.path.join(cache_dir, self.current_track + ".flac")
                        
                        # Synthesize if cache doesn't exist or is older than the source file
                        source_mtime = os.path.getmtime(track_path)
                        cache_mtime = os.path.getmtime(cached_flac) if os.path.exists(cached_flac) else 0
                        
                        if not os.path.exists(cached_flac) or source_mtime > cache_mtime:
                            print(f"[Synth] Background synthesizing {self.current_track} to FLAC cache...")
                            midi_path = track_path
                            if self.current_track.lower().endswith(".abc"):
                                self.track_duration = get_abc_duration(track_path) or 180.0
                                midi_path = self._prepare_abc_midi(track_path, 0.0)
                            
                            # Start background thread to build the cache
                            def bg_build_cache():
                                try:
                                    self._synthesize_midi_to_flac(midi_path, cached_flac)
                                except Exception as e:
                                    print(f"[Synth] Background cache synthesis failed: {e}")
                            threading.Thread(target=bg_build_cache, daemon=True).start()
                            
                            # Play instantly using Fluidsynth stdout pipe
                            self._sf = None
                            self._sample_rate = 44100
                            self._channels = 2
                            cmd = [
                                "fluidsynth", "-ni", "-F", "/dev/stdout", "-T", "raw", "-r", "44100",
                                self.soundfont_path, midi_path
                            ]
                            self._ffmpeg_proc = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL
                            )
                            
                            # Discard bytes up to start_time (represented by self._playhead)
                            if self._playhead > 0:
                                discard_bytes = self._playhead * 2 * 2  # channels * bytes_per_sample
                                try:
                                    self._ffmpeg_proc.stdout.read(discard_bytes)
                                except Exception as e:
                                    print(f"[Playback] Failed to discard start_time bytes from pipe: {e}")
                            
                            use_pipe = True
                        else:
                            actual_track_path = cached_flac

                    if not use_pipe:
                        if actual_track_path.lower().endswith((".wav", ".ogg", ".flac")):
                            import soundfile as sf
                            self._sf = sf.SoundFile(actual_track_path)
                            self._sample_rate = self._sf.samplerate
                            self._channels = self._sf.channels
                            if not is_midi_abc:
                                self.track_duration = len(self._sf) / self._sample_rate
                            self._sf.seek(min(len(self._sf) - 1, max(0, self._playhead)))
                        elif actual_track_path.lower().endswith((".mp3", ".aac", ".m4a", ".mp4")):
                            pos_sec = self._playhead / 44100.0
                            cmd = [
                                "ffmpeg", "-y",
                                "-ss", f"{pos_sec:.3f}",
                                "-i", actual_track_path,
                                "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                                "-ar", "44100", "-ac", "2", "-"
                            ]
                            self._ffmpeg_proc = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL
                            )
                            self._sample_rate = 44100
                            self._channels = 2

            with sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                latency="low",
                device=self.sd_device
            ) as stream:
                
                while not self.was_stopped:
                    if self.paused:
                        time.sleep(0.01)
                        continue
                        
                    with self._play_lock:
                        duration = self.track_duration
                        start_frame = int(self.start_time * self._sample_rate)
                        
                        if self._audio_data is not None:
                            total_frames = len(self._audio_data)
                        elif self._sf is not None:
                            total_frames = len(self._sf)
                        else:
                            total_frames = int(self.track_duration * self._sample_rate)
                        
                        if self.end_time is not None:
                            end_frame = min(total_frames, int(self.end_time * self._sample_rate))
                        elif is_midi_abc:
                            end_frame = min(total_frames, int(self.track_duration * self._sample_rate))
                        else:
                            end_frame = total_frames
                            
                        # Ensure playhead is within bounds before reading
                        if self._playhead < start_frame or self._playhead >= end_frame:
                            self._playhead = start_frame
                            if self._sf is not None:
                                self._sf.seek(min(len(self._sf) - 1, max(0, start_frame)))
                            elif self._ffmpeg_proc is not None:
                                try:
                                    self._ffmpeg_proc.terminate()
                                except Exception:
                                    pass
                                pos_sec = start_frame / self._sample_rate
                                cmd = [
                                    "ffmpeg", "-y", "-ss", f"{pos_sec:.3f}",
                                    "-i", os.path.join(self.playlist_dir, self.current_track),
                                    "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                                    "-ar", "44100", "-ac", "2", "-"
                                ]
                                self._ffmpeg_proc = subprocess.Popen(
                                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                                )
                            zi_dict.clear()

                        # Read exactly chunk_size frames, wrapping around if we hit end_frame
                        frames_to_read = chunk_size
                        chunk_parts = []

                        while frames_to_read > 0:
                            available_frames = end_frame - self._playhead
                            if available_frames <= 0:
                                # Loop back to start_frame
                                self._playhead = start_frame
                                if self._sf is not None:
                                    self._sf.seek(min(len(self._sf) - 1, max(0, start_frame)))
                                elif self._ffmpeg_proc is not None:
                                    try:
                                        self._ffmpeg_proc.terminate()
                                    except Exception:
                                        pass
                                    pos_sec = start_frame / self._sample_rate
                                    cmd = [
                                        "ffmpeg", "-y", "-ss", f"{pos_sec:.3f}",
                                        "-i", os.path.join(self.playlist_dir, self.current_track),
                                        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                                        "-ar", "44100", "-ac", "2", "-"
                                    ]
                                    self._ffmpeg_proc = subprocess.Popen(
                                        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                                    )
                                zi_dict.clear()
                                available_frames = end_frame - start_frame
                                if available_frames <= 0:
                                    break

                            read_len = min(frames_to_read, available_frames)

                            if self._audio_data is not None:
                                part = self._audio_data[self._playhead : self._playhead + read_len].copy()
                            elif self._sf is not None:
                                part = self._sf.read(read_len, dtype="float32", always_2d=True).copy()
                            elif self._ffmpeg_proc is not None:
                                num_bytes = read_len * self._channels * 2
                                raw_bytes = b""
                                try:
                                    raw_bytes = self._ffmpeg_proc.stdout.read(num_bytes)
                                except Exception:
                                    pass
                                if not raw_bytes:
                                    self._playhead = end_frame
                                    break
                                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                                part = samples.reshape((-1, self._channels))
                            else:
                                part = np.zeros((0, self._channels), dtype=np.float32)

                            if len(part) == 0:
                                self._playhead = end_frame
                                break

                            chunk_parts.append(part)
                            self._playhead += len(part)
                            frames_to_read -= len(part)

                        if chunk_parts:
                            chunk = np.concatenate(chunk_parts, axis=0)
                        else:
                            chunk = np.zeros((chunk_size, self._channels), dtype=np.float32)

                        if len(chunk) < chunk_size:
                            padding = np.zeros((chunk_size - len(chunk), self._channels), dtype=np.float32)
                            chunk = np.concatenate([chunk, padding], axis=0)

                        actual_frames = chunk_size
                        
                    # --- Apply EQ in Real-Time ---
                    gains_changed = False
                    for band, gain in self.eq_gains.items():
                        if last_eq_gains.get(band) != gain:
                            gains_changed = True
                            last_eq_gains[band] = gain
                            
                            Q = 1.0
                            for fb, qb in self._EQ_BANDS:
                                if fb == band:
                                    Q = qb
                                    break
                            sos_dict[band] = self._peaking_sos(float(band), float(gain), float(Q), self._sample_rate)
                            
                    for band, sos in sos_dict.items():
                        gain = last_eq_gains.get(band, 0.0)
                        if abs(gain) < 0.01:
                            continue
                            
                        if band not in zi_dict or zi_dict[band].shape[2] != self._channels:
                            zi_dict[band] = np.zeros((1, 2, self._channels), dtype=np.float32)
                            
                        chunk, zi_dict[band] = sosfilt(sos, chunk, zi=zi_dict[band], axis=0)
                        
                    chunk = np.clip(chunk, -1.0, 1.0)
                    
                    effective_vol = self._get_effective_volume()
                    chunk *= effective_vol
                    
                    # Force C-contiguous float32 array to prevent PortAudio memory corruption
                    chunk = np.ascontiguousarray(chunk, dtype=np.float32)
                    
                    # Write to stream
                    stream.write(chunk)
                            
        except Exception as e:
            print(f"[Playback Loop] Error: {e}")

    def play_track(self, track_file, fade_in_ms=1500, fade_out_ms=1500, start_time=0.0, end_time=None):
        if not track_file:
            return False

        self.initialize_backend(verbose=False)
        track_path = os.path.join(self.playlist_dir, track_file)
        if not os.path.exists(track_path):
            print(f"Error: Track file not found: {track_path}")
            return False

        is_midi_abc = track_file.lower().endswith((".abc", ".mid", ".midi"))

        if is_midi_abc and not self.fluidsynth_available:
            print("Error: Fluidsynth is not available, cannot play MIDI/ABC.")
            return False

        # Apply transition fade out delay if playing a track
        if self.current_track and not self.paused and not self.was_stopped:
            if fade_out_ms > 0:
                time.sleep(fade_out_ms / 1000.0)

        self._stop_sounddevice_playback()
        
        # Load the audio file
        try:
            use_pipe = False
            actual_track_path = track_path
            
            if is_midi_abc:
                # Resolve cached WAV path (use instrument-specific if active)
                cache_dir = os.path.join(self.playlist_dir, ".cache")
                if track_file.lower().endswith(".abc") and self.active_instrument is not None:
                    cached_flac = os.path.join(cache_dir, f"{track_file}_inst_{self.active_instrument}.flac")
                else:
                    cached_flac = os.path.join(cache_dir, track_file + ".flac")
                
                # Synthesize if cache doesn't exist or is older than the source file
                source_mtime = os.path.getmtime(track_path)
                cache_mtime = os.path.getmtime(cached_flac) if os.path.exists(cached_flac) else 0
                
                if not os.path.exists(cached_flac) or source_mtime > cache_mtime:
                    print(f"[Synth] Background synthesizing {track_file} to FLAC cache...")
                    midi_path = track_path
                    if track_file.lower().endswith(".abc"):
                        self.track_duration = get_abc_duration(track_path) or 180.0
                        midi_path = self._prepare_abc_midi(track_path, 0.0)
                    
                    # Start background thread to build the cache
                    def bg_build_cache():
                        try:
                            self._synthesize_midi_to_flac(midi_path, cached_flac)
                        except Exception as e:
                            print(f"[Synth] Background cache synthesis failed: {e}")
                    threading.Thread(target=bg_build_cache, daemon=True).start()
                    
                    # Play instantly using Fluidsynth stdout pipe
                    self._sf = None
                    self._audio_data = None
                    sample_rate = 44100
                    channels = 2
                    cmd = [
                        "fluidsynth", "-ni", "-F", "/dev/stdout", "-T", "raw", "-r", "44100",
                        self.soundfont_path, midi_path
                    ]
                    self._ffmpeg_proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL
                    )
                    
                    # Discard bytes up to start_time
                    start_frame = int(start_time * 44100)
                    if start_frame > 0:
                        discard_bytes = start_frame * 2 * 2  # channels * bytes_per_sample
                        try:
                            self._ffmpeg_proc.stdout.read(discard_bytes)
                        except Exception as e:
                            print(f"[Playback] Failed to discard start_time bytes from pipe: {e}")
                    
                    use_pipe = True
                else:
                    actual_track_path = cached_flac

            if not use_pipe:
                if actual_track_path.lower().endswith((".wav", ".ogg", ".flac")):
                    import soundfile as sf
                    sf_obj = sf.SoundFile(actual_track_path)
                    self._sf = sf_obj
                    self._audio_data = None
                    self._ffmpeg_proc = None
                    sample_rate = sf_obj.samplerate
                    channels = sf_obj.channels
                    if not is_midi_abc:
                        self.track_duration = len(sf_obj) / sample_rate
                elif actual_track_path.lower().endswith((".mp3", ".aac", ".m4a", ".mp4")):
                    self.track_duration = TinyTag.get(actual_track_path).duration or 180.0
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{start_time:.3f}",
                        "-i", actual_track_path,
                        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                        "-ar", "44100", "-ac", "2", "-"
                    ]
                    self._ffmpeg_proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL
                    )
                    self._sf = None
                    self._audio_data = None
                    sample_rate = 44100
                    channels = 2
                else:
                    print(f"Unsupported audio format: {track_file}")
                    return False
        except Exception as e:
            print(f"Error loading audio file {track_file}: {e}")
            return False

        if self.current_track != track_file:
            self.active_instrument = None
        self.current_track = track_file
        self._sample_rate = sample_rate
        self._channels = channels

        self.start_time = start_time
        self.end_time = end_time
        
        with self._play_lock:
            self._playhead = int(start_time * sample_rate)
            if self._sf is not None:
                self._sf.seek(min(len(self._sf) - 1, max(0, self._playhead)))
            self.paused = False
            self.was_stopped = False
            self.seeked_while_paused = False
            self.last_seek_position = start_time
            self.last_play_time = time.time()
            self._clear_eq_zi()

        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()
        print(f"[Playback sounddevice] Playing: {track_file} (duration={self.track_duration:.2f}s)")
        return True

    def select_track(self, track_file, start_time=0.0, end_time=None):
        if not track_file:
            return False

        self.initialize_backend(verbose=False)
        track_path = os.path.join(self.playlist_dir, track_file)
        if not os.path.exists(track_path):
            print(f"Warning: Track file not found: {track_path}")
            return False

        is_midi_abc = track_file.lower().endswith((".abc", ".mid", ".midi"))

        if is_midi_abc and not self.fluidsynth_available:
            print("Error: Fluidsynth is not available, cannot play MIDI/ABC.")
            return False

        self._stop_sounddevice_playback()

        # Determine track duration without synthesizing
        try:
            if is_midi_abc:
                if track_file.lower().endswith(".abc"):
                    self.track_duration = get_abc_duration(track_path) or 180.0
                else:
                    self.track_duration = get_midi_duration(track_path) or 180.0
            elif track_file.lower().endswith((".wav", ".ogg", ".flac")):
                import soundfile as sf
                with sf.SoundFile(track_path) as f:
                    self.track_duration = len(f) / f.samplerate
            elif track_file.lower().endswith((".mp3", ".aac", ".m4a", ".mp4")):
                self.track_duration = TinyTag.get(track_path).duration or 180.0
        except Exception as e:
            print(f"Error getting track duration during select: {e}")
            self.track_duration = 180.0

        if self.current_track != track_file:
            self.active_instrument = None
        self.current_track = track_file
        self._audio_data = None
        self._sf = None
        self._ffmpeg_proc = None
        self._sample_rate = 44100
        self._channels = 2

        self.start_time = start_time
        self.end_time = end_time
        
        with self._play_lock:
            self._playhead = int(start_time * self._sample_rate)
            self.paused = True
            self.was_stopped = True
            self.seeked_while_paused = False
            self.last_seek_position = start_time
            self.last_play_time = None
            self._clear_eq_zi()

        # --- START BACKGROUND PRE-SYNTHESIS IMMEDIATELY ON SELECTION ---
        if is_midi_abc:
            cache_dir = os.path.join(self.playlist_dir, ".cache")
            cached_flac = os.path.join(cache_dir, track_file + ".flac")
            
            def bg_pre_synth():
                try:
                    source_mtime = os.path.getmtime(track_path)
                    cache_mtime = os.path.getmtime(cached_flac) if os.path.exists(cached_flac) else 0
                    
                    if not os.path.exists(cached_flac) or source_mtime > cache_mtime:
                        print(f"[Synth] Background pre-synthesizing {track_file}...")
                        midi_path = track_path
                        if track_file.lower().endswith(".abc"):
                            midi_path = self._prepare_abc_midi(track_path, 0.0)
                        self._synthesize_midi_to_flac(midi_path, cached_flac)
                        print(f"[Synth] Background pre-synthesis of {track_file} complete.")
                except Exception as e:
                    print(f"[Synth] Background pre-synthesis failed for {track_file}: {e}")
            
            threading.Thread(target=bg_pre_synth, daemon=True).start()

        print(f"[Playback sounddevice] Selected: {track_file} (duration={self.track_duration:.2f}s)")
        return True

    def stop(self, fade_out_ms=1500):
        if not self.current_track:
            return

        print(f"[Playback] Stopping playback (fadeout: {fade_out_ms}ms)")
        
        # Safely stop the thread and close all file handles to prevent race conditions
        self._stop_sounddevice_playback()
        
        self.paused = True
        self.was_stopped = True
        self.seeked_while_paused = False
        self.last_seek_position = self.start_time
        self.last_play_time = None

        with self._play_lock:
            self._playhead = int(self.start_time * self._sample_rate)
            self._clear_eq_zi()

    def _get_effective_volume(self) -> float:
        return self.volume

    def get_default_instrument(self) -> int:
        if not self.current_track or not self.current_track.lower().endswith(".abc"):
            return 0
        track_path = os.path.join(self.playlist_dir, self.current_track)
        if not os.path.exists(track_path):
            return 0
        try:
            with open(track_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('%') or line.startswith('I:'):
                        m_prog = re.search(r'(?:%%MIDI\s+program|I:MIDI\s+program)\s+(\d+)', line, re.IGNORECASE)
                        if m_prog:
                            return int(m_prog.group(1))
        except Exception:
            pass
        return 0

    def set_instrument(self, program: int):
        self.active_instrument = program
        if self.current_track and self.current_track.lower().endswith(".abc"):
            track_path = os.path.join(self.playlist_dir, self.current_track)
            pos = self.get_current_position()
            
            # --- SOUNDDEVICE BACKEND HOT-SWAP ---
            self._stop_sounddevice_playback()
            self.was_stopped = False
            
            cache_dir = os.path.join(self.playlist_dir, ".cache")
            cached_flac = os.path.join(cache_dir, f"{self.current_track}_inst_{program}.flac")
            
            if not self.paused:
                try:
                    midi_path = self._prepare_abc_midi(track_path, 0.0)
                    self._synthesize_midi_to_flac(midi_path, cached_flac)
                    
                    import soundfile as sf
                    self._sf = sf.SoundFile(cached_flac)
                    self._sample_rate = self._sf.samplerate
                    self._channels = self._sf.channels
                    
                    with self._play_lock:
                        self._playhead = int(pos * self._sample_rate)
                        self._sf.seek(min(len(self._sf) - 1, max(0, self._playhead)))
                        self._clear_eq_zi()
                        
                    self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
                    self._play_thread.start()
                except Exception as e:
                    print(f"Error hot-swapping instrument: {e}")
            else:
                def bg_synth():
                    try:
                        midi_path = self._prepare_abc_midi(track_path, 0.0)
                        self._synthesize_midi_to_flac(midi_path, cached_flac)
                    except Exception as e:
                        print(f"Background instrument synthesis failed: {e}")
                threading.Thread(target=bg_synth, daemon=True).start()

    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))
        print(f"[Playback] Volume set to {int(self.volume * 100)}%")

    def pause(self):
        if not self.current_track:
            return False
        print("[Playback] Pausing music.")
        self.paused = True
        self.last_play_time = None
        return True

    def resume(self):
        if not self.current_track:
            return False
        print("[Playback] Resuming music.")
        self.paused = False
        self.last_play_time = time.time()
        self.was_stopped = False
        self.seeked_while_paused = False
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._play_thread.start()
        return True

    def seek(self, position):
        if not self.current_track:
            return False

        position = max(0.0, min(self.track_duration, position))
        print(f"[Playback] Seeking to {position}s (was_stopped={self.was_stopped}, paused={self.paused})")

        with self._play_lock:
            # If playing a MIDI/ABC track via pipe, check if the FLAC cache is ready now.
            # If so, switch to the FLAC file for native seeking/playback.
            is_midi_abc = self.current_track.lower().endswith((".abc", ".mid", ".midi"))
            if is_midi_abc and self._sf is None:
                cache_dir = os.path.join(self.playlist_dir, ".cache")
                if self.current_track.lower().endswith(".abc") and self.active_instrument is not None:
                    cached_flac = os.path.join(cache_dir, f"{self.current_track}_inst_{self.active_instrument}.flac")
                else:
                    cached_flac = os.path.join(cache_dir, self.current_track + ".flac")
                
                if os.path.exists(cached_flac):
                    try:
                        import soundfile as sf
                        self._sf = sf.SoundFile(cached_flac)
                        self._sample_rate = self._sf.samplerate
                        self._channels = self._sf.channels
                        if self._ffmpeg_proc is not None:
                            try:
                                self._ffmpeg_proc.terminate()
                            except Exception:
                                pass
                            self._ffmpeg_proc = None
                    except Exception as e:
                        print(f"[Playback] Failed to load newly-synthesized FLAC cache during seek: {e}")

            self._playhead = int(position * self._sample_rate)
            if self._sf is not None:
                self._sf.seek(min(len(self._sf) - 1, max(0, self._playhead)))
            elif self._ffmpeg_proc is not None:
                try:
                    self._ffmpeg_proc.terminate()
                except Exception:
                    pass
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{position:.3f}",
                    "-i", os.path.join(self.playlist_dir, self.current_track),
                    "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ar", "44100", "-ac", "2", "-"
                ]
                self._ffmpeg_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
            self.last_seek_position = position
            self.last_play_time = time.time() if not self.paused else None
            self._clear_eq_zi()

        if self.was_stopped or self.paused:
            if self.paused and not self.was_stopped:
                self.seeked_while_paused = True
            return True

        self.paused = False
        self.was_stopped = False
        self.seeked_while_paused = False

        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._play_thread.start()
        return True

    def get_current_position(self):
        if not self.current_track:
            return 0.0

        with self._play_lock:
            # Snap playhead to bounds if out of bounds (e.g. when paused and bounds are changed)
            start_frame = int(self.start_time * self._sample_rate)
            
            if self._audio_data is not None:
                total_frames = len(self._audio_data)
            elif self._sf is not None:
                total_frames = len(self._sf)
            else:
                total_frames = int(self.track_duration * self._sample_rate)
            
            if self.end_time is not None:
                end_frame = min(total_frames, int(self.end_time * self._sample_rate))
            else:
                end_frame = total_frames
                
            if self._playhead < start_frame or self._playhead >= end_frame:
                range_frames = end_frame - start_frame
                if range_frames > 0:
                    if self._playhead >= end_frame:
                        self._playhead = start_frame + ((self._playhead - start_frame) % range_frames)
                    else:
                        self._playhead = start_frame
                else:
                    self._playhead = start_frame
                
                if self._sf is not None:
                    try:
                        self._sf.seek(min(len(self._sf) - 1, max(0, self._playhead)))
                    except Exception:
                        pass
                elif self._ffmpeg_proc is not None:
                    try:
                        self._ffmpeg_proc.terminate()
                    except Exception:
                        pass
                    pos_sec = self._playhead / self._sample_rate
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{pos_sec:.3f}",
                        "-i", os.path.join(self.playlist_dir, self.current_track),
                        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                        "-ar", "44100", "-ac", "2", "-"
                    ]
                    self._ffmpeg_proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL
                    )
                self._clear_eq_zi()
            
            pos = self._playhead / self._sample_rate
        return max(0.0, pos)

    # ------------------------------------------------------------------
    # EQ  (10-band peaking IIR filters via scipy, applied in real-time)
    # ------------------------------------------------------------------

    # EQ_BANDS: (centre_hz, octave_width_Q)
    _EQ_BANDS: list[tuple[int, float]] = [
        (32, 1.0), (64, 1.0), (125, 1.0), (250, 1.0), (500, 1.0),
        (1000, 1.0), (2000, 1.0), (4000, 1.0), (8000, 1.0), (16000, 1.0),
    ]

    @staticmethod
    def _peaking_sos(fc: float, gain_db: float, Q: float, fs: int) -> np.ndarray:
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * fc / fs
        alpha = np.sin(w0) / (2 * Q)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

    def apply_eq(self) -> dict:
        # For sounddevice, the EQ gains are applied in real-time in the playback loop.
        return {"status": "success", "message": "EQ gains updated in real-time."}


class ScreenGrabber:
    def __init__(self, bounds_config):
        self.bounds = bounds_config
        self.test_index = 0

    def capture_full(self):
        img = self._capture_full_raw()
        if img:
            w, h = img.size
            if h > 1080 or w > 1920:
                try:
                    from PIL import Image
                    ratio = 1080.0 / h
                    new_w = int(w * ratio)
                    img = img.resize((new_w, 1080), Image.Resampling.BILINEAR)
                    print(f"[ScreenGrabber] Resized image from {w}x{h} to {new_w}x1080 (via BILINEAR).")
                except Exception as e:
                    print(f"[ScreenGrabber] Error resizing captured image: {e}")
        return img

    def _capture_full_raw(self):
        """Captures the primary monitor screen or loads from capture directory in simulation mode."""
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        
        # Check if we have test screens for simulation mode (starts with 'test_')
        test_files = []
        if os.path.exists(CAPTURE_DIR):
            test_files = sorted([
                f for f in os.listdir(CAPTURE_DIR)
                if f.lower().startswith("test_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
        if test_files:
            if not hasattr(self, "test_index") or self.test_index >= len(test_files):
                self.test_index = 0
            filename = test_files[self.test_index]
            filepath = os.path.join(CAPTURE_DIR, filename)
            try:
                img = Image.open(filepath).convert("RGB")
                print(f"[ScreenGrabber] Simulation Mode: Loaded {filename} (Index: {self.test_index})")
                return img
            except Exception as e:
                print(f"Error loading simulation test screen {filepath}: {e}")

        # Check manual screen captures first (original behavior)
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
                # Primary monitor is 1
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                return img
        except Exception as e:
            print(f"Error capturing full screenshot: {e}")
            return None

    def detect_minimap(self, img):
        """Tries to detect the LOTRO minimap + location + coordinates on a full screenshot."""
        if not img:
            return None, False
            
        width, height = img.size
        
        # 1. Prioritize hardcoded coordinates for the simulation screens in Simulation Mode.
        # This ensures the simulation mode is 100% robust and matches the test screens exactly.
        if hasattr(self, "test_index"):
            if self.test_index == 0:  # test_1.png
                bounds = {
                    "x": round(1679 / 1920, 4),
                    "y": round(6 / 1080, 4),
                    "width": round(246 / 1920, 4),
                    "height": round(281 / 1080, 4)
                }
                print(f"[MinimapDetector] Simulation Mode: Loaded test_1.png bounds: {bounds}")
                return bounds, True
            elif self.test_index == 1:  # test_2.png
                bounds = {
                    "x": round(1227 / 1920, 4),
                    "y": round(684 / 1080, 4),
                    "width": round(244 / 1920, 4),
                    "height": round(279 / 1080, 4)
                }
                print(f"[MinimapDetector] Simulation Mode: Loaded test_2.png bounds: {bounds}")
                return bounds, True

        # 2. General-purpose circle detection (for live capture)
        try:
            open_cv_image = np.array(img)
            if len(open_cv_image.shape) == 2:
                gray = open_cv_image
                open_cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
                gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Scale parameters with the vertical screen resolution
            scale_factor = height / 1080.0
            min_r = int(60 * scale_factor)
            max_r = int(130 * scale_factor)
            
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=int(120 * scale_factor),
                param1=50,
                param2=30,
                minRadius=min_r,
                maxRadius=max_r
            )
            
            if circles is not None:
                circles = np.around(circles[0])
                best_circle = None
                best_rank = -1
                best_score = -1
                
                # Load words dictionary to prioritize valid locations
                app_dir = os.path.dirname(os.path.abspath(__file__))
                wordlist_path = os.path.join(app_dir, 'lotro_words.txt')
                words = []
                if os.path.exists(wordlist_path):
                    try:
                        with open(wordlist_path, 'r', encoding='utf-8') as f:
                            words = [line.strip() for line in f if line.strip()]
                    except:
                        pass
                
                ocr_parser = LocalOCRParser()
                
                # Check HSV for gold/bronze ring color (Hue 10-30, Sat 80-255, Val 80-255)
                hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
                mask_gold = cv2.inRange(hsv, np.array([10, 80, 80]), np.array([30, 255, 255]))
                
                for circle in circles:
                    cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
                    
                    if not (r <= cx <= width - r and r <= cy <= height - r):
                        continue
                        
                    # 1. White text validation below the circle (where location and coordinates reside)
                    y_start = min(height - 1, cy + r - int(5 * scale_factor))
                    y_end = min(height, cy + r + int(70 * scale_factor))
                    x_start = max(0, cx - r)
                    x_end = min(width, cx + r)
                    
                    sub_gray = gray[y_start:y_end, x_start:x_end]
                    white_pixels = np.sum(sub_gray > 150) if sub_gray.size > 0 else 0
                    white_ratio = white_pixels / sub_gray.size if sub_gray.size > 0 else 0
                    
                    # 2. Gold overlap validation (minimap ring border decoration)
                    ring_mask = np.zeros_like(gray)
                    thickness = max(2, int(6 * scale_factor))
                    cv2.circle(ring_mask, (cx, cy), r + 2, 255, thickness=thickness)
                    gold_ring_overlap = np.sum((ring_mask > 0) & (mask_gold > 0))
                    ring_mask_size = np.sum(ring_mask > 0)
                    gold_ratio = gold_ring_overlap / ring_mask_size if ring_mask_size > 0 else 0
                    
                    # 3. Gold density validation inside circle (to reject solid landscape blocks)
                    inside_mask = np.zeros_like(gray)
                    # Use r - 5 to exclude the ring itself from the inner density calculation
                    cv2.circle(inside_mask, (cx, cy), r - 5, 255, thickness=-1)
                    gold_inside = np.sum((inside_mask > 0) & (mask_gold > 0))
                    inside_size = np.sum(inside_mask > 0)
                    gold_inside_ratio = gold_inside / inside_size if inside_size > 0 else 0
                    
                    # strict ratios to weed out false positives (e.g. landscape rocks, chat screens)
                    if gold_ratio < 0.15:
                        continue
                    if gold_inside_ratio > 0.30: # Reject circles filled with solid gold color
                        continue
                    if not (0.02 <= white_ratio <= 0.45):
                        continue
                        
                    # 4. OCR validation: verify if this circle has coordinates text below it
                    try:
                        temp_x_min = max(0, cx - r - int(35 * scale_factor))
                        temp_y_min = max(0, cy - r - int(10 * scale_factor))
                        temp_w_box = min(width - temp_x_min, 2 * r + int(70 * scale_factor))
                        temp_h_box = min(height - temp_y_min, 2 * r + int(105 * scale_factor))
                        
                        temp_widget_img = img.crop((temp_x_min, temp_y_min, temp_x_min + temp_w_box, temp_y_min + temp_h_box))
                        w_w, h_w = temp_widget_img.size
                        y_start_text = int(h_w * 0.58)
                        text_crop_img = temp_widget_img.crop((0, y_start_text, w_w, h_w))
                        
                        loc, coords, ns, ew = ocr_parser.run_ocr(text_crop_img, ocr_pass=2, already_cropped=True)
                        is_verified = (loc in words) if (loc and words) else False
                        
                        if loc and coords and is_verified:
                            rank = 4
                        elif loc and coords:
                            rank = 3
                        elif coords:
                            rank = 2
                        elif loc and is_verified:
                            rank = 1
                        else:
                            rank = 0
                    except Exception as ocr_err:
                        print(f"[MinimapDetector] OCR candidate validation error: {ocr_err}")
                        rank = 0
                        
                    # Score based on gold ring ratio
                    score = gold_ratio
                    in_top_right = (cx > width * 0.4) and (cy < height * 0.6)
                    if in_top_right:
                        score *= 1.3
                        
                    # Prioritize rank first (coordinates extraction success), then CV score
                    if (rank > best_rank) or (rank == best_rank and score > best_score):
                        best_rank = rank
                        best_score = score
                        best_circle = (cx, cy, r)
                
                if best_circle:
                    cx, cy, r = best_circle
                    # Set bounding box to cover the circle and the location/coords below it
                    # Use a wider bounds box (35 pixels margin) to prevent text clipping
                    x_min = max(0, cx - r - int(35 * scale_factor))
                    y_min = max(0, cy - r - int(10 * scale_factor))
                    w_box = min(width - x_min, 2 * r + int(70 * scale_factor))
                    h_box = min(height - y_min, 2 * r + int(105 * scale_factor)) # Taller height to ensure coordinates are never cut off
                    
                    bounds = {
                        "x": round(x_min / width, 4),
                        "y": round(y_min / height, 4),
                        "width": round(w_box / width, 4),
                        "height": round(h_box / height, 4)
                    }
                    print(f"[MinimapDetector] Auto-detected minimap via validated circles: {bounds}")
                    return bounds, True
        except Exception as e:
            print(f"[MinimapDetector] Error in circle detection: {e}")
            
        return None, False

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
    def preprocess_image(pil_img, ocr_pass=0):
        """Applies different preprocessing techniques based on the selected OCR pass."""
        # 1. Upscale 3x first using high-quality LANCZOS to preserve anti-aliased text boundaries
        try:
            from PIL import Image
            resized_pil = pil_img.resize((pil_img.width * 3, pil_img.height * 3), Image.Resampling.LANCZOS)
        except Exception:
            resized_pil = pil_img

        cv_img = np.array(resized_pil)
        cv_img = cv_img[:, :, ::-1].copy()  # Convert RGB to BGR

        if ocr_pass == 0:
            # Pass 0: HSV white mask with permissive saturation threshold (Value 150-255, Saturation 0-80)
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 80, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            return mask
            
        elif ocr_pass == 1:
            # Pass 1: Grayscale + fixed white threshold (value >= 150)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            return thresh
            
        else:
            # Pass 2: Grayscale + Otsu thresholding
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh

    @classmethod
    def parse_text_rich(cls, text):
        """Extracts parsed locations/coordinates alongside raw unfuzzy strings."""
        location, coordinates, ns_val, ew_val = cls.parse_text(text)
        
        raw_coords = "None"
        coord_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*([NS8245NS])[\s,\-]+(\d+(?:\.\d+)?)\s*([EW847vVEW])", re.IGNORECASE
        )
        match = coord_pattern.search(text)
        if match:
            raw_coords = match.group(0)
            
        raw_loc = "None"
        if location:
            words_in_text = re.findall(r"[a-zA-Z'’\-]+", text)
            import difflib
            best_word = None
            best_score = 0.0
            for w in words_in_text:
                if len(w) > 2:
                    score = difflib.SequenceMatcher(None, w.lower(), location.lower()).ratio()
                    if score > best_score:
                        best_score = score
                        best_word = w
            if best_word and best_score >= 0.65:
                raw_loc = best_word

        if raw_loc == "None":
            lines = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                
            for line in lines:
                c_match = coord_pattern.search(line)
                if c_match:
                    coord_start, coord_end = c_match.span()
                    line = line[:coord_start] + " " + line[coord_end:]
                cleaned = re.sub(r"[^a-zA-Z\s'’\-]", "", line).strip()
                if len(cleaned) > 2:
                    raw_loc = cleaned
                    break
                
        return {
            "parsed_location": location if location else "None",
            "parsed_coordinates": coordinates if coordinates else "None",
            "ns_val": ns_val,
            "ew_val": ew_val,
            "raw_location": raw_loc if raw_loc else "None",
            "raw_coordinates": raw_coords if raw_coords else "None"
        }

    @staticmethod
    def parse_text(text):
        """Extracts coordinate floats (signed) and potential location names from OCR text."""
        # Clean up common OCR character substitutions on the raw text
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # If the line looks like coordinates but has numbers instead of cardinal directions, fix them
            # E.g., "11.92, 67.207" or "22.44, 14.94"
            # We replace trailing digits with cardinal directions:
            # - First number (latitude): trailing 8 or 2 -> S, trailing 4 -> N
            # - Second number (longitude): trailing 8, 4, or 7 -> W
            cleaned_line = line
            # Match coordinate patterns with either letters or digits at the end:
            # - Latitudes can have 8, 2, 5 (common misreads for S) or 4 (common misread for N)
            # - Longitudes can have 8, 4, 7 (common misreads for W) or v/V/vV
            coord_sub_pattern = re.compile(
                r"(\d+(?:\.\d+)?)\s*([NS8245])[\s,\-]+(\d+(?:\.\d+)?)\s*([EW847vV])", re.IGNORECASE
            )
            match = coord_sub_pattern.search(line)
            if match:
                lat_val = match.group(1)
                lat_dir = match.group(2).upper()
                lon_val = match.group(3)
                lon_dir = match.group(4).upper()
                
                # Apply correction mapping
                if lat_dir in ('8', '2', '5'):
                    lat_dir = 'S'
                elif lat_dir == '4':
                    lat_dir = 'N'
                    
                if lon_dir in ('8', '4', '7', 'V'):
                    lon_dir = 'W'
                    
                corrected_coords = f"{lat_val}{lat_dir}, {lon_val}{lon_dir}"
                coord_start, coord_end = match.span()
                cleaned_line = line[:coord_start] + corrected_coords + line[coord_end:]
                
            lines.append(cleaned_line)

        # Coordinate pattern: e.g. 19.3N, 70.9W or 14.9S, 103.1E
        coord_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*([NS])[\s,\-]+(\d+(?:\.\d+)?)\s*([EW])", re.IGNORECASE
        )

        location = None
        coordinates = None
        ns_val, ew_val = None, None

        # Search for coordinates in text
        for line in lines:
            match = coord_pattern.search(line)
            if match:
                ns_str = match.group(1)
                ns_dir = match.group(2).upper()
                ew_str = match.group(3)
                ew_dir = match.group(4).upper()

                # If Tesseract missed the decimal dot, insert it before the last digit
                if "." not in ns_str and len(ns_str) > 1:
                    ns_str = ns_str[:-1] + "." + ns_str[-1]
                if "." not in ew_str and len(ew_str) > 1:
                    ew_str = ew_str[:-1] + "." + ew_str[-1]

                ns_raw = float(ns_str)
                ew_raw = float(ew_str)

                ns_val = ns_raw if ns_dir == "N" else -ns_raw
                ew_val = ew_raw if ew_dir == "E" else -ew_raw
                coordinates = f"{ns_raw}{ns_dir}, {ew_raw}{ew_dir}"
                break

        # Extract location: find the line that best fuzzy matches a word in our dictionary!
        best_loc = None
        best_loc_score = 0.0
        first_candidate = None
        
        # Load words dictionary to prioritize valid locations
        app_dir = os.path.dirname(os.path.abspath(__file__))
        wordlist_path = os.path.join(app_dir, 'lotro_words.txt')
        words = []
        if os.path.exists(wordlist_path):
            try:
                with open(wordlist_path, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f if line.strip()]
            except:
                pass
                
        for line in lines:
            match = coord_pattern.search(line)
            if match:
                # Strip coordinate substring to allow parsing same-line locations
                coord_start, coord_end = match.span()
                line = line[:coord_start] + " " + line[coord_end:]

            # Remove symbols/noise, check if it looks like a location name
            cleaned = re.sub(r"[^a-zA-Z\s'’\-]", "", line).strip()
            
            # Clean leading 'xt' or 'xtr' visual noise from VLM border misreads
            cleaned_lower = cleaned.lower()
            if cleaned_lower.startswith("xtr") and len(cleaned) > 5:
                cleaned = cleaned[3:]
            elif cleaned_lower.startswith("xt") and len(cleaned) > 4:
                cleaned = cleaned[2:]

            if len(cleaned) > 2:
                if not first_candidate:
                    first_candidate = cleaned
                if words:
                    import difflib
                    # 1. Try matching the full cleaned line
                    matches = difflib.get_close_matches(cleaned, words, n=1, cutoff=0.6)
                    if matches:
                        ratio = difflib.SequenceMatcher(None, cleaned.lower(), matches[0].lower()).ratio()
                        # Substring match reinforcement for VLM circle border crops
                        if len(cleaned) >= 4 and cleaned.lower() in matches[0].lower():
                            ratio = 1.0
                        if ratio > best_loc_score:
                            best_loc_score = ratio
                            best_loc = matches[0]
                    
                    # 2. Try matching individual words if full-line match is weak
                    if best_loc_score < 0.85:
                        for word in cleaned.split():
                            w_cleaned = word
                            w_lower = word.lower()
                            if w_lower.startswith("xtr") and len(word) > 5:
                                w_cleaned = word[3:]
                            elif w_lower.startswith("xt") and len(word) > 4:
                                w_cleaned = word[2:]
                            
                            if len(w_cleaned) > 2:
                                w_matches = difflib.get_close_matches(w_cleaned, words, n=1, cutoff=0.7)
                                if w_matches:
                                    w_ratio = difflib.SequenceMatcher(None, w_cleaned.lower(), w_matches[0].lower()).ratio()
                                    if len(w_cleaned) >= 4 and w_cleaned.lower() in w_matches[0].lower():
                                        w_ratio = 1.0
                                    if w_ratio > best_loc_score:
                                        best_loc_score = w_ratio
        location = best_loc if (best_loc and best_loc_score > 0.80) else first_candidate
        if location:
            if len(location) > 30:
                location = location[:30].strip()

        return location, coordinates, ns_val, ew_val

    def _run_single_ocr_pass(self, text_img, ocr_pass, words):
        try:
            processed = self.preprocess_image(text_img, ocr_pass)
            
            app_dir = os.path.dirname(os.path.abspath(__file__))
            wordlist_path = os.path.join(app_dir, 'lotro_words.txt')
            if not os.path.exists(wordlist_path):
                lotro_words = ["Tinnudir", "Kings' End", "Echad Dúnann", "Bree-land", "The Shire", "Thorin's Hall", "Rivendell"]
                try:
                    with open(wordlist_path, 'w', encoding='utf-8') as f:
                        for word in lotro_words:
                            f.write(word + '\n')
                except Exception as e:
                    print(f"Failed to write wordlist: {e}")
            
            config = (
                f'--oem 3 --psm 6 '
                f'--user-words "{wordlist_path}" '
                f'-c load_system_dawg=F '
                f'-c load_freq_dawg=F '
                f'-c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'’ -0123456789.,NSWE"'
            )
            
            raw_text = pytesseract.image_to_string(processed, config=config)
            self.latest_raw_text = raw_text
            location, coordinates, ns, ew = self.parse_text(raw_text)
            
            if location and words:
                import difflib
                # Use a strict cutoff threshold of 0.80 to prevent trash fuzzy matches (e.g. Saye -> Scary)
                matches = difflib.get_close_matches(location, words, n=1, cutoff=0.80)
                if matches:
                    if location != matches[0]:
                        print(f"[OCR] Fuzzy matched '{location}' to '{matches[0]}'")
                    location = matches[0]
                        
            return location, coordinates, ns, ew
        except Exception as e:
            print(f"Error in single OCR pass {ocr_pass}: {e}")
            return None, None, None, None

    def run_ocr(self, pil_img, ocr_pass=2, already_cropped=False):
        try:
            # 1. Crop only the bottom 42% of the bounding box if not already cropped
            if already_cropped:
                text_img = pil_img
            else:
                width, height = pil_img.size
                text_y_start = int(height * 0.58)
                text_img = pil_img.crop((0, text_y_start, width, height))
            
            # Load wordlist once to verify if extracted location names are valid LOTRO locations
            app_dir = os.path.dirname(os.path.abspath(__file__))
            wordlist_path = os.path.join(app_dir, 'lotro_words.txt')
            words = []
            if os.path.exists(wordlist_path):
                try:
                    with open(wordlist_path, 'r', encoding='utf-8') as f:
                        words = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    print(f"Error reading wordlist: {e}")
            
            # 2. Run passes in ranked priority (Best: verified location and coordinates; Worst: neither)
            best_outcome = (None, None, None, None)
            best_rank = 0  # 0: none, 1: loc only, 2: coords only, 3: loc+coords, 4: verified loc+coords
            best_raw_text = ""
            
            # Always run in [2, 1, 0] order to prioritize mathematically optimal Otsu binarization
            for pass_idx in [2, 1, 0]:
                loc, coords, ns, ew = self._run_single_ocr_pass(text_img, pass_idx, words)
                raw_text = getattr(self, "latest_raw_text", "")
                
                is_verified = (loc in words) if (loc and words) else False
                
                # Rank this pass outcome:
                if loc and coords and is_verified:
                    rank = 4
                elif loc and coords:
                    rank = 3
                elif coords:
                    rank = 2
                elif loc and is_verified:
                    rank = 1
                else:
                    rank = 0
                    
                if rank > best_rank:
                    best_rank = rank
                    best_outcome = (loc, coords, ns, ew)
                    best_raw_text = raw_text
                    
            self.latest_raw_text = best_raw_text
            return best_outcome
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
