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
import mss
import numpy as np
import pytesseract
from PIL import Image
from tinytag import TinyTag




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


# Safe pygame mixer initialization
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

        # Set SDL_SOUNDFONTS to enable correct MIDI instrument synthesis on Linux
        env_soundfont = os.environ.get("SDL_SOUNDFONTS")
        if env_soundfont and os.path.exists(env_soundfont):
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
                    print(f"Set SDL_SOUNDFONTS environment variable to {path}")
                    break

        # Check if fluidsynth is available on the system for WAV synthesis
        self.fluidsynth_available = False
        try:
            import shutil
            if shutil.which("fluidsynth") and self.soundfont_path and os.path.exists(self.soundfont_path):
                self.fluidsynth_available = True
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
                    print(f"Detected PulseAudio device at index {i}. Routing sounddevice output through it.")
                    break
        except Exception:
            pass

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
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.terminate()
                except Exception:
                    pass
                self._ffmpeg_proc = None

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

    def _synthesize_midi_to_wav(self, midi_path, target_wav_path):
        if not self.soundfont_path or not os.path.exists(self.soundfont_path):
            raise ValueError("No soundfont found for MIDI synthesis.")
            
        os.makedirs(os.path.dirname(target_wav_path), exist_ok=True)
        
        try:
            cmd = [
                "fluidsynth",
                "-ni",
                self.soundfont_path,
                midi_path,
                "-F",
                target_wav_path,
                "-r",
                "44100"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            if os.path.exists(target_wav_path):
                try:
                    os.unlink(target_wav_path)
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
                    
                    if is_midi_abc:
                        # Resolve cached WAV path (use instrument-specific if active)
                        cache_dir = os.path.join(self.playlist_dir, ".cache")
                        if self.current_track.lower().endswith(".abc") and self.active_instrument is not None:
                            cached_wav = os.path.join(cache_dir, f"{self.current_track}_inst_{self.active_instrument}.wav")
                        else:
                            cached_wav = os.path.join(cache_dir, self.current_track + ".wav")
                        
                        # Synthesize if cache doesn't exist or is older than the source file
                        source_mtime = os.path.getmtime(track_path)
                        cache_mtime = os.path.getmtime(cached_wav) if os.path.exists(cached_wav) else 0
                        
                        if not os.path.exists(cached_wav) or source_mtime > cache_mtime:
                            print(f"[Synth] Background synthesizing {self.current_track} to WAV cache...")
                            midi_path = track_path
                            if self.current_track.lower().endswith(".abc"):
                                self.track_duration = get_abc_duration(track_path) or 180.0
                                midi_path = self._prepare_abc_midi(track_path, 0.0)
                            self._synthesize_midi_to_wav(midi_path, cached_wav)
                            
                        actual_track_path = cached_wav

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
                latency="high",
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

        track_path = os.path.join(self.playlist_dir, track_file)
        if not os.path.exists(track_path):
            print(f"Error: Track file not found: {track_path}")
            return False

        is_midi_abc = track_file.lower().endswith((".abc", ".mid", ".midi"))

        if is_midi_abc and not self.fluidsynth_available:
            print("Error: Fluidsynth is not available, cannot play MIDI/ABC.")
            return False

        self._stop_sounddevice_playback()
        
        # Load the audio file
        try:
            actual_track_path = track_path
            
            if is_midi_abc:
                # Resolve cached WAV path (use instrument-specific if active)
                cache_dir = os.path.join(self.playlist_dir, ".cache")
                if track_file.lower().endswith(".abc") and self.active_instrument is not None:
                    cached_wav = os.path.join(cache_dir, f"{track_file}_inst_{self.active_instrument}.wav")
                else:
                    cached_wav = os.path.join(cache_dir, track_file + ".wav")
                
                # Synthesize if cache doesn't exist or is older than the source file
                source_mtime = os.path.getmtime(track_path)
                cache_mtime = os.path.getmtime(cached_wav) if os.path.exists(cached_wav) else 0
                
                if not os.path.exists(cached_wav) or source_mtime > cache_mtime:
                    print(f"[Synth] Synthesizing {track_file} to WAV cache...")
                    midi_path = track_path
                    if track_file.lower().endswith(".abc"):
                        self.track_duration = get_abc_duration(track_path) or 180.0
                        midi_path = self._prepare_abc_midi(track_path, 0.0)
                    self._synthesize_midi_to_wav(midi_path, cached_wav)
                    
                actual_track_path = cached_wav

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
                self._audio_data = None
                self._sf = None
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
            cached_wav = os.path.join(cache_dir, track_file + ".wav")
            
            def bg_pre_synth():
                try:
                    source_mtime = os.path.getmtime(track_path)
                    cache_mtime = os.path.getmtime(cached_wav) if os.path.exists(cached_wav) else 0
                    
                    if not os.path.exists(cached_wav) or source_mtime > cache_mtime:
                        print(f"[Synth] Background pre-synthesizing {track_file}...")
                        midi_path = track_path
                        if track_file.lower().endswith(".abc"):
                            midi_path = self._prepare_abc_midi(track_path, 0.0)
                        self._synthesize_midi_to_wav(midi_path, cached_wav)
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
            cached_wav = os.path.join(cache_dir, f"{self.current_track}_inst_{program}.wav")
            
            if not self.paused:
                try:
                    midi_path = self._prepare_abc_midi(track_path, 0.0)
                    self._synthesize_midi_to_wav(midi_path, cached_wav)
                    
                    import soundfile as sf
                    self._sf = sf.SoundFile(cached_wav)
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
                        self._synthesize_midi_to_wav(midi_path, cached_wav)
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
