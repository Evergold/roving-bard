import mido

INSTRUMENT_RANGES = {
    "theorbo": (36, 72),
    "lute": (48, 84),
    "harp": (48, 84),
    "horn": (48, 84),
    "clarinet": (48, 84),
    "fiddle": (48, 84),
    "flute": (60, 96),
    "pibgorn": (60, 96),
    "cowbell": (48, 84),
    "moor cowbell": (48, 84),
    "bagpipe": (48, 84),
    "drum": (48, 84),
}
DEFAULT_RANGE = (48, 84)

def midi_to_abc(midi_file_path: str, instrument: str = "lute") -> str:
    """
    Converts a MIDI file to an ABC notation string using the mido library.
    This is a basic conversion that extracts note on/off events.
    
    IMPORTANT LOTRO CONSTRAINTS:
    - LOTRO does NOT support ABC shorthand for triplets (e.g., `(3ABC`).
    - LOTRO does NOT support ABC shorthand for repeats (e.g., `|: :|`).
    - All notes must be rendered linearly in longhand duration format.
    - Pitches must be constrained to the 3-octave limit of the target instrument.
    """
    mid = mido.MidiFile(midi_file_path)
    
    abc_string = "X:1\n"
    abc_string += "T:Extracted MIDI\n"
    abc_string += "M:4/4\n"
    abc_string += "L:1/8\n"
    abc_string += "K:C\n"
    
    instrument_lower = instrument.lower() if instrument else "lute"
    min_note, max_note = INSTRUMENT_RANGES.get(instrument_lower, DEFAULT_RANGE)
    
    # Very basic mapping from MIDI note numbers to ABC notation
    notes = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']
    
    def midi_note_to_abc(note_number):
        # Transpose note into the valid 3-octave range
        while note_number < min_note:
            note_number += 12
        while note_number > max_note:
            note_number -= 12
            
        octave = (note_number // 12) - 1
        note_name = notes[note_number % 12]
        
        if octave == 4:
            return note_name
        elif octave == 5:
            return note_name.lower()
        elif octave > 5:
            return note_name.lower() + "'" * (octave - 5)
        elif octave < 4:
            return note_name + "," * (4 - octave)
        return note_name
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                abc_note = midi_note_to_abc(msg.note)
                abc_string += abc_note + " "
                
    abc_string += "|]\n"
    return abc_string

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inst = sys.argv[2] if len(sys.argv) > 2 else "lute"
        print(midi_to_abc(sys.argv[1], inst))
