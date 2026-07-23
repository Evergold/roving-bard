import mido

def midi_to_abc(midi_file_path: str) -> str:
    """
    Converts a MIDI file to an ABC notation string using the mido library.
    This is a basic conversion that extracts note on/off events.
    """
    mid = mido.MidiFile(midi_file_path)
    
    abc_string = "X:1\n"
    abc_string += "T:Extracted MIDI\n"
    abc_string += "M:4/4\n"
    abc_string += "L:1/8\n"
    abc_string += "K:C\n"
    
    # Very basic mapping from MIDI note numbers to ABC notation
    notes = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']
    
    def midi_note_to_abc(note_number):
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
        print(midi_to_abc(sys.argv[1]))
