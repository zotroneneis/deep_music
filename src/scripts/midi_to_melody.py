"""
This program converts midi files to Melody objects
"""

import os

import numpy as np
import tensorflow as tf

import magenta
from magenta.music import constants
from magenta.music import events_lib
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2



import click

HOME = os.path.expanduser('~')


# @click.command()
# @click.option('--number', default=10, help='Number of generated midi files')
# @click.option('--qpm', default=150, help='Speed of melodies (quarters per minute)')
# @click.option('--midi-file', default='deepmusic/input/midis/debug_midis', help='Path to save midi files')

midi_file = os.path.join(HOME, 'deepmusic/input/midis/debug_midis/midi-0.mid')

def main():
    melody = midi_file_to_melody(midi_file)


def midi_file_to_melody(midi_file, steps_per_quarter=4, qpm=None,
                        ignore_polyphonic_notes=True):
  """Loads a melody from a MIDI file.

  Args:
    midi_file: Absolute path to MIDI file.
    steps_per_quarter: Quantization of Melody. For example, 4 = 16th notes.
    qpm: Tempo in quarters per a minute. If not set, tries to use the first
        tempo of the midi track and defaults to
        magenta.music.DEFAULT_QUARTERS_PER_MINUTE if fails.
    ignore_polyphonic_notes: Only use the highest simultaneous note if True.

  Returns:
    A Melody object extracted from the MIDI file.
  """
  sequence = midi_io.midi_file_to_sequence_proto(midi_file)
  if qpm is None:
    if sequence.tempos:
      qpm = sequence.tempos[0].qpm
    else:
      qpm = constants.DEFAULT_QUARTERS_PER_MINUTE
  quantized_sequence = sequences_lib.quantize_note_sequence(
      sequence, steps_per_quarter=steps_per_quarter)
  melody = Melody()
  melody.from_quantized_sequence(
      quantized_sequence, ignore_polyphonic_notes=ignore_polyphonic_notes)
  import ipdb
  ipdb.set_trace()
  return melody


if __name__ == "__main__":
    main()
