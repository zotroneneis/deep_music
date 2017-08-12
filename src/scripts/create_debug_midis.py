"""
This program generates a chosen number of midi files of a chosen length and speed and saves them.
"""

import os

import numpy as np
import tensorflow as tf
import magenta
import click

HOME = os.path.expanduser('~')


@click.command()
@click.option('--number', default=128, help='Number of generated midi files')
@click.option('--length', default=300, help='Length of generated midi files')
@click.option('--qpm', default=120, help='Speed of melodies (quarters per minute)')
@click.option('--out', default='deepmusic/input/midis/debug_midis', help='Path to save midi files')


def main(number, qpm, out, length):
    """ Main entry point of the app """

    print('Generating {0} midi files, with {1} quarters per minute of length {2}'.format(number, qpm, length))
    generate_midis(number, qpm, out, length)

    print('Finished...')
    print('Saved midi files, to {0}'.format(out))


def generate_midis(number, qpm, out, length):
    res = np.empty((number, length))
    for n in range(number):
        notes = np.random.randint(48, 85, length).tolist()
        res[n] = notes
        melody = magenta.music.Melody(notes)
        input_sequence = melody.to_sequence(qpm=qpm)
        midi_path = os.path.join(HOME, out, 'midi-{}.mid'.format(n))
        magenta.music.sequence_proto_to_midi_file(input_sequence, midi_path)

    np.save(os.path.join(HOME, 'deepmusic/midi_matrix'), res.T)


if __name__ == "__main__":
    main()
