INPUT_DIRECTORY="~/deepmusic/data/midis/clean_midi/"

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD="~/deepmusic/data/notesequences/clean_midi_notesequence.tfrecord"

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive
