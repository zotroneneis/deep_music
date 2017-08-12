INPUT_DIRECTORY="~/deepmusic/input/midis/debug_midis/"

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD="~/deepmusic/input/notesequences/debug_notesequence.tfrecord"

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive
