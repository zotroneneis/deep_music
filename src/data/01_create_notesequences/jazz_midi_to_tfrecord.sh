INPUT_DIRECTORY="~/deepmusic/data/midis/jazz_midis/"

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD="~/deepmusic/data/notesequences/jazz_midis_notesequence.tfrecord"

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive
