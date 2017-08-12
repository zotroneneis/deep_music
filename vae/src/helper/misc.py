import os
import subprocess

import ipdb
import magenta
import numpy as np
import tensorflow as tf

from magenta.models.melody_rnn import (
    melody_rnn_config_flags, melody_rnn_model, melody_rnn_sequence_generator)
from magenta.music import midi_io, musicxml_reader, note_sequence_io
from magenta.pipelines import (dag_pipeline, melody_pipelines, pipeline,
                               pipelines_common)
from magenta.protobuf import generator_pb2, music_pb2

HOME = os.path.expanduser('~')


def clip_batch(tensor, lengths):
    """
    Clips a padded tensor to the length
    of the shortest sequence in the inputs tensor.

    Args:
        tensor: 3D (inputs) or 2D (labels) tensor
        lengths: list containing original lengths of the sequences

    Returns:
        tensor: clipped input tensor
    """
    shortest_sequence = np.min(lengths)
    longest_sequence = np.max(lengths)
    # middle = int((longest_sequence + shortest_sequence) / 2)

    if tensor.ndim == 3:
        clipped_tensor = tensor[:, :shortest_sequence, :]
    elif tensor.ndim ==2:
        clipped_tensor = tensor[:, :shortest_sequence]

    return clipped_tensor


def read_and_decode_tfrecord(input_size=38):
    file_list = tf.gfile.Glob(
        os.path.expanduser(
            '/tmp/sequence_examples/training_melodies.tfrecord'))
    file_queue = tf.train.string_input_producer(file_list, num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs':
        tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
        'labels':
        tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
    }

    sequence = tf.parse_single_sequence_example(
        serialized_example, sequence_features=sequence_features)

    inputs = sequence[1]['inputs']
    labels = sequence[1]['labels']

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        _inputs, _labels = sess.run([inputs, labels])

        coord.request_stop()
        coord.join()

    return _inputs, _labels


def make_batch(inputs, batch_size=128):
    """
    Transforms an input tensor into a batch with the
    correct size for our model.
    """
    batch = np.repeat(inputs[np.newaxis, :, :], batch_size, axis=0)

    return batch


def largest_indices(arr, n):
    """Gets the indices of the n largest entries
    in each row of arr

    # Sort the array
    arr_sorted = arr.argsort()
    # Get the indices of the n largest elements
    min_inds = arr_sorted[:, -n:]
    # Flip array to get decreasing oder
    max_inds = np.fliplr(min_inds)

    # Example
    # a = np.array([[1, 5, 3, 6, 3],
    #               [5, 7, 3, 1, 9],
    #               [8, 4, 9, 5, 9]])

    # for n in range(4):
    #     res = largest_indices(a, n)
    #     print('res: ', res)


    # res:  [[3 1 4 2 0]
    #        [4 1 0 2 3]
    #        [4 2 0 3 1]]

    # res:  [[3]
    #        [4]
    #        [4]]

    # res:  [[3 1]
    #       [4 1]
    #       [4 2]]

    # res:  [[3 1 4]
    #       [4 1 0]
    #       [4 2 0]]
    Args:
        arr (numpy array): clear
        n (numpy array): clear

    Returns: max_inds (numpy array)

    """

    return np.fliplr(arr.argsort()[:, -n:])


def get_padded_batch(file_list,
                     batch_size,
                     input_size,
                     num_enqueuing_threads=3):
    """
    Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.

    Args:
    file_list: A list of paths to TFRecord files containing SequenceExamples.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
        will have a shape [batch_size, num_steps, input_size].
    num_enqueuing_threads: The number of threads to use for enqueuing
        SequenceExamples.

    Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
    labels: A tensor of shape [batch_size, num_steps] of int64s.
    lengths: A tensor of shape [batch_size] of int32s. The lengths of each
        SequenceExample before padding.

    """
    # INFO: file_list = ['create_melody_rnn/basic_rnn_dataset/training_melodies.tfrecord']
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=10000)  # Step 1
    # A reader is capable of reading data directly from the filesystem
    # It opens the file and reads it lines one by one.
    # A reader is a stateful operation: it preserves its state across
    # multiple runs of the graph, keeping track of which file it it
    # currently reading and what its current position is in the file
    reader = tf.TFRecordReader()  # Step 2
    # The read operation will read one record at a time and return
    # a key/value pair. The key is the records unique identifier
    # the value is a string containing the content of the line
    _, serialized_example = reader.read(file_queue)  # Step 2
    sequence_features = {
        'inputs':
        tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
        'labels':
        tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
    }
    # We now have to parse the string to get the sequence
    _, sequence = tf.parse_single_sequence_example(
        serialized_example, sequence_features=sequence_features)  # Step 3
    length = tf.shape(sequence['inputs'])[0]
    # Create FIFOQueue with padding to support batching
    # variable_size tensors
    queue = tf.PaddingFIFOQueue(
        capacity=1000,
        dtypes=[tf.float32, tf.int64, tf.int32],
        shapes=[(None, input_size), (None, ), ()])

    # We can now push the sequence to a queue that will be shared with
    # the training graph such that it can pull mini-batches from it
    enqueue_ops = [
        queue.enqueue([sequence['inputs'], sequence['labels'], length])
    ] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    # Dequeue a batch of examples

    return queue.dequeue_many(batch_size)


# if __name__ == '__main__':
#     a = np.array([[-5, -5.3, -2.4, -1.9, 3], [3, 3, -2, 1, -5]])
#     b = largest_indices(a, 3)
#     print(b)
