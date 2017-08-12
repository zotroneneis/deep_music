import copy
import glob
import os
from datetime import datetime

import ipdb
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# import magenta
# from helper.misc import (clip_batch, get_padded_batch, largest_indices,
#                          make_batch, read_and_decode_tfrecord)
# from helpers import get_batch

HOME = os.path.expanduser('~')



class BasicVariationalAutoencoder:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        tf.logging.debug('Config file used: {}'.format(self.config))

        self.random_seed = self.config['general']['random_seed']
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.input_file = os.path.join(HOME,
                                       self.config['general']['input_file'])
        self.result_dir = os.path.join(HOME,
                                       self.config['general']['result_dir'])
        self.midi_dir = os.path.join(HOME, self.config['general']['midi_dir'])
        self.tensorboard_dir = os.path.join(HOME,
                                            self.config['general']['tb_dir'])
        tf.logging.info('=======')
        tf.logging.info('GENERAL')
        tf.logging.info('=======')
        tf.logging.info('random_seed: {}'.format(self.random_seed))
        tf.logging.info('input_file: {}'.format(self.input_file))
        tf.logging.info('result_dir: {}'.format(self.result_dir))
        tf.logging.info('midi_dir: {}'.format(self.midi_dir))
        tf.logging.info('tensorboard_dir: {}'.format(self.tensorboard_dir))

        # Hyperparameters
        self.activation_fct = tf.nn.elu
        # self.activation_fct = self.config['hparams']['activation_fct']
        self.max_iter = self.config['hparams']['max_iter']
        self.lr = self.config['hparams']['lr']
        self.h1_encoder = self.config['hparams']['h1_encoder']
        self.h2_encoder = self.config['hparams']['h2_encoder']
        self.h1_decoder = self.config['hparams']['h1_decoder']
        self.h2_decoder = self.config['hparams']['h2_decoder']
        self.n_latent = self.config['hparams']['n_latent']
        self.batch_size = self.config['hparams']['batch_size']
        self.qpm = self.config['hparams']['qpm']
        self.n_steps = self.config['hparams']['n_steps']
        self.input_size = 38
        self.num_classes = 38

        tf.logging.info('===============')
        tf.logging.info('HYPERPARAMETERS')
        tf.logging.info('===============')

        tf.logging.info('max_iter: {}'.format(self.max_iter))
        tf.logging.info('lr: {}'.format(self.lr))
        tf.logging.info('batch_size: {}'.format(self.batch_size))
        tf.logging.info('h1_encoder: {}'.format(self.h1_encoder))
        tf.logging.info('h2_encoder: {}'.format(self.h2_encoder))
        tf.logging.info('h1_decoder: {}'.format(self.h1_decoder))
        tf.logging.info('h2_decoder: {}'.format(self.h2_decoder))
        tf.logging.info('n_latent: {}'.format(self.n_latent))
        tf.logging.info('qpm: {}'.format(self.qpm))

        self.graph = self.build_graph(tf.Graph())

        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=10, )
            self.init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())
            self.local_init_op = tf.local_variables_initializer()

        sess_config = tf.ConfigProto()
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.sw = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()



    def _create_summaries(self):
        """
        Adds summaries for visualization in tensorboard
        """
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("latent_loss", self.latent_loss)
            tf.summary.scalar("reconstruction_loss", self.reconstr_loss)
            self.merged = tf.summary.merge_all()


    def _create_placeholders(self):
        """
        Creates placeholder for the input data
        """
        with tf.name_scope("input_data"):
            sequence_example_file_paths = tf.gfile.Glob(
                os.path.expanduser(self.input_file))
            self.inputs, self.labels, self.lengths = get_padded_batch(
                sequence_example_file_paths, self.batch_size,
                self.input_size)

            # input_path = os.path.join(HOME, self.input_file)
            # file_list = glob.glob(input_path)
            # self.inputs = get_batch(file_list, self.batch_size)

    def _create_encoder(self):
        """
        Computes the output of the encoder network for a given input vector

        The encoder network is used to approximate the true posterior p(z|x)
        using the variational distribution q_phi(z|x). Usually, q_phi(z|x) is
        taken to be a Gaussian distribution with a diagonal covariance matrix
        whose mean and variance vectors are parametrized by a neural network
        with input x.
        So the encoder network takes x as an input and produces a vector of
        means and a vector of variances (or more precise a vector of log
        squared variances) of a Gaussian distribution.
        From this we can sample values of z, i.e. z ~ q_phi(z|x)
        """
        with tf.name_scope("encoder"):
            layer1 = tf.layers.dense(
                self.inputs,
                self.h1_encoder,
                activation=self.activation_fct,
                # kernel_initializer=tf.contrib.layers.xavier_initializer())
                kernel_initializer=self.initializer)
            layer2 = tf.layers.dense(
                layer1,
                self.h2_encoder,
                activation=self.activation_fct,
                kernel_initializer=self.initializer)
            means = tf.layers.dense(
                layer2,
                self.n_latent,
                activation=None,
                kernel_initializer=self.initializer)
            log_sigmas_sq = tf.layers.dense(
                layer2,
                self.n_latent,
                activation=None,
                kernel_initializer=self.initializer)

            # Vector of means
            self.encoder_mean = means
            # Vector of log squared variances
            self.encoder_loq_sigma_sq = log_sigmas_sq

    def _create_decoder(self):
        """
        Computes the output of the decoder network (i.e. the reconstruction)

        for some latent vector z. The latent vector is sampled from the encoder
        using the reparametrization trick. The decoder network takes a
        latent variable z as an input and reproduces the input x.
        """
        with tf.name_scope("decoder"):
            # Reparametrization trick
            eps = tf.random_normal(
                tf.shape(self.encoder_loq_sigma_sq), dtype=tf.float32)
            self.z = self.encoder_mean + tf.sqrt(
                tf.exp(self.encoder_loq_sigma_sq)) * eps
            # Layer activations
            layer1 = tf.layers.dense(
                self.z,
                self.h1_decoder,
                activation=self.activation_fct,
                kernel_initializer=self.initializer)
            layer2 = tf.layers.dense(
                layer1,
                self.h2_decoder,
                activation=self.activation_fct,
                kernel_initializer=self.initializer)
            # layer2_flat = tf.reshape(layer2, [-1, self.h2_decoder])
            # Reconstruction
            self.logits = tf.layers.dense(layer2, self.input_size)
            # logits = tf.contrib.layers.linear(layer2_flat, self.input_size)
            self.reconstr = tf.sigmoid(self.logits)
            # reconstr = tf.sigmoid(logits)

    def _create_loss(self):
        """
        Computes the loss of the network. The loss function has two terms
        1) The reconstruction loss: -log(p(x|z)) which rewards a good reconstruction

        2) The latent loss: Kullback-Leibler divergence between q_phi(z|x)
        and the prior p(z) (which is given by a standard normal distribution).
        This loss acts as a regularizer because it penalizes q_phi(z|x) for
        deviating from a standard normal distribution.
        """
        with tf.name_scope("loss"):
            with tf.name_scope("latent_loss"):
                self.latent_loss = -0.5 * tf.reduce_sum(
                    1 + self.encoder_loq_sigma_sq - tf.square(
                        self.encoder_mean) - tf.exp(self.encoder_loq_sigma_sq))

            with tf.name_scope("reconstruction_loss"):
                self.reconstr_loss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.inputs, logits=self.logits))

            self.loss = self.latent_loss + self.reconstr_loss

    def _create_optimizer(self):
        """
        Minimizes the loss using an Adam optimizer
        """
        with tf.name_scope("optimizer"):
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step",
                collections=[
                    tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES
                ])

            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss, global_step=self.global_step)


    def build_graph(self, graph):
        """
        Creates the VAE network by creating placeholders and setting up the
        encoder and decoder network, the loss, the optimizer, summaries for
        tensorboard and initializing all variables.
        """
        with graph.as_default():
            tf.logging.info('=======')
            tf.logging.info('BUILD GRAPH')
            tf.logging.info('=======')
            tf.set_random_seed(self.random_seed)
            tf.logging.info("Creating placeholders")
            self._create_placeholders()
            tf.logging.info("Create encoder")
            self._create_encoder()
            tf.logging.info("Create decoder")
            self._create_decoder()
            tf.logging.info("Create loss")
            self._create_loss()
            tf.logging.info("Create optimizer")
            self._create_optimizer()
            tf.logging.info("Create summaries")
            self._create_summaries()

        return graph

    def train(self, save_every=20):
        with self.graph.as_default():
            self.coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(
                sess=self.sess, coord=self.coord)

            for epoch_id in range(self.max_iter):
                _inputs, _lengths = self.sess.run(
                    [self.inputs, self.lengths])

                # Clip the length of inputs and labels
                # _inputs = clip_batch(_inputs, _lengths)

                import ipdb
                ipdb.set_trace()
                temp = self.sess.run(tf.argmax(_inputs, axis=2))

                for i in range(128):
                    notes = temp[i]
                    e = [x - 2 if x in [0, 1] else x + 48 for x in notes]
                    m = magenta.music.Melody(e)
                    s = m.to_sequence(qpm=self.qpm)
                    midi = 'generated_midi_{}.mid'.format(i)
                    midi_path = os.path.join(self.midi_dir, midi)
                    magenta.music.sequence_proto_to_midi_file(s, midi_path)


                _train_step, loss, latent_l, reconstr_l, reconstr, summary = self.sess.run([
                    self.optimizer, self.loss, self.latent_loss,
                    self.reconstr_loss, self.reconstr, self.merged
                ])
                # feed_dict={self.inputs: _inputs})

                if epoch_id % 20 == 0:
                    self.sw.add_summary(summary, epoch_id)

                    tf.logging.info('=======')
                    tf.logging.info('TRAINING')
                    tf.logging.info('=======')
                    tf.logging.info("Current training step: {}".format(epoch_id))
                    tf.logging.info("Current loss: {}".format(loss))
                    tf.logging.info("Current latent loss: {}".format(latent_l))
                    tf.logging.info(
                        "Current reconstruction loss: {}".format(reconstr_l))

                if save_every > 0 and epoch_id % save_every == 0:
                    self.save(epoch_id)

            self.coord.request_stop()
            self.coord.join(threads)
            self.sw.close()

    def reconstruct(self, input_data):
        """
        Reconstructs a given batch of input data
        """
        with tf.name_scope("reconstruct_data"):
            return self.sess.run(self.reconstr, {self.inputs: input_data})

    def generate(self):
        """
        Generates new data by sampling from the latent dimension and running the decoder
        """
        with self.graph.as_default():
            with tf.name_scope('generate_data'):
                z = tf.random_normal(shape=(self.batch_size, self.n_steps, self.n_latent))
                z = self.sess.run(z)

                new_data = self.sess.run(self.reconstr, {self.z: z})
                new_data = self.sess.run(tf.argmax(new_data, axis=2))
                notes = new_data[10]
                notes2 = new_data[33]

                # To use the magenta.music.Meldoy function we have to change
                # the encoding of the predicted notes.
                # In the needed encoding -2 and -1 are special events and
                # 0-127 are pitches.
                notes = [x - 2 if x in [0, 1] else x + 48 for x in notes]
                notes2 = [x - 2 if x in [0, 1] else x + 48 for x in notes2]

                # Transform the generated melody into a midi file and save it
                m = magenta.music.Melody(notes)
                m2 = magenta.music.Melody(notes2)
                s = m.to_sequence(qpm=self.qpm)
                s2 = m2.to_sequence(qpm=self.qpm)
                midi_path = os.path.join(self.midi_dir, 'generated_midi.mid')
                midi_path2 = os.path.join(self.midi_dir, 'generated_midi2.mid')
                magenta.music.sequence_proto_to_midi_file(s, midi_path)
                magenta.music.sequence_proto_to_midi_file(s2, midi_path2)

                return new_data

    def save(self, epoch_id):
        global_step_t = tf.train.get_global_step(self.graph)
        global_step = self.sess.run(global_step_t)

        # tf.logging.info('=======')
        # tf.logging.info('SAVING MODEL')
        # tf.logging.info('=======')
        tf.logging.info('Saving to {} with global step {}'.format(
            self.result_dir, global_step))
        save_name = 'model-ep_{}-{}'.format(epoch_id, global_step)
        self.saver.save(self.sess, os.path.join(self.result_dir, save_name))

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)

        if checkpoint is None:
            tf.logging.info('=======')
            tf.logging.info('INITIALIZING MODEL VARIABLES')
            tf.logging.info('=======')
            self.sess.run(self.init_op)

        else:
            tf.logging.info('=======')
            tf.logging.info('LOADING MODEL')
            tf.logging.info('=======')
            tf.logging.info(
                'Loading the model from: {}'.format(self.result_dir))

            self.sess.run(self.local_init_op)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)



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

