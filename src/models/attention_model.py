import copy
import os

import ipdb
import pprint
import magenta
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from helper.misc import (clip_batch, get_padded_batch, largest_indices,
                         make_batch)
from basic_model import  BasicModel

HOME = os.path.expanduser('~')


class AttentionModel(BasicModel):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        tf.logging.debug('Config file used: {}'.format(self.config))

        self.random_seed = self.config['general']['random_seed']

        self.input_file = os.path.join(HOME,
                                       self.config['general']['input_file'])
        self.test_file = os.path.join(HOME,
                                      self.config['general']['test_file'])
        self.result_dir = os.path.join(HOME,
                                       self.config['general']['result_dir'])
        self.midi_dir = os.path.join(HOME, self.config['general']['midi_dir'])
        self.tensorboard_dir = os.path.join(HOME,
                                            self.config['general']['tb_dir'])

        # Preprocessing
        self.min_note = self.config['preprocess']['min_note']
        self.max_note = self.config['preprocess']['max_note']
        self.transpose_to_key = self.config['preprocess']['transpose_to_key']
        self.input_size = 38
        self.num_classes = 38

        # Hyperparameters
        self.max_iter = self.config['hparams']['max_iter']
        self.n_epochs = self.config['hparams']['n_epochs']
        self.lr = self.config['hparams']['lr']
        self.dropout_keep_prob = self.config['hparams']['dropout_keep_prob']
        self.clip_norm = self.config['hparams']['clip_norm']
        self.batch_size = self.config['hparams']['batch_size']
        self.n_layers = self.config['hparams']['n_layers']
        self.n_hidden = self.config['hparams']['n_hidden']
        self.qpm = self.config['hparams']['qpm']

        self.graph = self.build_graph(tf.Graph())

        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=2, )
            self.init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())
            self.local_init_op = tf.local_variables_initializer()

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        sess_config = tf.ConfigProto()
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        # Nicely print the config
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        self.init()

    def _create_summaries(self):
        """
        Adds summaries for visualization in TensorBoard
        """
        with tf.name_scope("summaries"):
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('perplexity', self.perplexity)
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('logits_flat', self.logits_flat)
            tf.summary.histogram('outputs_flat', self.outputs_flat)
            tf.summary.histogram('Predictions', self.predictions_flat)
            self.merged = tf.summary.merge_all()

    def build_graph(self, graph):
        """
        Sets up the computational graph
        """
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            ##############################################
            #### GET INPUTS AND TEST INPUTS
            ##############################################
            with tf.name_scope('input_pipe'):
                sequence_example_file_paths = tf.gfile.Glob(
                    os.path.expanduser(self.input_file))
                self.inputs, self.labels, self.lengths = get_padded_batch(
                    sequence_example_file_paths, self.batch_size,
                    self.input_size, num_epochs=self.n_epochs)

                with tf.name_scope('melodies'):
                    self.melodies = tf.expand_dims(self.inputs, -1)

            with tf.name_scope("test_input_pipeline"):
                test_sequence_example_file_paths = tf.gfile.Glob(
                    os.path.expanduser(self.test_file))
                self.test_inputs, self.test_labels, self.test_lengths = get_padded_batch(
                    test_sequence_example_file_paths, self.batch_size,
                    self.input_size, num_epochs=1)

            ##############################################
            #### SETTING UP RNN STRUCTURE
            ##############################################
            with tf.name_scope('rnn'):
                cells = []

                for i in range(self.n_layers):
                    # We use several layers of LSTM cells with peephole connections,
                    # dropout and an attention wrapper.
                    # The weights are initialized usin the xavier initialization.
                    # For each initialized LSTM cell we need to specify how many hidden
                    # units the cell should have.
                    cell = tf.contrib.rnn.LSTMCell(
                        self.n_hidden, initializer=tf.contrib.layers.xavier_initializer(), use_peepholes=True)
                    # Add attention
                    cell = tf.contrib.rnn.AttentionCellWrapper(
                        cell, attn_length=20, state_is_tuple=True)
                    # Add dropout (only used during training)
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell,
                        output_keep_prob=(1.0 if not self.config['train'] else
                                          self.dropout_keep_prob))
                    cells.append(cell)

                # To create multiple layers we call the MultiRNNCell function that takes
                # a list of RNN cells as an input and wraps them into a single cell
                # cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

                self.cell = tf.contrib.rnn.MultiRNNCell(
                    cells, state_is_tuple=True)

                # Create a zero-filled state tensor as an initial state
                self.init_state = self.cell.zero_state(self.batch_size,
                                                       tf.float32)

                # Create a recurrent neural network, i.e. unroll the network.
                # The dynamic_rnn function returns
                # 1. an output tensor containing the hidden state of the last layer
                # across all time steps. This tensor has shape
                # (batch_size, ? (later n_steps), n_hidden)
                # 2. the last state of every layer in the network as an LSTMStateTuple
                # that contains the final hidden and the final cell state of the
                # respective layer.
                # outputs, self.final_state = tf.nn.dynamic_rnn(
                #     cell, inputs=self.inputs, initial_state=rnn_tuple_state)
                outputs, self.final_state = tf.nn.dynamic_rnn(
                    self.cell,
                    inputs=self.inputs,
                    initial_state=self.init_state)

                logits = tf.contrib.layers.linear(outputs, self.num_classes)

            with tf.name_scope('flatten_variables'):
                # Flattened outputs of shape (batch_size*? (later n_steps), n_hidden)
                self.outputs_flat = tf.reshape(outputs, [-1, cell.output_size])

                # Compute the networks logits using the flattended outputs
                self.logits_flat = tf.contrib.layers.linear(
                    self.outputs_flat, self.num_classes)
                # Flatten the labels
                labels_flat = tf.reshape(self.labels, [-1])

                # Compute real-valued predictions of the network
                # from the one-hot encoded predictions
                # In other words: transform the one-hot encoding into
                # the 0-38 encoding
                self.predictions_flat = tf.argmax(self.logits_flat, axis=1)

                # Compute the softmax
                softmax_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_flat, logits=self.logits_flat)

            with tf.name_scope("loss"):
                # Compute the loss (cross-entropy)
                self.loss = tf.reduce_mean(softmax_ce)

            with tf.name_scope("metrics"):
                # Compute accuracy and perplexity for evaluation
                correct_predictions = tf.to_float(
                    tf.equal(labels_flat, self.predictions_flat))

                self.perplexity = tf.reduce_mean(tf.exp(softmax_ce))
                self.accuracy = tf.reduce_mean(correct_predictions)

            with tf.name_scope('train'):
                # Create a global step variable to keep track of the current
                # training iteration
                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    name="global_step",
                    collections=[
                        tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES
                    ])

                # Get all variables created with trainable=True
                parameters = tf.trainable_variables()
                # Compute the gradient of the loss w.r.t to the parameters
                gradients = tf.gradients(self.loss, parameters)
                # Clip the gradients. How this works: Given a tensor t, and a maximum
                # clip value clip_norm the op normalizes t so that its L2-norm is less
                # than or equal to clip_norm
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.clip_norm)
                # Apply the optimizer
                self.train_step = tf.train.AdamOptimizer(
                    self.lr, epsilon=0.1).apply_gradients(
                        zip(clipped_gradients, parameters),
                        global_step=self.global_step)

                # If not clipping the gradients, minimize the loss directly
                # self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

            self._create_summaries()

        return graph

    def train(self, save_every=20):
        """
        Trains the model on a given set of training data
        """
        with self.graph.as_default():
            # self.sess.run(self.init_op)
            self.coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(
                sess=self.sess, coord=self.coord)

            ##############################################
            #### CODE WHEN REFEEDING FINAL LSTM STATE
            ##############################################
            # _current_state = np.zeros((self.n_layers, 2, self.batch_size,
            #                            self.n_hidden))

            # feed2 = np.zeros((self.batch_size, self.batch_size))
            # feed3 = np.zeros((self.batch_size, 5120))
            # t = tuple((feed2, feed2))
            # _current_state = np.array([[t, feed2, feed3],[t, feed2, feed3]])


            ##############################################
            #### NETWORK TRAINING
            ##############################################
            epoch_id = 0

            # We run the training as long as possible. Alternatively, the training can
            # run for a specified number of steps
            try:
                while not self.coord.should_stop():

                # Run the training step.
                # Decide whether the final state of the net is fed back into the net.
                # If so, include it in the feed_dict (i.e. uncomment the corresponding code)
                    _predictions_flat, _outputs_flat, _train_step, _current_state, _loss, _acc, _perpl, summary = self.sess.run(
                        [
                            self.predictions_flat, self.outputs_flat,
                            self.train_step, self.final_state, self.loss,
                            self.accuracy, self.perplexity, self.merged
                        ])
                        # feed_dict={
                        #     self.init_state[0][0]: _current_state[0][0],
                        #     self.init_state[0][1]: _current_state[0][1],
                        #     self.init_state[0][2]: _current_state[0][2],
                        #     self.init_state[1][0]: _current_state[1][0],
                        #     self.init_state[1][1]: _current_state[1][1],
                        #     self.init_state[1][2]: _current_state[1][2],
                        # })

                    epoch_id +=1

                    if epoch_id % 20 == 0:
                        nan = np.isnan(np.sum(_outputs_flat))
                        tf.logging.info("Do outputs contain nan? {}".format(nan))
                        self.sw.add_summary(summary, epoch_id)
                        tf.logging.info(
                            "Current training step: {}".format(epoch_id))
                        tf.logging.info('Unique predicted notes: {}'.format(
                            np.unique(_predictions_flat)))
                        tf.logging.info("Current loss: {}".format(_loss))
                        tf.logging.info("Current accuracy: {}".format(_acc))
                        tf.logging.info("Current perplexity: {}".format(_perpl))

                    # Save the model to a checkpoint file
                    if save_every > 0 and epoch_id % save_every == 0:
                        self.save(epoch_id)
                        # We also want to save the final state of the LSTM for prediction
                        # As a workaround we save the elements of the state as numpy arrays
                        p = os.path.join(HOME, self.result_dir)
                        np.save(os.path.join(p, 'final_state_00'), _current_state[0][0])
                        np.save(os.path.join(p, 'final_state_01'), _current_state[0][1])
                        np.save(os.path.join(p, 'final_state_02'), _current_state[0][2])
                        np.save(os.path.join(p, 'final_state_10'), _current_state[1][0])
                        np.save(os.path.join(p, 'final_state_11'), _current_state[1][1])
                        np.save(os.path.join(p, 'final_state_12'), _current_state[1][2])


            # When training is finised, close the coordinator and summary writer
            except tf.errors.OutOfRangeError:
                tf.logging.info("Done training - epoch limit reached!")
                tf.logging.info("Number of training steps: {}".format(epoch_id))
            finally:
                self.coord.request_stop()

            self.coord.join(threads)
            self.sw.close()

    def test(self):
        """
        Tests how well the trained model performs on a separate test set
        """
        self.coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        iteration_id = 0

        # Load the saved final state from the checkpoints folder
        p = os.path.join(HOME, 'deepmusic/models/checkpoints/')
        feed00 = np.load(os.path.join(p, 'final_state_00.npy'))
        # To be able to feed the final state from training back into the net
        # we have to create a tuple
        feed00 = tuple((feed00[0], feed00[1]))
        feed01 = np.load(os.path.join(p, 'final_state_01.npy'))
        feed02 = np.load(os.path.join(p, 'final_state_02.npy'))

        feed10 = np.load(os.path.join(p, 'final_state_10.npy'))
        feed10 = tuple((feed10[0], feed10[1]))
        feed11 = np.load(os.path.join(p, 'final_state_11.npy'))
        feed12 = np.load(os.path.join(p, 'final_state_12.npy'))

        # Combine all the saved numpy arrays
        _current_state = np.array([[feed00, feed01, feed02], [feed10, feed11, feed12]])
        loss = []
        acc = []

        # Run testing on the entire test set, saving the loss and accuracy
        try:
            while not self.coord.should_stop():
                test_inputs, test_lengths, test_labels = self.sess.run(
                        [self.test_inputs, self.test_lengths, self.test_labels])

                _train_step, _current_state, _loss, _acc, _perpl = self.sess.run(
                        [
                            self.train_step, self.final_state, self.loss,
                            self.accuracy, self.perplexity
                            ],
                        feed_dict={
                            self.inputs: test_inputs,
                            self.labels: test_labels,
                            self.init_state[0][0]: _current_state[0][0],
                            self.init_state[0][1]: _current_state[0][1],
                            self.init_state[0][2]: _current_state[0][2],
                            self.init_state[1][0]: _current_state[1][0],
                            self.init_state[1][1]: _current_state[1][1],
                            self.init_state[1][2]: _current_state[1][2],
                            })

                loss.append(_loss)
                acc.append(_acc)
                tf.logging.info("Current test step: {}".format(iteration_id))
                iteration_id +=1
                tf.logging.info("Current loss: {}".format(_loss))
                tf.logging.info("Current accuracy: {}".format(_acc))
                tf.logging.info("Current perplexity: {}".format(_perpl))
                tf.logging.debug("=========================================================================================")

        except tf.errors.OutOfRangeError:
            tf.logging.info("Done testing")
            tf.logging.info("Number of testing steps: {}".format(iteration_id))
            avg_loss = np.mean(loss)
            avg_acc = np.mean(acc)
            tf.logging.info("Average loss: {}".format(avg_loss))
            tf.logging.info("Average accuracy: {}".format(avg_acc))
        finally:
            self.coord.request_stop()

        self.coord.join(threads)
        self.sw.close()


    def get_input_melodies(self):
        """
        Saves a sample three training melodies as MIDI files
        """
        with self.graph.as_default():
            self.coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(
                sess=self.sess, coord=self.coord)

            _inputs, _lengths, _labels = self.sess.run(
                [self.inputs, self.lengths, self.labels])

            # Convert inputs from one hot encoding to encoding
            # using real numbers (0-38)
            _inputs = self.sess.run(tf.argmax(_inputs, axis=2))

            # Select one of the generated melodies
            melody1 = _inputs[0]
            melody2 = _inputs[1]
            melody3 = _inputs[2]

            # To use the magenta.music.Meldoy function we have to change
            # the encoding of the predicted notes.
            # In the needed encoding -2 and -1 are special events and
            # 0-127 are pitches.
            melody1 = [x - 2 if x in [0, 1] else x + 48 for x in melody1]
            melody2 = [x - 2 if x in [0, 1] else x + 48 for x in melody2]
            melody3 = [x - 2 if x in [0, 1] else x + 48 for x in melody3]

            # Transform the meldodies into midi files and save them
            m1 = magenta.music.Melody(melody1)
            m2 = magenta.music.Melody(melody2)
            m3 = magenta.music.Melody(melody3)

            m1 = m1.to_sequence(qpm=self.qpm)
            m2 = m2.to_sequence(qpm=self.qpm)
            m3 = m3.to_sequence(qpm=self.qpm)

            midi_path1 = os.path.join(self.midi_dir, 'input_melody_1.mid')
            midi_path2 = os.path.join(self.midi_dir, 'input_melody_2.mid')
            midi_path3 = os.path.join(self.midi_dir, 'input_melody_3.mid')

            tf.logging.info('Saving songs to {}'.format(self.midi_dir))
            magenta.music.sequence_proto_to_midi_file(m1, midi_path1)
            magenta.music.sequence_proto_to_midi_file(m2, midi_path2)
            magenta.music.sequence_proto_to_midi_file(m3, midi_path3)


    def generate(self):
        """
        Generates new melodies using the (trained) model, given a primer note
        or a primer note sequence
        """
        with self.sess as sess:
            self.coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(
                sess=self.sess, coord=self.coord)

            # Get the primer and convert it into one-hot encoding
            primer = self.config['generate']['primer']
            _inputs = np.zeros((len(primer), 38))
            for i, x in enumerate(primer):
                _inputs[i, x] = 1

            tf.logging.info("primer sequence: {}".format(self.sess.run(tf.argmax(_inputs, axis=1))))

            # Create a batch using the primer mutliple times. In this
            # way we can generate several differnent melodies simultaneously
            _test_inputs = make_batch(_inputs)
            original_input = np.copy(_test_inputs)
            final_result = np.copy(_test_inputs)
            _, length, depth = _test_inputs.shape

            # Load the saved final state from the checkpoints folder
            p = os.path.join(HOME, 'deepmusic/models/checkpoints/')
            feed00 = np.load(os.path.join(p, 'final_state_00.npy'))
            # To be able to feed the final state from training back into the net
            # we have to create a tuple
            feed00 = tuple((feed00[0], feed00[1]))
            feed01 = np.load(os.path.join(p, 'final_state_01.npy'))
            feed02 = np.load(os.path.join(p, 'final_state_02.npy'))

            feed10 = np.load(os.path.join(p, 'final_state_10.npy'))
            feed10 = tuple((feed10[0], feed10[1]))
            feed11 = np.load(os.path.join(p, 'final_state_11.npy'))
            feed12 = np.load(os.path.join(p, 'final_state_12.npy'))

            # Combine all the saved numpy arrays
            _current_state = np.array([[feed00, feed01, feed02],[feed10, feed11, feed12]])

            # Let's generate new music!
            # During generation, the state is fed back into the network
            for i in range(90):
                _logits_flat, _current_state = self.sess.run(
                    [self.logits_flat, self.final_state],
                    feed_dict={
                        self.inputs: _test_inputs,
                        self.init_state[0][0]: _current_state[0][0],
                        self.init_state[0][1]: _current_state[0][1],
                        self.init_state[0][2]: _current_state[0][2],
                        self.init_state[1][0]: _current_state[1][0],
                        self.init_state[1][1]: _current_state[1][1],
                        self.init_state[1][2]: _current_state[1][2],
                    })

                # Transform the logits into probabilities
                # log_sm = tf.nn.softmax(_logits_flat).eval()
                m, n = _logits_flat.shape

                # Compute the five most probable notes predicted by the net
                # notes = largest_indices(log_sm, 5)
                notes = largest_indices(_logits_flat, 5)
                tf.logging.info('Most probable notes: {}'.format(notes[0]))

                # Randomly select one of the notes
                # The most probable note has the highest probability
                # of being selected
                idx = []

                for i in range(m):
                    a = np.random.choice(
                        np.arange(0, 5), p=[0.4, 0.3, 0.2, 0.05, 0.05])
                    idx.append(a)

                # Generate random integers between 0 and 4 to
                # later select the 5 most probable notes
                # idx = np.random.randint(0, 5, m)
                # idx = idx.tolist()

                # Select the notes
                selected_notes = notes[np.arange(0, m), idx]

                # We are only interested in the prediction of the last time step
                # so we reshape the array
                selected_notes = selected_notes.reshape(128, -1)

                # Select only the prediction for the last timestep
                ton = selected_notes[:, -1]
                tf.logging.info('Predicted notes: {}'.format(ton[0]))

                # Transform the output tensor into a one-hot encoded slice such
                # that it can be attached to the input tensor
                ton_one_hot = tf.one_hot(ton, depth).eval()

                # Add the produced slice to the original tensor
                tmp = np.concatenate((_test_inputs,
                                      ton_one_hot[:, np.newaxis]), 1)
                final_result = np.concatenate((final_result,
                                               ton_one_hot[:, np.newaxis]), 1)

                # Select the relevant part of the tensor as the new input
                _test_inputs = tmp[:, -length:, :]

                tf.logging.info('Length of generated melodies: {}'.format(
                    final_result.shape))

            self.coord.request_stop()
            # Wait for threads to finish.
            self.coord.join(threads)

            # Transform the one-hot predictions into real valued numbers
            tmp_seq = tf.argmax(final_result, 2).eval()

            # Select one of the generated melodies
            notes = tmp_seq[0]
            # To use the magenta.music.Meldoy function we have to change
            # the encoding of the predicted notes.
            # In the needed encoding -2 and -1 are special events and
            # 0-127 are pitches.
            notes = [x - 2 if x in [0, 1] else x + 48 for x in notes]

            # Select the original melody
            original_notes = tf.argmax(original_input, 2).eval()
            original_notes = original_notes[0]
            # Again, we have to change the encoding
            original_notes = [
                x - 2 if x in [0, 1] else x + 48 for x in original_notes
            ]

            m = magenta.music.Melody(notes)
            original = magenta.music.Melody(original_notes)

            generated_sequence = m.to_sequence(qpm=self.qpm)
            original_sequence = original.to_sequence(qpm=self.qpm)

            # model = '~/deepmusic/new_deepmusic/models/results/trained_model.ckpt'
            midi_path_original = os.path.join(self.midi_dir,
                                              'generated_midi_original.mid')
            midi_path = os.path.join(self.midi_dir, 'generated_midi.mid')

            tf.logging.info('Saving songs to {}'.format(self.midi_dir))
            magenta.music.sequence_proto_to_midi_file(generated_sequence,
                                                      midi_path)
            magenta.music.sequence_proto_to_midi_file(original_sequence,
                                                      midi_path_original)

