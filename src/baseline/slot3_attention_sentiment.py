"""
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from read.data_processing import semeval_itterator

from read.config_reader import CONST
from util.utils import load_pickle as load
from util.utils import save_pickle

import datetime
import argparse

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def variable_summaries(variable, name):
    """Attach a lot of summaries to a Tensor"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(variable)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(variable - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(variable))
        tf.scalar_summary('min/' + name, tf.reduce_min(variable))
        tf.histogram_summary(name, variable)


class RNNModel(object):
    """RNN model."""

    def __init__(self, is_training, config):
        """

        :rtype: model to train or evaluate
        """
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])  # 20 x 70 x [100]
        self._targets = tf.placeholder(tf.int32, [batch_size, config.classes])  # 20 x 13

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable("embedding",
                                             [vocab_size, config.embedding_size],
                                             dtype=data_type(),
                                             trainable=False)
            inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)

        self._output, inputs = self.getRNNCell(config, inputs, is_training)

        if config.get_summary:
            variable_summaries(self._output, config.__class__.__name__)

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        # vectorize category entity & attribute
        # self._input_e = tf.placeholder(tf.int32, [batch_size, 1])
        # self._input_a = tf.placeholder(tf.int32, [batch_size, 1])
        # v_entities = tf.nn.embedding_lookup(self.embedding, self._input_e)
        # v_aspects = tf.nn.embedding_lookup(self.embedding, self._input_a)

        # self.entity_attention = self.apply_attention(inputs, self._output, v_entities, config.max_sequence)
        # self.attribute_attention = self.apply_attention(inputs, self._output, v_aspects, config.max_sequence)

        # r_attention = tf.concat(0, [self.entity_attention, self.aspects_attention])

        # Apply attention
        with tf.variable_scope("attention", initializer=initializer, reuse=None):

            w_et = tf.get_variable("w_et", [batch_size, config.embedding_size], dtype=data_type())
            w_at = tf.get_variable("w_at", [batch_size, config.embedding_size], dtype=data_type())

            self._input_e = tf.placeholder(tf.int32, [batch_size])
            self._input_a = tf.placeholder(tf.int32, [batch_size])

            v_entities = tf.nn.embedding_lookup(self.embedding, self._input_e)
            v_aspects = tf.nn.embedding_lookup(self.embedding, self._input_a)

            e_t = tf.matmul(tf.matmul(tf.squeeze(v_entities), tf.transpose(w_et)), self._output)
            a_t = tf.matmul(tf.matmul(tf.squeeze(v_aspects), tf.transpose(w_at)), self._output)

            self.entity_attention = tf.nn.softmax(e_t, name="entity_attention")
            self.aspects_attention = tf.nn.softmax(a_t, name="aspect_attention")

            r_attention = tf.concat(1, [tf.matmul(self.entity_attention, self._output, transpose_b=True),
                                        tf.matmul(self.aspects_attention, self._output, transpose_b=True)])

        with tf.variable_scope("softmax", reuse=None, initializer=initializer):
            # and tf.control_dependencies([self.embedding, self.output, inputs]):
            hidden_size = config.hidden_size * 2 if config.__class__.__name__ == "BiRNN" else config.hidden_size
            softmax_w = tf.get_variable("softmax_w", [config.batch_size*2, config.classes], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [config.classes], dtype=data_type())
            self._logits = tf.matmul(r_attention, softmax_w) + softmax_b

            if config.get_summary:
                variable_summaries(softmax_w, "linear_classifier_w")
                variable_summaries(softmax_b, "linear_classifier_b")

            # target is valid distribution, is one hot ok on multi-class?
            loss = tf.nn.softmax_cross_entropy_with_logits(self._logits,
                                                           tf.to_float(self._targets))

            self._loss = tf.reduce_mean(loss)
            if config.get_summary:
                tf.scalar_summary('cross entropy', self._loss)
            self._cost = cost = tf.reduce_sum(self._loss) / batch_size

            self._predictions = tf.nn.softmax(self.logits)

            self._init_op = tf.initialize_all_variables()

            if config.get_summary:
                self._merged_summary = tf.merge_all_summaries()

            # No need for gradients if not training
            if not is_training:
                return

            # Optimizer.
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(10.0,
                                                       global_step,
                                                       5000,
                                                       0.1,
                                                       staircase=True)
            self._lr = learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                       global_step=global_step)

    def apply_attention(self, inputs, rep, vec, seq_length):
        """
        :param inputs: list of (seq_len) batch_size x input_size
        :param rep: input_size
        :param input_size:  usually size of the state
        :param hidden_size: scores size
        :param seq_length: length of the sequence
        :return: output: batch_size x input_size
                 weights: alphas: list of (seq_len) batch_size x 1
        """
        with tf.variable_scope("attention"):
            tf.get_variable_scope().reuse_variables()

            # seq_len x batch_size x input_size
            scores = [tf.nn.tanh(tf.matmul(inputs[i], tf.transpose(rep))) for i in range(seq_length)]
            # batch_size x seq_len
            pre_activations = tf.concat(1, scores)
            # batch_size x seq_len
            activation = tf.nn.softmax(pre_activations)
            # seq_len x batch_size x 1
            weights = tf.split(1, seq_length, activation)
            # seq_len x batch_size x input_size
            weighted_inputs = [tf.mul(inputs[i], weights[i]) for i in range(seq_length)]
            # batch_size x input_size
            output = tf.add_n(weighted_inputs)
            alphas = tf.concat(1, weights)
        return output, alphas

    def getRNNCell(self, config, inputs, is_training):

        # Need scope for each direction here: https://github.com/tensorflow/tensorflow/issues/799
        with tf.variable_scope('forward_multi'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size)
            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layers)
            if is_training and config.keep_prob > 0:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=config.keep_prob)

        with tf.variable_scope('backward_multi'):
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size)
            lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * config.num_layers)
            if is_training and config.keep_prob > 0:
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=config.keep_prob)

        if is_training and config.keep_prob > 0:
            inputs = tf.nn.dropout(inputs, config.keep_prob, noise_shape=None, seed=None)

        inputs = [tf.squeeze(x) for x in tf.split(0, config.batch_size, inputs)]

        output, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell,
                                               lstm_bw_cell,
                                               inputs,
                                               dtype=tf.float32)

        output = tf.reduce_mean(output, 2)

        return output, inputs

    @property
    def input_data(self):
        return self._input_data

    @property
    def input_a(self):
        return self._input_a

    @property
    def input_e(self):
        return self._input_e

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def merged_summary(self):
        return self._merged_summary

    @property
    def output(self):
        return self._output

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions

    @property
    def loss(self):
        return self._loss


class BiRNN(object):
    """Large config."""
    init_scale = 0.1
    learning_rate = 0.0017
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 128
    max_epoch = 2
    max_max_epoch = 150
    keep_prob = 0.90
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    classes = 4
    input_size = 100
    embedding_size = 100
    max_sequence = 70
    get_summary = False


def run_epoch(session, m, x_data, y_data, writer=None, run_options=None, run_metadata=None, verbose=False, category=False):
    """Runs the model on the given data.
    :param session:
    :param m:
    :param y_data:
    :param x_data:
    :param eval_op: REMOVED!!!
    :param verbose:
    """
    epoch_size = ((len(x_data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    epsilon = 1e-8
    delta_cost = 0.5
    prev_cost = 0.0

    losses = []

    for step, (x, y, e, a) in enumerate(semeval_itterator(x_data,
                                                    y_data,
                                                    m.batch_size,
                                                    m.num_steps, category=category)):
        # if delta_cost < epsilon:
        # print("delta: ", delta_cost, " epsilon: ", epsilon)
        # break

        if writer:
            cost, loss, state, summary, _ = session.run([m.cost, m.loss, m.merged_summary, m.optimizer],
                                                        {m.input_data: x,
                                                         m.targets: y,
                                                         m.input_e: e,
                                                         m.input_a: a},
                                                        options=run_options,
                                                        run_metadata=run_metadata)
        else:
            cost, loss, _ = session.run([m.cost, m.loss, m.optimizer],
                                               {m.input_data: x,
                                                m.targets: y,
                                                m.input_e: e,
                                                m.input_a: a})

        # writer.add_run_metadata(run_metadata, 'step%03d' % step)
        if writer:
            writer.add_summary(summary, step)

        delta_cost = abs(cost - prev_cost)
        prev_cost = cost
        costs += cost
        iters += m.num_steps
        losses.append(loss)

        # print("iterations: %d cost %.4f loss %.6f" % (iters, cost, loss))
        # print("updating?", w)

        if verbose and iters % (m.batch_size * 5) == 0:
            print("step %.3f loss : %.6f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, loss, iters * m.batch_size / (time.time() - start_time)))

    return np.exp(abs(costs) / iters), loss, losses


def get_config(model):
    if model == "BiRNN":
        return BiRNN()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(CONST, data):
    config = get_config("BiRNN")

    #if not CONST.TRAIN:
    #    config.num_steps = 1

    config.vocab_size = len(data["embeddings"])
    config.max_max_epoch = CONST.MAX_EPOCH

    tf.reset_default_graph()
    # start graph and session
    # config=tf.ConfigProto(log_device_placement=True) pass to tf.Session
    # to see which devices all operations can run on
    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
            training_model = RNNModel(is_training=True, config=config)  # model class

        if CONST.TRAIN:

            tf.initialize_all_variables().run()
            session.run(training_model.embedding.assign(data["embeddings"]))  # train

            all_losses = []

            train_writer = run_metadata = run_options = False

            if config.get_summary:
                # session = tf.InteractiveSession()
                train_writer = tf.train.SummaryWriter(
                    CONST.SLOT1_MODEL_PATH + "attention_graph/" + config.__class__.__name__,
                    session.graph)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            for i in range(config.max_max_epoch):  # setup epoch

                train_perplexity, loss, losses = run_epoch(session,
                                                           training_model,
                                                           data["x_train"],
                                                           data["p_train"],
                                                           train_writer,
                                                           run_options,
                                                           run_metadata,
                                                           category=data["a_train"],
                                                           verbose=True)
                all_losses = all_losses + [np.mean(losses)]

                print("Epoch: %d Avg. Total Mean Loss: %.6f" % (i + 1, np.mean(all_losses)))

            from util.evaluations import print_config
            print_config(config)

            import matplotlib.pyplot as plt
            # plt.plot([np.mean(all_losses[i-50:i]) for i in range(len(all_losses))])
            x = [i for i in range(len(all_losses))]
            plt.plot(np.array(x), np.array(all_losses))
            plt.savefig("losses_slot3" + config.__class__.__name__ + ".png")
            save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "_slot3", all_losses)
            print("saved slot3 losses.png and losses data")

            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess=session, save_path=CONST.SLOT3_MODEL_PATH + config.__class__.__name__ + "/slot3")
            print("model saved: " + path)

            if config.get_summary:
                train_writer.close()

            session.close()  # doesn't seem to close under scope??

        if not CONST.TRAIN:
            with tf.variable_scope("model", reuse=True):  # reuse scope to evaluate model :-)

                #config.batch_size = len(data["x_test"])

                # Initialize Model Graph
                validation_model = RNNModel(is_training=False, config=config)

                # tf.initialize_all_variables().run()
                # set embeddings again
                session.run(validation_model.embedding.assign(data["embeddings"]))

                # Load Data Back Trained Model
                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.SLOT3_MODEL_PATH + config.__class__.__name__ + "/slot3")

                predictions = []

                # Get Predictions
                for step, (x, y, e, a) in enumerate(semeval_itterator(data["x_test"],
                                                                      data["p_test"],
                                                                      validation_model.batch_size,
                                                                      validation_model.num_steps,
                                                                      category=data["a_train"])):

                    pred = session.run(validation_model.predictions,
                                       {validation_model.input_data: x,
                                        validation_model.targets: y,
                                        validation_model.input_e: e,
                                        validation_model.input_a: a})

                    predictions = predictions + pred.tolist()


                even_batch = len(data["x_test"]) % config.batch_size
                remove_added_batch = config.batch_size - even_batch

                del predictions[-remove_added_batch:]

                predictions = np.asarray(predictions)

                from util.evaluations import print_config

                print_config(config)

                from util.evaluations import evaluate_multiclass

                y = [np.asarray(e) for e in data["p_test"]]

                save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "slot3_predictions",
                            {"predictions": predictions, "y": y})

                evaluate_multiclass(predictions, y, True)

                print("predictions saved")
                # session.close()  # doesn't seem to close under scope??


if __name__ == "__main__":
    # Set and Overload Arguments
    CONST.parse_argument(argparse.ArgumentParser())

    # Set Time of Experiment
    now = datetime.datetime.now()
    time_stamp = "_".join([str(a) for a in [now.month, now.day, now.hour, now.minute, now.second]])

    data = load(CONST.DATA_DIR + CONST.DATA_FILE)

    """
        x_train = data["x_train"]
        x_dev = data["x_dev"]
        x_test = data["x_test"]
        y_train = data["y_train"]
        y_dev = data["y_dev"]
        y_test = data["y_test"]
        l_train = data["l_train"]
        l_dev = data["l_dev"]
        l_test = data["l_test"]
        train_sentences = data["train_sentences"]
        dev_sentences = data["dev_sentences"]
        test_sentences = data["test_sentences"]
        embeddings = data["embeddings"]
        aspects = data["aspects"]
    """

    # tf.app.run(CONST, data)
    main(CONST, data)
