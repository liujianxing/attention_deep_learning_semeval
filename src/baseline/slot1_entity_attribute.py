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

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

from read.config_reader import CONST
from util.utils import load_pickle as load
from util.utils import save_pickle

import datetime
import argparse


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

        self._input_data = tf.placeholder(tf.int32, [batch_size, config.num_steps])  # 20 x 70 x [100]
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
        with tf.variable_scope("softmax", reuse=None, initializer=initializer):
            #and tf.control_dependencies([self.embedding, self.output, inputs]):
            hidden_size = config.hidden_size * 2 if config.__class__.__name__ == "BiRNN" else config.hidden_size
            softmax_w = tf.get_variable("softmax_w", [hidden_size, config.classes], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [config.classes], dtype=data_type())
            self._logits = tf.matmul(self._output, softmax_w) + softmax_b

            if config.get_summary:
                variable_summaries(softmax_w, "linear_classifier_w")
                variable_summaries(softmax_b, "linear_classifier_b")
            #weighted_cross_entropy_with_logits
            loss = tf.nn.weighted_cross_entropy_with_logits(self._logits,
                                                            tf.to_float(self._targets), 0.8)

            self._loss = tf.reduce_mean(loss)
            if config.get_summary:
                tf.scalar_summary('cross entropy', self._loss)
            self._cost = cost = tf.reduce_sum(self._loss) / batch_size

            self._predictions = tf.nn.softmax(self.logits)

            self._init_op = tf.initialize_all_variables()
            # No need for gradients if not training
            if not is_training:
                return

            #self._lr = tf.Variable(0.0, trainable=False)

            #tvars = tf.trainable_variables()
            #grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
            #                                  config.max_grad_norm)

            #self.optimizer = tf.train.AdamOptimizer(0.003).minimize(loss)
            if config.get_summary:
                self._merged_summary = tf.merge_all_summaries()

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
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self.optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                       global_step=global_step)

            # Not needed minimize calls this
            # self._train_op = self.optimizer.apply_gradients(zip(grads, tvars))

    def getRNNCell(self, config, inputs, is_training):

        if config.__class__.__name__ in ["LSTM", "GRU"]:

            if config.__class__.__name__ == "GRU":
                unit_cell = tf.nn.rnn_cell.GRUCell(config.hidden_size)  # 200

            if config.__class__.__name__ == "LSTM":
                # Slightly better results can be obtained with forget gate biases
                # initialized to 1 but the hyperparameters of the model would need to be
                # different than reported in the paper.
                unit_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size, forget_bias=1.0)  # 200

            # ADD - DROPOUT
            if is_training and config.keep_prob < 1:
                unit_cell = tf.nn.rnn_cell.DropoutWrapper(unit_cell, output_keep_prob=config.keep_prob)

            cell = tf.nn.rnn_cell.MultiRNNCell([unit_cell] * config.num_layers)  # 200 x 2

            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

            self._initial_state = cell.zero_state(config.batch_size, data_type())

            # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
            # This builds an unrolled LSTM for tutorial purposes only.
            # In general, use the rnn() or state_saving_rnn() from rnn.py.
            #
            # The alternative version of the code below is:
            #
            # from tensorflow.models.rnn import rnn
            # inputs = [tf.squeeze(input_, [1])
            #           for input_ in tf.split(1, num_steps, inputs)]
            # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
            outputs = []
            state = self._initial_state
            with tf.variable_scope("RNN"):
                for time_step in range(config.num_steps):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)  # 20 x 20 x 200

            output = tf.reduce_sum(outputs, 0) / config.num_steps  # reduce to 20 x 200
            self._final_state = state

        if config.__class__.__name__ in ["BiRNN"]:
            self._final_state = tf.zeros(config.batch_size, data_type())
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size)

            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layers)  # 200 x 2
            lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * config.num_layers)  # 200 x 2

            if is_training and config.keep_prob > 0:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=config.keep_prob)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=config.keep_prob)
                inputs = tf.nn.dropout(inputs, config.keep_prob, noise_shape=None, seed=None)

            inputs = [tf.squeeze(x) for x in tf.split(0, config.batch_size, inputs)]

            output, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell,
                                                   lstm_bw_cell,
                                                   inputs,
                                                   dtype=tf.float32)

            output = tf.reduce_mean(output, 1)

            # Just to make Model Run
            self._initial_state = tf.zeros(config.batch_size, data_type())

        return output, inputs

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

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


class GRU(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 128
    max_epoch = 25
    max_max_epoch = 100
    keep_prob = 0.3
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    max_sequence = 70
    classes = 13
    input_size = 100
    embedding_size = 100
    get_summary = False


class LSTM(object):
    """Medium config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 128
    max_epoch = 25
    max_max_epoch = 100
    keep_prob = 0.8
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    classes = 13
    input_size = 100
    embedding_size = 100
    max_sequence = 70
    get_summary = False


class BiRNN(object):
    """Large config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 128
    max_epoch = 25
    max_max_epoch = 100
    keep_prob = 0.80
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    classes = 13
    input_size = 100
    embedding_size = 100
    max_sequence = 70
    get_summary = False


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    classes = 13
    input_size = 100
    embedding_size = 100


def run_epoch(session, m, x_data, y_data, writer = None, run_options = None, run_metadata = None, verbose=False):
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

    state = m.initial_state.eval()

    losses = []

    for step, (x, y) in enumerate(semeval_itterator(x_data,
                                                    y_data,
                                                    m.batch_size,
                                                    m.num_steps)):
        # if delta_cost < epsilon:
        # print("delta: ", delta_cost, " epsilon: ", epsilon)
        # break

        if writer:
            cost, loss, state, summary, _ = session.run([m.cost, m.loss, m.final_state, m.merged_summary, m.optimizer],
                                                        {m.input_data: x,
                                                         m.targets: y,
                                                         m.initial_state: state},
                                                     options=run_options,
                                                     run_metadata=run_metadata)
        else:
            cost, loss, state, _ = session.run([m.cost, m.loss, m.final_state, m.optimizer],
                                                     {m.input_data: x,
                                                      m.targets: y,
                                                      m.initial_state: state})
        #writer.add_run_metadata(run_metadata, 'step%03d' % step)
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
    if model == "GRU":
        return GRU()
    elif model == "LSTM":
        return LSTM()
    elif model == "BiRNN":
        return BiRNN()
    elif model == "TEST":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(CONST, data):
    config = get_config(CONST.MODEL)

    #if not CONST.TRAIN:
    #   config.num_steps = 1

    config.vocab_size = len(data["embeddings"])

    tf.reset_default_graph()
    # start graph and session
    # config=tf.ConfigProto(log_device_placement=True) pass to tf.Session
    # to see which devices all operations can run on
    with tf.Graph().as_default(), tf.Session() as session:
        # Returns an initializer that generates tensors with a uniform distribution.
        #initializer = tf.random_uniform_initializer(-config.init_scale,
        #                                            config.init_scale)

        # Use initializer, mode with variable scope and both called mode
        # Returns a context for variable scope.
        with tf.variable_scope("model", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
            training_model = RNNModel(is_training=True, config=config)  # model class

        if CONST.TRAIN:

            tf.initialize_all_variables().run()
            session.run(training_model.embedding.assign(data["embeddings"]))  # train

            # if config.__class__.__name__ is not "BiRNN":
            #    training_model.initial_state.eval()
            all_losses = []
            #training_model.assign_lr(session, 0.945)  # RUN GRAPH? but no data?

            train_writer = run_metadata = run_options = False

            if config.get_summary:
                # session = tf.InteractiveSession()
                train_writer = tf.train.SummaryWriter(CONST.SLOT1_MODEL_PATH + "graph/" + config.__class__.__name__ ,
                                                      session.graph)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            for i in range(config.max_max_epoch):  # setup epoch
                # lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)  # learning rate decay?
                # training_model.assign_lr(session, config.learning_rate * lr_decay)  # RUN GRAPH? but no data?

                #print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(training_model.lr)))

                train_perplexity, loss, losses = run_epoch(session,
                                                           training_model,
                                                           data["x_train"],
                                                           data["y_train"],
                                                           train_writer,
                                                           run_options,
                                                           run_metadata,
                                                           verbose=True)
                all_losses = all_losses + [np.mean(losses)]

                print("Epoch: %d Avg. Total Mean Loss: %.6f" % (i + 1, np.mean(all_losses)))


            from util.evaluations import print_config
            print_config(config)

            import matplotlib.pyplot as plt
            # plt.plot([np.mean(all_losses[i-50:i]) for i in range(len(all_losses))])
            x = [i for i in range(len(all_losses))]
            plt.plot(np.array(x), np.array(all_losses))
            plt.savefig("losses" + config.__class__.__name__ + ".png")
            save_pickle(CONST.DATA_DIR + config.__class__.__name__, all_losses)
            print("saved losses.png and data")

            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess=session, save_path=CONST.SLOT1_MODEL_PATH + config.__class__.__name__ + "/slot1")
            print("model saved: " + path)


            if config.get_summary:
                train_writer.close()

            session.close()  # doesn't seem to close under scope??

        if not CONST.TRAIN:
            with tf.variable_scope("model", reuse=True):  # reuse scope to evaluate model :-)

                config.batch_size = len(data["x_test"])

                # Initialize Model Graph
                validation_model = RNNModel(is_training=False, config=config)

                #tf.initialize_all_variables().run()
                # set embeddings again
                session.run(validation_model.embedding.assign(data["embeddings"]))

                # Load Data Back Trained Model
                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.SLOT1_MODEL_PATH + config.__class__.__name__ + "/slot1")

                x_test = [n[:config.num_steps] for n in data["x_test"]]

                # Get Predictions
                predictions = session.run(validation_model.predictions,
                                          {validation_model.input_data: x_test,
                                           validation_model.targets: data["y_test"]})

                from util.evaluations import print_config

                print_config(config)

                from util.evaluations import evaluate

                save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "predictions", {"predictions": predictions, "y": data["y_test"]})

                evaluate(predictions, data["y_test"], CONST.THRESHOLD)

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
