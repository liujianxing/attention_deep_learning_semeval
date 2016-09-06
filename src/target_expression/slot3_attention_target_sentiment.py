"""
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

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

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

        #with tf.device("/gpu:0"):
        self.embedding = tf.get_variable("embedding",
                                         [vocab_size, config.embedding_size],
                                         dtype=data_type(),
                                         trainable=False)
        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        self._output, inputs = self.getRNNCell(config, inputs, initializer, is_training)

        if config.get_summary:
            variable_summaries(self._output, config.__class__.__name__)

        #initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    #config.init_scale)

        #initializer = tf.contrib.layers.xavier_initializer()


        # Apply attention
        with tf.variable_scope("attention", initializer=initializer, reuse=None):
            self._context_v = tf.get_variable("context", [batch_size, batch_size * config.hidden_size * 2] ,dtype=data_type())
            self._w_at = tf.get_variable("w_et", [config.hidden_size * 2, batch_size * config.hidden_size * 2], dtype=data_type())

            # just for alphas
            self._alpha = tf.nn.softmax(tf.matmul(self._context_v, tf.reshape(self._output, [-1, num_steps])))

            if is_training:

                self._input_t = tf.placeholder(tf.int32, [batch_size])

                h_source = [tf.slice(self._output, [i, self._input_t[i], 0], [1, 1, config.hidden_size*2]) for i in range(batch_size)]

                h_source = tf.squeeze(h_source)

                score = tf.matmul(h_source, self._w_at)

                # Permuting batch_size and n_steps
                x = tf.transpose(self._output, [0, 2, 1])
                # Reshape to (n_steps*batch_size, n_input)
                x = tf.reshape(x, [-1, num_steps])
                # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                score = tf.matmul(score, x)
                alpha_t = tf.nn.softmax(score)

                self._context_v = tf.assign(self._context_v, tf.matmul(alpha_t, tf.reshape(self._output, [num_steps, -1])))

        with tf.variable_scope("softmax", reuse=None, initializer=initializer):
            # and tf.control_dependencies([self.embedding, self.output, inputs]):
            #softmax_w = tf.get_variable("softmax_w", [config.hidden_size*2*config.batch_size, config.classes], dtype=data_type())
            #softmax_b = tf.get_variable("softmax_b", [config.classes], dtype=data_type())
            #self._logits = tf.nn.tanh(tf.matmul(self._context_v, softmax_w) + softmax_b)
            cO_w = tf.get_variable("cO_w", [config.batch_size * config.hidden_size*2, config.num_steps*config.hidden_size*2], dtype=data_type())
            cO_w = tf.matmul(self._context_v,cO_w)
            out_con = tf.concat(1,[tf.reshape(self._output,[config.batch_size,-1]),cO_w])
            softmax_w = tf.get_variable("softmax_w", [config.hidden_size*4*config.num_steps, config.classes], dtype=data_type())
            softmax_b = tf.get_variable("_b", [config.classes], dtype=data_type())
            #self._logits = tf.nn.tanh(tf.matmul(self._context_v, softmax_w) + softmax_b)
            self._logits = tf.nn.tanh(tf.matmul(out_con, softmax_w) + softmax_b)

            self._predictions = tf.nn.softmax(self.logits)

            if config.get_summary:
                variable_summaries(softmax_w, "linear_classifier_w")
                variable_summaries(softmax_b, "linear_classifier_b")

            # target is valid distribution, is one hot ok on multi-class?
            #loss = tf.nn.softmax_cross_entropy_with_logits(self._logits,
            #                                               tf.nn.softmax(tf.to_float(self._targets)))
            # No need for gradients if not training
            if not is_training:
                return
            loss = tf.nn.softmax_cross_entropy_with_logits(self._logits,
                                                            tf.to_float(self._targets))

            self._loss = tf.reduce_mean(loss)
            if config.get_summary:
                tf.scalar_summary('cross entropy', self._loss)
            self._cost = cost = tf.reduce_sum(self._loss) / batch_size

            self._init_op = tf.initialize_all_variables()

            #if config.get_summary:
            #    self._merged_summary = tf.merge_all_summaries()


            # Optimizer.
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                        global_step,
                                                        1000,
                                                        0.9,
                                                        staircase=True)
            #self._lr = learning_rate
            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #reg_constant = 0.01  # Choose an appropriate one.
            #reg_loss = self._loss + reg_constant * sum(reg_losses)
            #optimizer = tf.train.AdamOptimizer(config.learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
            #gradients, v = zip(*optimizer.compute_gradients(self._loss))
            #gradients, _ = tf.clip_by_global_norm(gradients, 5)
            #self.optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                       #global_step=global_step)
            #self._optimizer = optimizer.apply_gradients(zip(gradients, v)) #, global_step=global_step)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self._loss)

    def getRNNCell(self, config, inputs, initializer, is_training):

        # Need scope for each direction here: https://github.com/tensorflow/tensorflow/issues/799
        with tf.variable_scope('forward_multi'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size, initializer=initializer, use_peepholes=True)
            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layers)
            if is_training and config.keep_prob > 0:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=config.keep_prob)

        with tf.variable_scope('backward_multi'):
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size, initializer=initializer, use_peepholes=True)
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

        # output = tf.reduce_mean(output, 2)

        return output, inputs

    @property
    def input_data(self):
        return self._input_data

    @property
    def input_t(self):
        return self._input_t

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
    def context(self):
        return self._context_v

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions

    @property
    def loss(self):
        return self._loss

    @property
    def alpha(self):
        return self._alpha

    @property
    def optimizer(self):
        return self._optimizer

class BiRNN(object):
    """Large config."""
    init_scale = 5.1
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 2
    num_steps = 46
    hidden_size = 64
    max_max_epoch = 10
    keep_prob = 0.50
    lr_decay = 1 / 1.15
    batch_size = 12
    vocab_size = 10000
    classes = 3
    input_size = 100
    embedding_size = 100
    max_sequence = 70
    get_summary = False


def run_epoch(session, m, x_data, y_data, writer=None, run_options=None, run_metadata=None, verbose=False, targets=False):
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
    merged = tf.merge_all_summaries()
    for step, (x, y, t) in enumerate(semeval_itterator(x_data,
                                                    y_data,
                                                    m.batch_size,
                                                    m.num_steps, target=targets)):
        # if delta_cost < epsilon:
        # print("delta: ", delta_cost, " epsilon: ", epsilon)
        # break

        if writer:
            cost, loss, summary, _ = session.run([m.cost, m.loss, merged, m.optimizer],
                                                        {m.input_data: x,
                                                         m.targets: y,
                                                         m.input_t: t},
                                                        options=run_options,
                                                        run_metadata=run_metadata)
        else:
            cost, loss, _ = session.run([m.cost, m.loss, m.optimizer],
                                               {m.input_data: x,
                                                m.targets: y,
                                                m.input_t: t})

        # writer.add_run_metadata(run_metadata, 'step%03d' % step)
        if writer:
            writer.add_run_metadata(run_metadata, 'step%d' % step)
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
    from util.evaluations import print_config
    print_config(config)

    tf.reset_default_graph()
    # start graph and session
    # config=tf.ConfigProto(log_device_placement=True) pass to tf.Session
    # to see which devices all operations can run on
    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:
        with tf.variable_scope("model", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
            training_model = RNNModel(is_training=CONST.TRAIN, config=config)  # model class

        if CONST.TRAIN:

            tf.initialize_all_variables().run()
            session.run(training_model.embedding.assign(data["embeddings"]))  # train

            # Reload save epoch training time
            if CONST.RELOAD_TRAIN:
                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.SLOT3_TARGET_MODEL_PATH + "slot3_target")

            all_losses = []

            train_writer = run_metadata = run_options = False

            if config.get_summary:
                # session = tf.InteractiveSession()
                train_writer = tf.train.SummaryWriter(
                    CONST.SLOT3_TARGET_MODEL_PATH + "attention_graph/slot3_target", session.graph)

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
                                                           targets=data["l_train"],
                                                           verbose=True)
                all_losses = all_losses + [np.mean(losses)]

                print("Epoch: %d Avg. Total Mean Loss: %.6f" % (i + 1, np.mean(all_losses)))

            from util.evaluations import print_config
            print_config(config)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # plt.plot([np.mean(all_losses[i-50:i]) for i in range(len(all_losses))])
            figure_name = CONST.OUT_DIR + "loss/" + "losses_slot3" + config.__class__.__name__ + ".png"
            x = [i for i in range(len(all_losses))]
            plt.plot(np.array(x), np.array(all_losses))
            plt.savefig(figure_name)
            save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "_slot3_target", all_losses)
            print("saved slot3 losses.png and losses data", figure_name)
            print("loss: ", figure_name)
            print("loss data: ", CONST.DATA_DIR + config.__class__.__name__ + "_slot3_target" + ".pickle")

            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess=session, save_path=CONST.SLOT3_TARGET_MODEL_PATH + "slot3_target")
            print("model saved: " + path)

            if config.get_summary:
                train_writer.close()

            session.close()  # doesn't seem to close under scope??

        if not CONST.TRAIN:
            with tf.variable_scope("model", reuse=True):  # reuse scope to evaluate model :-)

                #config.batch_size = len(data["x_test"])

                # tf.initialize_all_variables().run()
                # set embeddings again
                session.run(training_model.embedding.assign(data["embeddings"]))

                # Load Data Back Trained Model
                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.SLOT3_TARGET_MODEL_PATH + "slot3_target")

                # Initialize Model Graph
                validation_model = RNNModel(is_training=False, config=config)

                predictions = []
                alphas = []

                # Get Predictions
                for step, (x, y, t) in enumerate(semeval_itterator(data["x_test"],
                                                                      data["p_test"],
                                                                      validation_model.batch_size,
                                                                      validation_model.num_steps,
                                                                        target=data["l_test"],
                                                                   shuffle_examples=False)):

                    # do not pass in test targets
                    if CONST.HEATMAP:

                       alpha = session.run(validation_model.alpha,
                                           {validation_model.input_data: x}) #, training_model.targets: y})

                       alphas = alphas + alpha.tolist()

                    else:

                        pred = session.run(validation_model.predictions,
                                           {validation_model.input_data: x}) #, training_model.targets: y})

                        predictions = predictions + pred.tolist()

                if not CONST.HEATMAP:
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
                elif CONST.HEATMAP:
                    even_batch = len(data["x_test"]) % config.batch_size
                    remove_added_batch = config.batch_size - even_batch

                    del alphas[-remove_added_batch:]
                    distance = 0
                    sentences = data["test_sentences"]
                    print("computing heatmaps and avg distance...")
                    from util.heatmap import avg_distance_and_heatmaps
                    avg_distance_and_heatmaps(alphas, sentences, CONST.SENTENCE_PATH + "hursh/")



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
