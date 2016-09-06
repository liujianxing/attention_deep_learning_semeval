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
from util.file_utils import check_saved_file
from util.utils import load_pickle as load
from util.utils import save_pickle

import datetime
import argparse


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


class shared_Embeddings(object):
    """ Shared Embeddings for Multi-task """

    def __init__(self, config):
        self._input_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])  # 20 x 70 x [100]
        self._embeddings = tf.get_variable("embedding",
                                          [config.vocab_size, config.embedding_size],
                                           dtype=data_type(),
                                          trainable=False)

        self._inputs = tf.nn.embedding_lookup(self._embeddings, self._input_data)

    @property
    def source_data(self):
        return self._input_data

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def inputs(self):
        return self._inputs


class shared_BiRNN(object):
    """shared Multi-task RNN model."""

    def __init__(self, is_training, config, inputs, initializer):
        # Need scope for each direction here: https://github.com/tensorflow/tensorflow/issues/799
        with tf.variable_scope('forward_multi'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size, initializer=initializer)
            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * config.num_layers)
            if is_training and config.keep_prob > 0:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=config.keep_prob)

        with tf.variable_scope('backward_multi'):
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.hidden_size, initializer=initializer)
            lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * config.num_layers)
            if is_training and config.keep_prob > 0:
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=config.keep_prob)

        if is_training and config.keep_prob > 0:
            inputs = tf.nn.dropout(inputs, config.keep_prob, noise_shape=None, seed=None)

        inputs = [tf.squeeze(x) for x in tf.split(0, config.batch_size, inputs)]

        self._output, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell,
                                                    lstm_bw_cell,
                                                        inputs,
                                                        dtype=tf.float32)

    @property
    def cells(self):
        return self._cells

    @property
    def output(self):
        return self._output


class slot1_entity_attribute(object):
    def __init__(self, is_training, config, rnn_input, initializer):

        # target for cross entropy
        self._targets = tf.placeholder(tf.int32, [config.batch_size, config.slot1_classes], name="slot1_targets")

        with tf.variable_scope("tanh_linear", initializer=initializer):
            rnn_tensor = tf.reduce_mean(rnn_input, 1)
            softmax_w = tf.get_variable("softmax_w", [config.hidden_size * 2, config.slot1_classes], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [config.slot1_classes], dtype=data_type())
            self._logits = tf.matmul(rnn_tensor, softmax_w) + softmax_b

            if config.get_summary:
                variable_summaries(softmax_w, "linear_classifier_w")
                variable_summaries(softmax_b, "linear_classifier_b")
                variable_summaries(self._logits, "slot1_tanh_logit")
            # weighted_cross_entropy_with_logits
            loss = tf.nn.weighted_cross_entropy_with_logits(self._logits,
                                                            tf.to_float(self._targets), 0.8)

            self._loss = tf.reduce_mean(loss)

            self._cost = cost = tf.reduce_sum(self._loss) / config.batch_size

            self._predictions = tf.nn.softmax(self.logits)

            if config.get_summary:
                tf.scalar_summary('cross entropy slot1', self._loss)
                variable_summaries(self._cost, "slot1_cost")
                variable_summaries(self._predictions, "slot1_predictions")

            self._init_op = tf.initialize_all_variables()

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
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self._optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                        global_step=global_step)

    @property
    def optimizer(self):
        return self._optimizer

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


class slot3_polarity(object):

    def __init__(self, is_training, config, rnn_output, initializer):

        self._targets = tf.placeholder(tf.int32, [config.batch_size, config.slot3_classes], name="slot3_targets")  # 20 x 13

        with tf.variable_scope("attention", initializer=initializer):
            self._context_v = tf.get_variable("context", [config.batch_size, config.batch_size * config.hidden_size * 2] ,dtype=data_type())
            # just for alphas
            if config.get_summary:
                variable_summaries(self._context_v, "context_attention")
            self._alpha = tf.nn.softmax(tf.matmul(self._context_v, tf.reshape(rnn_output, [-1, config.num_steps])))

            if is_training:
                wt = tf.get_variable("w_et", [config.hidden_size * 2, config.batch_size * config.hidden_size * 2], dtype=data_type())

                self._input_t = tf.placeholder(tf.int32, [config.batch_size])

                h_source = [tf.slice(rnn_output, [i, self._input_t[i], 0], [1, 1, config.hidden_size*2]) for i in range(config.batch_size)]

                h_source = tf.squeeze(h_source)

                score = tf.matmul(h_source, wt)

                # Permuting batch_size and n_steps
                x = tf.transpose(rnn_output, [0, 2, 1])
                # Reshape to (n_steps*batch_size, n_input)
                x = tf.reshape(x, [-1, config.num_steps])
                # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                score = tf.matmul(score, x)
                alpha_t = tf.nn.softmax(score)

                self._context_v = tf.assign(self._context_v, tf.matmul(alpha_t, tf.reshape(rnn_output, [config.num_steps, -1])))

            with tf.variable_scope("softmax", initializer=initializer):
                # and tf.control_dependencies([self.embedding, self.output, inputs]):
                #softmax_w = tf.get_variable("softmax_w", [config.hidden_size*2*config.batch_size, config.slot3_classes], dtype=data_type())
                #softmax_b = tf.get_variable("softmax_b", [config.slot3_classes], dtype=data_type())
                #self._logits = tf.matmul(self._context_v, softmax_w) + softmax_b
                cO_w = tf.get_variable("cO_w", [config.batch_size * config.hidden_size*2, config.num_steps*config.hidden_size*2], dtype=data_type())
                cO_w = tf.matmul(self._context_v,cO_w)
                out_con = tf.concat(1,[tf.reshape(rnn_output,[config.batch_size,-1]),cO_w])
                softmax_w = tf.get_variable("softmax_w", [config.hidden_size*4*config.num_steps, config.slot3_classes], dtype=data_type())
                softmax_b = tf.get_variable("_b", [config.slot3_classes], dtype=data_type())
                #self._logits = tf.nn.tanh(tf.matmul(self._context_v, softmax_w) + softmax_b)
                self._logits = tf.nn.tanh(tf.matmul(out_con, softmax_w) + softmax_b)


            if config.get_summary:
                variable_summaries(softmax_w, "slot3_linear_classifier_w")
                variable_summaries(softmax_b, "slot3_linear_classifier_b")

            # target is valid distribution, is one hot ok on multi-class?
            #loss = tf.nn.softmax_cross_entropy_with_logits(self._logits,
            #                                               tf.nn.softmax(tf.to_float(self._targets)))

            loss = tf.nn.softmax_cross_entropy_with_logits(self._logits,
                                                            tf.to_float(self._targets))

            self._loss = tf.reduce_mean(loss)
            if config.get_summary:
                tf.scalar_summary('cross entropy slot3', self._loss)
            self._cost = cost = tf.reduce_sum(self._loss) / config.batch_size

            self._predictions = tf.nn.softmax(self.logits)

            self._init_op = tf.initialize_all_variables()

            # No need for gradients if not training
            if not is_training:
                return

            # Optimizer.
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                       global_step,
                                                       5000,
                                                       0.1,
                                                       staircase=True)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_constant = 0.01  # Choose an appropriate one.
            reg_loss = self._loss + reg_constant * sum(reg_losses)
            optimizer = tf.train.AdadeltaOptimizer(config.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(reg_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self._optimizer = optimizer.apply_gradients(zip(gradients, v))


    @property
    def optimizer(self):
        return self._optimizer

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

    @property
    def input_t(self):
        return self._input_t

    @property
    def alpha(self):
        return self._alpha

class multitask(object):
    """Mult-task config."""
    init_scale = 0.1
    learning_rate = .0017
    max_grad_norm = 10
    num_layers = 2
    num_steps = 12
    hidden_size = 128
    # max_epoch = 25
    max_max_epoch = 1
    keep_prob = 0.80
    lr_decay = 1 / 1.15
    batch_size = 10
    vocab_size = 10000
    slot1_classes = 13
    slot3_classes = 3
    input_size = 100
    embedding_size = 100
    max_sequence = 70
    get_summary = False


def run_epoch(session, m, x_data, y_data, y_polarity, writer=None, run_options=None, run_metadata=None, verbose=False,
              category=False, config=False, target=None):
    epoch_size = ((len(x_data) // config.batch_size) - 1) // config.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0

    #state = m.slot1_model.initial_state.eval()

    slot1_losses = []
    slot3_losses = []
    merged = tf.merge_all_summaries()

    for step, (x, y_1, y_3, t) in enumerate(semeval_itterator(x_data,
                                                                 y_data,
                                                                 config.batch_size,
                                                                 config.num_steps,
                                                                 target=target,
                                                                 polarity=y_polarity)):
        if writer:

            slot1_cost, slot3_cost, \
            slot1_loss, slot3_loss, \
            summary, _, _ = session.run([m.slot1_model.cost, m.slot3_model.cost,
                                         m.slot1_model.loss, m.slot3_model.loss,
                                         merged,
                                        m.slot1_model.optimizer, m.slot3_model.optimizer],
                                       {m.embeddings.source_data: x,
                                        m.slot1_model.targets: y_1, m.slot3_model.targets: y_3,
                                        m.slot3_model.input_t: t},
                                                           options=run_options,
                                                           run_metadata=run_metadata)
        else:
            slot1_cost, slot3_cost, \
            slot1_loss, slot3_loss, _, _ = session.run([m.slot1_model.cost, m.slot3_model.cost,
                                                     m.slot1_model.loss, m.slot3_model.loss,
                                                     m.slot1_model.optimizer, m.slot3_model.optimizer],
                                                    {m.embeddings.source_data: x,
                                                     m.slot3_model.input_t: t,
                                                     m.slot1_model.targets: y_1, m.slot3_model.targets: y_3})

        if writer:
            writer.add_run_metadata(run_metadata, 'step%d' % step)
            writer.add_summary(summary, step)

        costs += slot1_cost
        iters += config.num_steps
        slot1_losses.append(slot1_loss)
        slot3_losses.append(slot3_loss)

        if verbose and iters % (config.batch_size * 5) == 0:
            print("step %.3f slot_loss : %.6f slot3_loss: %.6f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, slot1_loss, slot3_loss, iters * config.batch_size / (time.time() - start_time)))

    return np.exp(abs(costs) / iters), slot1_losses, slot3_losses


class slot1_and_slot3_model(object):
    def __init__(self, is_training, config):

        self._embeddings = shared_Embeddings(config)

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        self._rnn_output = shared_BiRNN(is_training,
                                        config,
                                        inputs=self._embeddings.inputs, initializer=initializer)


        """
            Both Slot1 and Slot3 Models share BiRNN above Representations
        """

        # Slot1 Model
        with tf.name_scope("slot1_EA"):
            self._slot1_model = slot1_entity_attribute(is_training,
                                                       config,
                                                       self._rnn_output.output, initializer=initializer)

        # Slot 3 Model
        with tf.name_scope("slot3_Sentiment"):
            self._slot3_model = slot3_polarity(is_training,
                                               config,
                                               self._rnn_output.output, initializer=initializer)

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def birrn_output(self):
        return self._rnn_output

    @property
    def slot1_model(self):
        return self._slot1_model

    @property
    def slot3_model(self):
        return self._slot3_model

    @property
    def merged_summary(self):
        return self._merged_summary


def train_task(CONST, data):
    config = multitask()

    config.vocab_size = len(data["embeddings"])
    config.max_max_epoch = CONST.MAX_EPOCH

    tf.reset_default_graph()

    with tf.Graph().as_default(), tf.Session(config=session_conf) as session:

        with tf.variable_scope("model", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
            training_model = slot1_and_slot3_model(is_training=CONST.TRAIN, config=config)  # model class

        if CONST.TRAIN:
            tf.initialize_all_variables().run()

            # Check if Model is Saved and then Load
            if CONST.RELOAD_TRAIN:
                saver = tf.train.Saver()
                print("CHECKPATH",CONST.MULTITASK_CHECKPOINT_PATH_HURSH + "task")
                saver.restore(sess=session, save_path=CONST.MULTITASK_CHECKPOINT_PATH_HURSH + "task")

            session.run(training_model.embeddings.embeddings.assign(data["embeddings"]))  # train

            slot1_losses = []
            slot3_losses = []
            slot4_losses = []

            train_writer = run_metadata = run_options = False

            if config.get_summary:
                # session = tf.InteractiveSession()
                train_writer = tf.train.SummaryWriter(
                    CONST.MULTITASK_CHECKPOINT_PATH_HURSH + "attention_graph/" + config.__class__.__name__,
                    session.graph)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            for i in range(config.max_max_epoch):  # setup epoch

                train_perplexity, slot1_loss, slot3_loss = run_epoch(session,
                                                                     training_model,
                                                                     data["x_train"],
                                                                     data["y_train"],
                                                                     data["p_train"],
                                                                     train_writer,
                                                                     run_options,
                                                                     run_metadata,
                                                                     target=data["l_train"],
                                                                     verbose=True,
                                                                     config=config)

                slot1_losses = slot1_losses + [np.mean(slot1_loss)]
                slot3_losses = slot3_losses + [np.mean(slot3_loss)]

                print("Epoch: %d Avg. Total Mean Loss slot1: %.6f slot3: %.6f" % (i + 1,
                                                                               np.mean(slot1_losses),
                                                                               np.mean(slot3_losses)))



            # Output Config/Losses
            from util.evaluations import print_config
            print_config(config)


            # Save Losses for Later
            loss = {"slot1": slot1_losses, "slot3": slot3_losses}
            save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "multi_task", loss)

            # Save CheckPoint
            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess=session, save_path=CONST.MULTITASK_CHECKPOINT_PATH_HURSH + "task")
            print("model saved: " + path)

            if config.get_summary:
                train_writer.close()

            session.close()

            # doesn't seem to close under scope??
            # Try and plot the losses
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            x = [i for i in range(len(slot1_losses))]
            plt.plot(np.array(x), np.array(slot1_losses))
            plt.plot(np.array(x), np.array(slot3_losses))
            plt.legend(['slot1', 'slot3'], loc='upper right')
            plt.savefig("losses_" + config.__class__.__name__ + ".png")

        if not CONST.TRAIN:
            with tf.variable_scope("model", reuse=True):  # reuse scope to evaluate model
                #validation_model = slot1_and_slot3_model(is_training=False, config=config)  # model class

                session.run(training_model.embeddings.embeddings.assign(data["embeddings"]))  # load embeddings

                saver = tf.train.Saver()
                saver.restore(sess=session, save_path=CONST.MULTITASK_CHECKPOINT_PATH_HURSH + "task")

                slot1_predictions = []
                slot3_predictions = []
                alphas = []

                # Get Predictions
                for step, (x, y_1, y_3, t) in enumerate(semeval_itterator(data["x_test"],
                                                                             data["y_test"],
                                                                             config.batch_size,
                                                                             config.num_steps,
                                                                             target=data["l_test"],
                                                                             polarity=data["p_test"],
                                                                            shuffle_examples=False)):
                    if CONST.HEATMAP:

                       alpha = session.run([training_model.slot3_model.alpha],
                                           {training_model.embeddings.source_data: x,
                                            training_model.slot1_model.targets: y_1,
                                            training_model.slot3_model.targets: y_3})

                       alphas = alphas + alpha[0].tolist()

                    else:
                        slot1_prediction, slot3_prediction = session.run([training_model.slot1_model.predictions,
                                                                          training_model.slot3_model.predictions],
                                                                         {training_model.embeddings.source_data: x,
                                                                          training_model.slot1_model.targets: y_1,
                                                                          training_model.slot3_model.targets: y_3})

                        slot1_predictions = slot1_predictions + slot1_prediction.tolist()
                        slot3_predictions = slot3_predictions + slot3_prediction.tolist()

                if not CONST.HEATMAP:
                    even_batch = len(data["x_test"]) % config.batch_size
                    remove_added_batch = config.batch_size - even_batch

                    del slot1_predictions[-remove_added_batch:]
                    del slot3_predictions[-remove_added_batch:]

                    slot1_predictions = [np.asarray(x) for x in slot1_predictions]
                    slot3_predictions = [np.asarray(x) for x in slot3_predictions]

                    # print congiuration for test predictions
                    from util.evaluations import print_config
                    print_config(config)

                    slot3_y = [np.asarray(x) for x in data["p_test"]]
                    # save predictions
                    predictions = {"slot1": slot1_predictions, "slot3": slot3_predictions,
                                   "slot1_y": data["y_test"], "slot3_y": slot3_y}
                    save_pickle(CONST.DATA_DIR + config.__class__.__name__ + "_predictions",
                                predictions)
                    print("predictions saved to file ", CONST.DATA_DIR + config.__class__.__name__ + "_predictions")

                    from util.evaluations import evaluate_multilabel
                    from util.evaluations import evaluate_multiclass
                    from util.evaluations import find_best_slot1
                    # evaluate_multilabel(predictions["slot1"], predictions["slot1_y"], CONST.THRESHOLD)

                    print("\nslot3 sentiment ...\n")
                    evaluate_multiclass(np.asarray(predictions["slot3"]), predictions["slot3_y"], True)

                    print("\nfinding best threshold slot1 E\A pairs...\n")
                    find_best_slot1("multitask_hursh", np.asarray(predictions["slot1"]), predictions["slot1_y"])
                elif CONST.HEATMAP:
                    even_batch = len(data["x_test"]) % config.batch_size
                    remove_added_batch = config.batch_size - even_batch

                    del alphas[-remove_added_batch:]
                    distance = 0
                    sentences = data["test_sentences"]
                    print("computing heatmaps and avg distance...")
                    from util.heatmap import avg_distance_and_heatmaps
                    avg_distance_and_heatmaps(alphas, sentences, CONST.SENTENCE_PATH + "/hursh_multitask")



if __name__ == "__main__":
    # Set and Overload Arguments
    CONST.parse_argument(argparse.ArgumentParser())

    # Set Time of Experiment
    now = datetime.datetime.now()
    time_stamp = "_".join([str(a) for a in [now.month, now.day, now.hour, now.minute, now.second]])

    data = load(CONST.DATA_DIR + CONST.DATA_FILE)

    train_task(CONST, data)
