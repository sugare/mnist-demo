# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import time

import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


# build parameters
def build_flags():
    parser = argparse.ArgumentParser()
    parameters = [
        # str type default help
        ['--data_dir', str, '/clever/admin/mnist_data', 'Directory for storing mnist data'],
        ['--log_dir', str, '/clever/admin/log', 'Directory for train log'],
        ['--model_dir', str, '/clever/admin/mnist_model', 'Directory for storing mnist model'],
        ['--model_version', int, 1, 'Version of mnist model'],
        ['--num_gpus', int, 1, 'Total number of gpus for each machine'],
        ['--fake_data', bool, False, 'If true, uses fake data for unit testing'],
        ['--dropout', float, 0.9, 'Keep probability for training dropout'],
        ['--learning_rate', float, 0.001, 'Initial learning rate'],
        ['--max_steps', int, 500, 'Number of steps to run trainer'],

        ['--localhost', bool, False, 'Run locally or not'],

        #
        ['--task_index', int, None, 'Worker task index, should be >= 0'],
        ['--job_name', str, None, 'worker or ps or chief'],
        ['--task', dict, None, ''],
        ['--cluster', dict, None, ''],
        ['--has_chief', bool, None, ''],
        ['--is_chief', bool, None, ''],
    ]
    for p in parameters:
        parser.add_argument(
            p[0],
            type=p[1],
            default=p[2],
            help=p[3]
        )
    return parser


def init_tf_config():
    if FLAGS.localhost:
        # run local with one ps and two worker
        if FLAGS.job_name == 'ps':
            TF_CONFIG = {
                "cluster": {
                    "ps": ["localhost:2222"],
                    "worker": ["localhost:2223", "localhost:2224"]
                },
                "task": {"type": "ps", "index": 0}
            }
        elif FLAGS.job_name == 'worker1':
            TF_CONFIG = {
                "cluster": {
                    "ps": ["localhost:2222"],
                    "worker": ["localhost:2223", "localhost:2224"]
                },
                "task": {"type": "worker", "index": 0}
            }
        else:
            TF_CONFIG = {
                "cluster": {
                    "ps": ["localhost:2222"],
                    "worker": ["localhost:2223", "localhost:2224"]
                },
                "task": {"type": "worker", "index": 1}
            }

    else:
        TF_CONFIG = json.loads(os.environ.get("TF_CONFIG", "{}"))

    FLAGS.task = TF_CONFIG.get('task', {})
    FLAGS.cluster = TF_CONFIG.get('cluster', {})
    FLAGS.job_name = FLAGS.task['type']
    FLAGS.task_index = FLAGS.task['index']
    FLAGS.has_chief = 'chief' in FLAGS.cluster
    FLAGS.is_chief = (FLAGS.job_name == 'chief') or (not FLAGS.has_chief and FLAGS.task_index == 0)


def train():
    init_tf_config()
    server = tf.train.Server(
        tf.train.ClusterSpec(FLAGS.cluster),
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
    )
    # start ps server
    if FLAGS.job_name == 'ps':
        server.join()

    if FLAGS.num_gpus > 0:
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, 0)

    if FLAGS.is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
        tf.reset_default_graph()
    else:
        print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

    # Between-graph replication
    with tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=tf.train.ClusterSpec(FLAGS.cluster))):
        # count the number of updates
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False)

        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)

        # We can't initialize these variables to 0 - the network will get stuck.
        def weight_variable(shape):
            """Create a weight variable with appropriate initialization."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Reusable code for making a simple neural net layer.
            It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases)
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights) + biases
                    tf.summary.histogram('pre_activations', preactivate)
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations

        hidden1 = nn_layer(x, 784, 500, 'layer1')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder_with_default(1.0, shape=())
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(hidden1, keep_prob)

        # Do not apply softmax activation yet, see below.
        y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

        with tf.name_scope('cross_entropy'):
            logits = tf.nn.softmax(y, name='logits')
            cross_entropy = -tf.reduce_sum(y_ * tf.log(logits), name='cross_entropy')
            tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_step = opt.minimize(cross_entropy, global_step)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to
        # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()

        init_op = tf.global_variables_initializer()

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    sv = tf.train.Supervisor(
        is_chief=FLAGS.is_chief,
        global_step=global_step,
        init_op=init_op,
        logdir=FLAGS.log_dir
    )

    with sv.prepare_or_wait_for_session(server.target) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        # Train the model, and also write summaries.
        # Every 100th step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries
        local_step = 0
        while True:
            if local_step % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, local_step)
                print('Accuracy at step %s: %s' % (local_step, acc))
                # Record train set summaries, and train
            elif local_step % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % local_step)
                train_writer.add_summary(summary, local_step)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, local_step)

            local_step += 1
            global_train_step = tf.train.global_step(sess, global_step)

            now = time.time()
            print("%f: Worker %d: training step %d done (global step: %s)" %
                  (now, FLAGS.task_index, local_step, global_train_step))
            if int(global_train_step) >= FLAGS.max_steps:
                break

        # export model
        if FLAGS.is_chief:
            new_graph = tf.get_default_graph()
            new_x = new_graph.get_tensor_by_name('input/x-input:0')
            print(new_x)
            new_y = new_graph.get_tensor_by_name('cross_entropy/logits:0')
            print(new_y)

            export_path = os.path.join(
                compat.as_bytes(FLAGS.model_dir),
                compat.as_bytes(str(FLAGS.model_version)))
            print('Exporting trained model to', export_path)
            builder = saved_model_builder.SavedModelBuilder(export_path)

            # Build the signature_def_map.
            tensor_info_x = utils.build_tensor_info(new_x)
            tensor_info_y = utils.build_tensor_info(new_y)
            prediction_signature = signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=signature_constants.PREDICT_METHOD_NAME)

            sess.graph._unsafe_unfinalize()
            legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                },
                legacy_init_op=legacy_init_op,
                clear_devices=True)
            builder.save()
            sess.graph.finalize()
            print('Done exporting!')
        train_writer.close()
        test_writer.close()


def main(_):
    train()


if __name__ == '__main__':
    FLAGS, unparsed = build_flags().parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)