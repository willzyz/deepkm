# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time, os, sys 

import tensorflow as tf

import cifar10
import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = cifar10.inference(images)
    
    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def train_deepkmeans(): 
  ## logic: 
  
  ## create the model and build the graph with both h-output as session-run target 
  ## and with nca objective and update as session-run target
  
  ## create a textlinedataset [DataAllSequences] with all text data 
  
  ## for loop epoch: {
  
  ## forward propagate the whole dataset DataAllSequences through 
  ## the model [session.run(h-output)] to obtain HvecAllSequences 
  
  ## perform pca and k-means on the numpy array of hidden embedding 
  ## data and obtain centroids and cluster assignments, we can use a large set, e.g. 5000 
  ## at first for language model word prediction routing. (Later, consider a small set, e.g. 50 for
  ## sentiment classification routing) 
  
  ## create a zipped dataset [ZippedClusDataset] across 
  ## [DataAllSequences, Centroids, ClusterAssignment] 
  ## and create an iterator [FinetuneClusIterator] for iterative gd optimization 
  
  ## for loop step: iteratively train the model using session.run(model-finetune-update) 
  ## }
  

  # Get images and labels for CIFAR-10.
  # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
  # GPU and resulting in a slow down.
  
  dkm_num_centroids = 300 
  learning_rate = 0.001 
  
  graph = tf.Graph()
  with graph.as_default(): 
    with tf.device('/cpu:0'):
      mb_images, mb_labels = cifar10.distorted_inputs()
      images, labels = cifar10.alltraininputs()
  
  # Build a Graph that computes the logits predictions from the
  # inference model.
  # logits = cifar10.inference(images)

  # Calculate loss.
  # loss = cifar10.loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters.
  #train_op = cifar10.train(loss, global_step)
  
  ## this is the forward propagation graph 
  
  sess = tf.Session(graph=graph) 
  
  with graph.as_default(): 
    _, local3_acts = cifar10.inference(images) 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./modelsave/deepkmeans.ckpt.epoch_68.step_1561.global_step_107777")

    var = [v for v in tf.trainable_variables() if v.name == "conv1/weights:0"]
    kernel = sess.run(var) 
    import sys; sys.path.insert(1, './conviz/') ; import conviz
    weights = kernel[0]
    name = 'conv1' 
    conviz.plot_conv_weights(weights, name, channels_all=True)
    
  
  global_step = 0
  for epoch in range(200):  
    with graph.as_default(): 
      print('------------ starting epoch: ' + str(epoch) + '-------------------------')  
      #with graph.as_default():
      embeddings = sess.run(local3_acts) 
      #print('Sanity check - printing out first 5 elements of embeddings:') 
      #print(embeddings[0:5])
      #print(embeddings.shape)
      
      ## use Kmeans from kmtools (FAISS) 
      from kmtools import Kmeans
      KM = Kmeans(dkm_num_centroids) ## todo: define this hparams
      data_assignment, km_centroids, km_loss, pcawhiten_mat = KM.cluster(embeddings)
      
      print('k-means loss: ' + str(km_loss))
      #print('k-means assignment: ')
      #print(data_assignment)
      #print('k-means centroids: ') 
      #print(km_centroids) 
      #print(km_centroids.shape) 
      
      import numpy as np 
      km_centroids = np.reshape(km_centroids, [dkm_num_centroids, 128]) 
      #km_data = KMData(centroids=km_centroids, pcawhiten_mat=pcawhiten_mat) 
      
      ## next use data_sequences, data_assignment to make new [dataset, iterator] 
      
      data_assignment = np.reshape(data_assignment, [embeddings.shape[0], ])
      
      ft_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(images), tf.data.Dataset.from_tensor_slices(data_assignment)))
      ft_dataset = ft_dataset.prefetch(-1)
      ft_dataset = ft_dataset.shuffle(embeddings.shape[0]).repeat().batch(FLAGS.batch_size) 
      ft_iterator = tf.data.make_initializable_iterator(ft_dataset) 
    
    with graph.as_default(): 
      #ft_dataset = tf.data.Dataset.from_tensor_slices((, data_seq_lens, data_assignment)) 
      
      mb_images, mb_assigns = ft_iterator.get_next() 
      
      mb_logits, _ = cifar10.inference(mb_images)
      
      # Calculate loss. 
      train_loss = cifar10.loss(mb_logits, mb_assigns) 
      with tf.variable_scope('adam_ft', reuse=tf.AUTO_REUSE) as scope: 
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate) 
        train_op = optimizer.minimize(train_loss) 
      #train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
      
      ## remember to run the iterator initializer 
      sess.run(ft_iterator.initializer) 
      sess.run(tf.variables_initializer(optimizer.variables())) 
      ## run one epoch of fine-tuning 
      epoch_steps = int(embeddings.shape[0]/FLAGS.batch_size * 4 ) 
      for step in range(epoch_steps): 
        loss, up = sess.run([train_loss, train_op]) 
        global_step = step + epoch * epoch_steps 
        if step % 100 == 0: 
          print(' running fine-tune optimizer epoch : ' + str(epoch) + ' loss: ' + str(loss) +' -step: ' + str(step) + ', ' + ' -global_step: ' + str(global_step) + ' -lr: '+ str(learning_rate))
    
    print('------------ finished epoch: ' + str(epoch) + '-------------------------')
    saver.save(sess, os.path.join("./modelsave/deepkmeans.ckpt" + ".epoch_" + str(epoch) + ".step_" + str(step) + ".global_step_" + str(global_step))) 

def main(argv=None):  # pylint: disable=unused-argument
  
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train_deepkmeans() 

if __name__ == '__main__':
  tf.app.run()
