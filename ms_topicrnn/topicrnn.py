import tensorflow as tf
import numpy as np
import util 
import time
import os

from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.layers.core import Dense

try:
  linear = tf.layers.dense
except:
  from tensorflow.python.ops import rnn_cell_impl
  linear = rnn_cell_impl._linear
  #from tensorflow.python.ops.rnn_cell import _linear as linear

class TopicRNN:
  def __init__(self, sess, config, reader):
    self.sess = sess
    self.reader = reader
    self.dataset = config.dataset
    self.task = config.task
    self.projector_embed_dim = config.projector_embed_dim
    self.generator_embed_dim = config.generator_embed_dim
    self.n_topics = config.n_topics
    self.n_stops = config.n_stops
    self.vocab_size = config.vocab_size
    self.n_layers = config.n_layers
    self.n_hidden = config.n_hidden
    self.dropout = config.dropout
    self.max_grad_norm = config.max_grad_norm
    self.total_epoch = config.total_epoch
    self.init_scale = config.init_scale
    self.checkpoint_dir = config.checkpoint_dir
    self.cell_type = config.cell_type

    self.lda_vocab_size = self.vocab_size - self.n_stops

    self.step = tf.Variable(0, dtype=tf.int32, 
    			trainable=False, name="global_step")

    self.lr = tf.train.exponential_decay(
        	      config.learning_rate, self.step, config.decay_step, 
        	      config.decay_rate, staircase=True, name="lr")

    self._attrs = ["n_hidden", "n_topics", "n_layers", "total_epoch"]

    self.saver = tf.train.Saver()

    self.build_model()

  def get_model_dir(self):
    model_dir = self.dataset
    for attr in self._attrs:
      if hasattr(self, attr):
        model_dir += "/%s=%s" % (attr, getattr(self, attr))
    return model_dir

  def save(self, checkpoint_dir, global_step=None):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__
    model_dir = self.get_model_dir()

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, 
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def initialize(self, log_dir="./logs"):
    self.merged_sum = tf.merge_all_summaries()
    self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_iter = self.step.eval()

  def load(self, checkpoint_dir):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    model_dir = self.get_model_dir()
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Load SUCCESS")
      return True
    else:
      print(" [!] Load failed...")
      return False

  def build_model(self):
    self._X = tf.placeholder(tf.int32, [None, None], name="X")
    self._Xc = tf.placeholder(tf.float32, 
                [self.vocab_size], name="Xc")
    self._Y = tf.placeholder(tf.int32, [None, None], name="Y")
    #L...observed stop word indicators for computing elbo
    self._L = tf.placeholder(tf.int32, [None, None], name="L")
    #sequence lengths
    self._seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    #batch size
    self._n_batch = tf.placeholder(tf.int32, name="batch_size")
    #build the inference network
    self.build_projector()
    #build the generator
    self.build_generator()
    #compute elbo
    self.compute_elbo()
    #compute cross entropy
    self.compute_cross_entropy()
    #optimizer and gradients
    trainable_vars = tf.trainable_variables()
    
    grads, _ = tf.clip_by_global_norm(
    			        tf.gradients(self.loss, trainable_vars), 
    				        self.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.optim = optimizer.apply_gradients(zip(grads, trainable_vars), 
    				      global_step=self.step)
    
    #tf.scalar_summary("learning rate", self.lr)
  
  def forward_data(self, ):
    print('hello world') 
    
  def build_unsupervised_model(self):
    ## logic 
    self._X = tf.placeholder(tf.int32, [None, None], name="X")
    self._Xc = tf.placeholder(tf.float32, 
                [self.vocab_size], name="Xc")
    self._Y = tf.placeholder(tf.int32, [None, None], name="Y")
    #L...observed stop word indicators for computing elbo
    self._L = tf.placeholder(tf.int32, [None, None], name="L")
    #sequence lengths
    self._seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    #batch size
    self._n_batch = tf.placeholder(tf.int32, name="batch_size")
    #build the inference network
    self.build_projector()
    #build the generator
    self.build_generator()
    #compute elbo
    self.compute_elbo()
    #compute cross entropy
    self.compute_cross_entropy()
    #optimizer and gradients
    trainable_vars = tf.trainable_variables()
    
    grads, _ = tf.clip_by_global_norm(
    			        tf.gradients(self.loss, trainable_vars), 
    				        self.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.optim = optimizer.apply_gradients(zip(grads, trainable_vars), 
    				      global_step=self.step)

    #tf.scalar_summary("learning rate", self.lr)

  def build_projector(self):
    with tf.variable_scope("projector"):
      with tf.variable_scope("l1"):
        self.l1_lin = linear(tf.expand_dims(self._Xc, 0), self.projector_embed_dim, use_bias=True)
        self.l1 = tf.nn.relu(self.l1_lin)

      with tf.variable_scope("l2"):
        self.l2_lin = linear(self.l1,	 
      		            self.projector_embed_dim, use_bias=True)
        self.l2 = tf.nn.relu(self.l2_lin)

      with tf.variable_scope("mu"):
        self.mu = linear(self.l2, self.n_topics, use_bias=True)

      with tf.variable_scope("log_sigma_sq"):
        self.log_sigma_sq = linear(self.l2, self.n_topics, use_bias=True)
        
      self.sigma_sq = tf.exp(self.log_sigma_sq)
      self.sigma = tf.sqrt(self.sigma_sq)
      self.eps = tf.random_normal((1, self.n_topics), 0, 1, dtype=tf.float32)
      self.theta = tf.add(self.mu, tf.expand_dims(tf.multiply(tf.squeeze(self.sigma), tf.squeeze(self.eps)), 0))
      #print("theta: ", self.theta)
      #self.theta_prop = tf.nn.softmax(self.theta)
      #self.theta = tf.nn.softmax(tf.identity(self.mu)) #point estimate
      #tf.histogram_summary("theta", self.theta)

  def build_generator(self):
    initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)

    with tf.variable_scope("generator", initializer=initializer):
      if self.cell_type == 'rnn':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.n_hidden)
      elif self.cell_type == 'lstm':
        rnn_cell = LSTMCell(self.n_hidden)
      elif self.cell_type == 'gru':
        rnn_cell = GRUCell(self.n_hidden)

      if self.dropout < 1:
        rnn_cell = DropoutWrapper(rnn_cell, dtype=tf.float32,
                                  output_keep_prob=self.dropout,)

      #cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * self.n_layers)
      cell = MultiRNNCell([rnn_cell for i in range(self.n_layers)])
      
      with tf.device("/device:GPU:0"): #"/cpu:0"):
        embedding = tf.get_variable("embedding", 
                    [self.vocab_size, self.generator_embed_dim])
        self.embedding = embedding #to visualize nearest neighbors
        
        X_minus_1 = self._X - 1
        mask = tf.sign(tf.to_float(self._X))
        X_minus_1 = tf.cast(mask, tf.int32) * X_minus_1
        
        inputs = tf.nn.embedding_lookup(embedding, X_minus_1) 
        
    	if self.dropout < 1:
    		inputs = tf.nn.dropout(inputs, self.dropout)

      input_layer = Dense(self.n_hidden, dtype=tf.float32, name='input_projection')
      inputs = input_layer(inputs)
      #print("inputs: ", inputs)
      outputs, state = tf.nn.dynamic_rnn(cell, inputs, 
                            sequence_length=self._seq_len,
                            dtype=tf.float32)

      if self.dropout < 1:
        outputs = tf.nn.dropout(outputs, self.dropout)

      output = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])
      self.final_state = state #final state for each sequence

      self.V = tf.get_variable("V", [self.n_hidden, self.vocab_size])
      self.b = tf.get_variable("b", [self.vocab_size])
      Gamma = tf.get_variable("Gamma", [self.n_hidden])
      self.B = tf.get_variable("B", [self.lda_vocab_size, self.n_topics])
      
      logits = tf.matmul(output, self.V) + self.b
      self.logits = logits
      self.logits_stops = logits_stops = logits[:, self.lda_vocab_size : self.vocab_size]
      self.logits_nonstops = logits_nonstops = logits[:, :self.lda_vocab_size]
      #p_stops is the probability that the next word is a stop word
      self.p_stops = tf.sigmoid(tf.matmul(output, tf.expand_dims(Gamma, 1)))
      #reshape and mask the padded values
      pred_stops = self.p_stops * tf.nn.softmax(logits_stops)
      self.pred_stops = tf.concat([tf.zeros_like(logits_nonstops), pred_stops], 1)
      topic_bias = tf.matmul(self.B, self.theta, transpose_b=True)
      topic_bias = tf.reshape(topic_bias, (1, self.lda_vocab_size))
      #mask 1- self.p_stops to have zeros in the padded entries
      #print("afte preds")
      pred_nonstops = (1 - self.p_stops) * tf.nn.softmax(logits_nonstops + topic_bias)
      self.pred_nonstops = tf.concat([pred_nonstops, tf.zeros_like(logits_stops)], 1)
      self.p_y_i = self.pred_stops + self.pred_nonstops
  
  def build_km_on_rnn(self): 
    ## this function builds the projection layer for k-means using rnns 
    
    initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)
    
    ## build the kmeans model from additional variables: 
    ## centroids [K x hidden_dim] 
    ## assignments [D x K] 
    
    with tf.variable_scope("kmeans", initializer=initializer):
      if self.cell_type == 'rnn':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.n_hidden)
      elif self.cell_type == 'lstm':
        rnn_cell = LSTMCell(self.n_hidden)
      elif self.cell_type == 'gru':
        rnn_cell = GRUCell(self.n_hidden)
      
      if self.dropout < 1:
        rnn_cell = DropoutWrapper(rnn_cell, dtype=tf.float32,
                                  output_keep_prob=self.dropout,)
      
      #cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * self.n_layers)
      cell = MultiRNNCell([rnn_cell for i in range(self.n_layers)])
      
      with tf.device("/device:GPU:0"): #"/cpu:0" 
        km_embedding = tf.get_variable("km_embedding",  #[todo: create km_embedding variable] 
                    [self.vocab_size, self.generator_embed_dim])
        self.km_embedding = km_embedding #to visualize nearest neighbors
        
        X_minus_1 = self._X - 1
        mask = tf.sign(tf.to_float(self._X))
        X_minus_1 = tf.cast(mask, tf.int32) * X_minus_1
        
        inputs = tf.nn.embedding_lookup(km_embedding, X_minus_1) 
        
    	if self.dropout < 1:
    		inputs = tf.nn.dropout(inputs, self.dropout)
      
      input_layer = Dense(self.n_hidden, dtype=tf.float32, name='input_projection')
      inputs = input_layer(inputs)
      #print("inputs: ", inputs)
      outputs, state = tf.nn.dynamic_rnn(cell, inputs, 
                            sequence_length=self._seq_len,
                            dtype=tf.float32)

      if self.dropout < 1:
        outputs = tf.nn.dropout(outputs, self.dropout)

      output = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])
      self.final_state = state #final state for each sequence

      self.V = tf.get_variable("V", [self.n_hidden, self.vocab_size])
      self.b = tf.get_variable("b", [self.vocab_size])
      Gamma = tf.get_variable("Gamma", [self.n_hidden])
      self.B = tf.get_variable("B", [self.lda_vocab_size, self.n_topics])

      logits = tf.matmul(output, self.V) + self.b
      self.logits = logits
      self.logits_stops = logits_stops = logits[:, self.lda_vocab_size : self.vocab_size]
      self.logits_nonstops = logits_nonstops = logits[:, :self.lda_vocab_size]
      #p_stops is the probability that the next word is a stop word
      self.p_stops = tf.sigmoid(tf.matmul(output, tf.expand_dims(Gamma, 1)))
      #reshape and mask the padded values
      pred_stops = self.p_stops * tf.nn.softmax(logits_stops)
      self.pred_stops = tf.concat([tf.zeros_like(logits_nonstops), pred_stops], 1)
      topic_bias = tf.matmul(self.B, self.theta, transpose_b=True)
      topic_bias = tf.reshape(topic_bias, (1, self.lda_vocab_size))
      #mask 1- self.p_stops to have zeros in the padded entries
      #print("afte preds")
      pred_nonstops = (1 - self.p_stops) * tf.nn.softmax(logits_nonstops + topic_bias)
      self.pred_nonstops = tf.concat([pred_nonstops, tf.zeros_like(logits_stops)], 1)
      self.p_y_i = self.pred_stops + self.pred_nonstops

  def compute_elbo(self):
    tmp = tf.matmul(self.B, self.theta, transpose_b=True)
    tmp = tf.concat([tmp, tf.zeros([self.n_stops, 1])], 0)
    
    Y = tf.reshape(self._Y, [-1])
    #need Y-1 for embedding_look_up but Y has 0 padded so do Y-1 and mask again
    Y_minus_1 = Y - 1
    mask = tf.sign(tf.to_float(Y))
    Y_minus_1 = tf.cast(mask, tf.int32) * Y_minus_1
    topic_biases = tf.nn.embedding_lookup(tmp, Y_minus_1)
    topic_biases = tf.reshape(topic_biases, [-1])
    L = tf.to_float(tf.reshape(self._L, [-1])) #this is where I changed to l x softmax + 1-l x softmax

    tmp = tf.nn.softmax(self.logits_stops)
    tmp = tf.concat([tf.zeros_like(self.logits_nonstops), tmp], 1)
    target_one_hots = tf.one_hot(tf.reshape(self._Y, [-1]), self.vocab_size)
    tmp = tmp * target_one_hots
    tmp = tf.reduce_sum(tmp, reduction_indices=1)
    tmp = L * tmp
    
    nonstops = tf.nn.softmax(self.logits_nonstops)
    nonstops = tf.concat([nonstops, tf.zeros_like(self.logits_stops)], 1)
    nonstops = nonstops * target_one_hots
    nonstops = tf.reduce_sum(nonstops, reduction_indices=1)
    nonstops = (1-L) * nonstops
    
    tmp = tf.log(tmp + nonstops + 1e-10)

    tmp = tmp + tf.multiply(L, tf.reshape(tf.log(self.p_stops + 1e-10), [-1]))
    tmp = tmp + tf.multiply(1-L, tf.reshape(tf.log(1 - self.p_stops + 1e-10), [-1]))
    tmp = mask * tmp
    tmp = tf.reduce_sum(tmp)

    kl_term = tf.reduce_sum(self.sigma_sq)
    kl_term = kl_term + tf.squeeze(tf.matmul(self.mu, self.mu, transpose_b=True))
    kl_term = kl_term - self.n_topics
    kl_term = kl_term - tf.reduce_sum(self.log_sigma_sq)

    elbo = tmp - kl_term
    self.loss = -elbo

    #tf.scalar_summary("-elbo", self.loss)

  def compute_cross_entropy(self):
    Y = tf.reshape(self._Y, [-1])
    target_one_hots = tf.one_hot(Y, self.vocab_size)

    preds = self.p_y_i * target_one_hots
    preds = tf.reduce_max(preds, reduction_indices=1)
    preds = tf.reshape(preds, [-1, 1])
    
    ce = util.cross_entropy_loss(Y, preds, self.vocab_size)
    mask = tf.sign(tf.to_float(Y)) #puts zero wherever Y is zero and 1 otherwise
    self.ce = tf.reduce_sum(ce * mask)

    #tf.scalar_summary("document cross entropy", self.ce)

  def compute_perplexity(self, iterator, sess, n_docs):
    costs = 0.0
    iters = 0
    
    for d in xrange(int(n_docs)):
      X, Xc = iterator[0].next()
      Y, L, seq_len, n_batch = iterator[1].next()

      if d == 0:
        if self.task == "word_prediction":
          theta = sess.run(self.theta, feed_dict={self._Xc: Xc})
          continue

      if self.task == "sentiment_analysis":
        theta = sess.run(self.theta, feed_dict={self._Xc: Xc})

      feed_dict = {self.theta: theta,
                   self._X: X,
                   self._Xc: Xc,
                   self._Y: Y,
                   self._L: L,
                   self._seq_len: seq_len,
                   self._n_batch: n_batch}

      ce, mu = sess.run([self.ce, self.mu], feed_dict=feed_dict)

      if self.task == "word_prediction": theta = mu

      costs += ce
      iters += np.sum(seq_len)

    return np.exp(costs/iters)

  def train(self, config):
    #merged_sum = tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter("./logs", self.sess.graph)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_time = time.time()
    start_iter = self.step.eval()
    iterator = self.reader.iterator()
    iterator_val = self.reader.iterator(data_type="valid")
    training_steps = int(self.reader.n_train_docs * self.total_epoch)

    costs = 0.0
    iters = 0
    for step in xrange(start_iter, training_steps):
      X, Xc = iterator[0].next()
      Y, L, seq_len, n_batch = iterator[1].next()

      feed_dict = {self._X: X,
        	   self._Xc: Xc,
        	   self._Y: Y,
        	   self._L: L,
             self._seq_len: seq_len,
             self._n_batch: n_batch}

      _, loss, ce, lr = self.sess.run(
              [self.optim, self.loss, self.ce, self.lr], 
              feed_dict=feed_dict)

      # _, loss, ce, lr, summary_str = self.sess.run(
      #   			[self.optim, self.loss, self.ce, self.lr, merged_sum], 
      #   			feed_dict=feed_dict)

      costs += ce
      iters += np.sum(seq_len)

      # if step % 2 == 0:
      #   writer.add_summary(summary_str, step)

      if step % 10 == 0:
        print("Step: [%4d/%4d] time: %4.4f, lr: %.8f, loss: %.8f" \
        	% (step, training_steps, time.time() - start_time, lr, loss))

      if step != 0 and step % 1000 == 0:
        self.save(self.checkpoint_dir, step)

      min_pp_val = 1000000
      if step != 0 and (step + 1) % self.reader.n_train_docs == 0:
        #one epoch completed...print train and valid perplexity
        print("Epoch: [%4d/%4d] time: %4.4f, perplexity: %.8f" \
          % (step/self.reader.n_train_docs + 1, self.total_epoch, time.time() - start_time, np.exp(costs / iters)))
        pp_val = self.compute_perplexity(iterator_val, self.sess, self.reader.n_valid_docs)
        print("validation perplexity: %.8f" % (pp_val))
        if pp_val < min_pp_val:
          self.save(self.checkpoint_dir, step)
          min_pp_val = pp_val

        costs = 0.0
        iters = 0

  def generate_text(self, sess, start_text='eos', n_words=50):
    print("generating some text...")
    #pick a seed document...for example the 10th doc from the training data 
    iterator = self.reader.iterator()
    for i in range(1):
      X, Xc = iterator[0].next()
      Y, L, seq_len, n_batch = iterator[1].next()
    #get theta
    seed_text = [self.reader.idx2word[word_idx] 
                  for word_idx in list(np.reshape(X, [-1])) if word_idx!=0]
    print("ptb seed text: ", seed_text)
    util.save_pkl('seed_text_gru.pkl', seed_text)
    theta, theta_prop, state = sess.run([self.theta, self.theta_prop, self.final_state], 
                                   feed_dict={self._Xc: Xc, 
                                              self._X: X,
                                              self._seq_len: seq_len})
    print("corresponding topic distribution for the seed text: {}".format(theta_prop))
    util.save_pkl('./theta_gru.pkl', theta_prop)
    tokens = [self.reader.vocab[word] for word in start_text.split()]

    for i in xrange(n_words):
      X = np.reshape(np.array([tokens[-1:]]), [1, 1])
      feed_dict = {self._X: X,
                   self._seq_len: [1],
                   self.theta: theta,
		   self.final_state: state}
      state, pred = sess.run(
          [self.final_state, self.p_y_i], feed_dict=feed_dict)

      next_word_idx = np.random.choice( 
                      np.arange(self.reader.vocab_size), 
                      replace=False, p=pred.reshape([-1]))
      tokens.append(next_word_idx)

    output = [self.reader.idx2word[word_idx] for word_idx in tokens]

    return output
  
  def get_hs_from_datatype(self, iterator, sess):
    h_list = []
    for i in range(25000):
      print("i: {}".format(i))
      X, Xc = iterator[0].next()
      Y, L, seq_len, n_batch = iterator[1].next()
      if n_batch == 0: continue
      feed_dict = \
            {self._X: X,
             self._Xc: Xc,
             self._Y: Y,
             self._L: L,
             self._seq_len: seq_len,
             self._n_batch: n_batch}
      h_i, h_prime_i = sess.run([self.final_state, self.theta], feed_dict=feed_dict)
      
      #pick the last state and concatenate with topics
      n_topics = self.n_topics
      vect = list(h_i[len(h_i)-1][0]) + list(np.reshape(h_prime_i, [n_topics])) 
      h_list.append(vect)
    
    h_list = np.array(h_list)
    
    return h_list
