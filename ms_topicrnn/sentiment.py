#train topicrnn for sentiment analysis
#use a config file or FLAGS
#use checkpoints to save session for later use
#use summarywriter for tensorboard visualization

import tensorflow as tf
import numpy as np
import matplotlib

from matplotlib.backends.backend_pdf import PdfPages
from config import IMDBConfig
from imdb import IMDBReader
from topicrnn import TopicRNN 
from copy import deepcopy
from util import *
from sklearn.decomposition import PCA as sklearnPCA

np.random.seed(54)

def onehot(x):
  N = len(x)
  new_x = np.zeros([N, 2])

  for i in xrange(N):
    if x[i] == 1:
      new_x[i, 1] = 1
    else:
      new_x[i, 0] = 1

  return new_x

def network_classifier(config, htrain_list, labelstrain_list, 
                        htest_list, labelstest_list):
  """
  Computational graph for doing logistic regression
  with neural net with one hidden layer of 5o units
  """
  learning_rate = 0.05
  training_epochs = 50000
  batch_size = 25000
  display_step = 10
  n_topics = config.n_topics # MNIST data input (img shape: 28*28)

  # Network Parameters
  n_hidden_1 = 50 # 1st layer number of features
  n_classes = 2 # MNIST total classes (0-9 digits)
  n_input = config.n_layers*config.n_hidden+n_topics
  # tf Graph input
  x = tf.placeholder("float", [None, n_input])
  y = tf.placeholder("float", [None, n_classes])

  # Store layers weight & bias
  with tf.variable_scope("mlp"):
    weight_in = tf.get_variable("weight_in", [n_input, n_hidden_1])
    weight_out = tf.get_variable("weight_out", [n_hidden_1, n_classes])
    bias_in = tf.get_variable("bias_in", initializer=tf.random_normal([n_hidden_1]))
    bias_out = tf.get_variable("bias_out", initializer=tf.random_normal([n_classes]))

  layer_1 = tf.add(tf.matmul(x, weight_in), bias_in)
  layer_1 = tf.sigmoid(layer_1)
  
  pred = tf.add(tf.matmul(layer_1, weight_out), bias_out)
  
  # Define loss and optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  
  # Initializing the variables
  init = tf.initialize_all_variables()

  # Launch the graph
  with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
      avg_cost = 0.0
      total_batch = int(len(htrain_list)/batch_size)
      # Loop over all batches
      for i in range(total_batch):
        batch_x = np.array(htrain_list[i*batch_size: (i+1)*batch_size])
        batch_x = np.reshape(batch_x, [batch_size, n_input])        
        batch_y = np.array(labelstrain_list[i*batch_size: (i+1)*batch_size])
        batch_y = onehot(batch_y)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
      # Display logs per epoch step
      if epoch % display_step == 0:
        print("Epoch: {}...cost: {}".format(epoch+1, avg_cost))

    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    xtest = np.array(htest_list)
    xtest = np.reshape(xtest, [len(htest_list), n_input])
    ytest = np.array(labelstest_list)
    #print("ytest: ", np.array(ytest).shape)
    ytest = onehot(ytest)
    print("Accuracy: {}".format(accuracy.eval({x: xtest, y: ytest})))

def get_hs_from_datatype(model, iterator, sess):
  h_list = []
  for i in range(25000):
    print("i: {}".format(i))
    X, Xc = iterator[0].next()
    Y, L, seq_len, n_batch = iterator[1].next()
    if n_batch == 0: continue
    feed_dict = \
            {_X: X,
             _Xc: Xc,
             _Y: Y,
             _L: L,
             _seq_len: seq_len,
             _n_batch: n_batch}
    """
    feed_dict = \
            {self._X: X,
             self._Xc: Xc,
             self._Y: Y,
             self._L: L,
             self._seq_len: seq_len,
             self._n_batch: n_batch}
    """
    h_i, h_prime_i = sess.run([model.final_state, model.theta], feed_dict=feed_dict)
    
    #pick the last state and concatenate with topics
    n_topics = model.n_topics
    vect = list(h_i[len(h_i)-1, :]) + list(np.reshape(h_prime_i, [n_topics])) 
    h_list.append(vect)
    
  h_list = np.array(h_list)
  
  return h_list

def main(_):
  config = IMDBConfig()
  reader = IMDBReader(config)
  
  with tf.Session() as sess:
    model = TopicRNN(sess, config, reader)
    
    if config.forward_only:
      model.load(config.checkpoint_dir)
    
    else:
      model.train(config)
    
    #run classification and compute error rate
    train_labelled_iterator = reader.iterator(data_type='train_labelled')
    test_iterator = reader.iterator(data_type='test')
    
    if not os.path.exists('./data/imdb/htrain_list.pkl'):
      #htrain_list = get_hs_from_datatype(model, train_labelled_iterator, sess)
      htrain_list = model.get_hs_from_datatype(train_labelled_iterator, sess)      
      save_pkl('./data/imdb/htrain_list.pkl', htrain_list)
      htest_list = model.get_hs_from_datatype(test_iterator, sess)
      save_pkl('./data/imdb/htest_list.pkl', htrain_list)

    else:
      htrain_list = load_pkl('./data/imdb/htrain_list.pkl')
      htest_list = load_pkl('./data/imdb/htest_list.pkl')

    labelstrain_list = load_pkl('./data/imdb/train_labelled_labels.pkl')#reader.train_labelled_labels
    labelstest_list = load_pkl('./data/imdb/test_labels.pkl')#reader.test_labels

    htrain_list = htrain_list.tolist()
    #labelstrain_list = labelstrain_list.tolist()
    htest_list = htest_list.tolist()
    #labelstest_list = labelstest_list.tolist()
    
    network_classifier(config, htrain_list, labelstrain_list, htest_list, labelstest_list)

    #visualize topics...
    if config.print_topics:
      B, embedding = model.sess.run([model.B, model.embedding])
      tops = []
      for k in range(model.n_topics):
        topic_k = B[:, k]
        words = [reader.idx2word[idx] for idx in topic_k.argsort()[-10:][::-1]]
        tops.append(words)

      print("top words in the topics: ", tops)
      util.save_pkl('./B_matrix_gru.pkl', B)
      #printing the k-nearest neighbors
      vocab = util.load_pkl('./data/ptb/vocab.pkl')
      word_list = [vocab[w] for w in ['law', 'employees', 'democratic', 'stock', 'cars']]
      print("word list for computing nearest neighbors...: ", word_list)
      nearest_neighbors = []
      N = 10
      for word in word_list:
        neighbors_tokens = util.nearest_neighbor(embedding, word, N)
        neighbors_words = [model.reader.idx2word[idx] for idx in neighbors_tokens]
        nearest_neighbors.append(neighbors_words)
      print("nearest neighbor list: ", nearest_neighbors)
      util.save_pkl('./nearest_neighbors_gru.pkl', nearest_neighbors)
      util.save_pkl('./embedding_gru.pkl', embedding)
    
    #generate text...it should look like a movie review
    if config.generate_text:
      text = model.generate_text(model.sess, start_text='eos', n_words=50)
      print("generated text: {}".format(text))
      save_pkl('../data/imdb/generated_text.pkl', text)

    #clustering output of topicRNN for visualization purposes
    htest_list = list(load_pkl('./data/imdb/htest_list.pkl'))
    htest_list = htest_list[:4000]+htest_list[14200:15200]+htest_list[20000:]
    print("clustering some documents...")
    noofclusters = 2
    vectors = np.array(htest_list)
    centroids, assignments = TFKMeansCluster(vectors, noofclusters)
    save_pkl('./imdb_latest_assignments_10000docs', assignments)
    assignments = load_pkl('./imdb_latest_assignments_10000docs')
    #plot the output using pos/neg tag...
    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(vectors)
    print("pca results: ", sklearn_transf)
    x = sklearn_transf[:, 0]
    y = sklearn_transf[:, 1]
    label = assignments
    colors = ['red','green']
    
    with PdfPages('./clustersmoredocs.pdf') as pdf:
      plt.figure(figsize=(30, 8))
      plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors), marker='o', s=30)
      plt.axis('off')
      pdf.savefig()  
      plt.show()

if __name__ == '__main__':
	tf.app.run()
