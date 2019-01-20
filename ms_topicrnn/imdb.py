import os
import itertools
import numpy as np
import nltk
import math

import tensorflow as tf

from util import *
from collections import Counter
from config import IMDBConfig

np.random.seed(54)

class IMDBReader(object):
  def __init__(self, config):
    self.eos_token = config.eos_token
    self.lda_vocab_size = config.vocab_size - config.n_stops
    self.vocab_size = config.vocab_size
    self.n_stops = config.n_stops
    self.data_path = data_path = config.data_path

    self.vocab_path = vocab_path = os.path.join(data_path, "vocab.pkl")

    #use train data to build vocabulary
    if os.path.exists(vocab_path):
      self._load()
    else:
      self.trainposdir = './data/imdb/train/pos/'
      self.trainnegdir = './data/imdb/train/neg/'
      self.unsupdir = './data/imdb/train/unsup/'
      self.testposdir = './data/imdb/test/pos/'
      self.testnegdir = './data/imdb/test/neg/'
      print("getting and saving the train, valid and test ids...")
      #get train, valid, train labelled, and test labelled
      print("getting the test data...")
      test, self.test_labels = \
              self.get_labelled_data(self.testposdir, self.testnegdir, 'test')
      save_pkl('./data/imdb/test_labels.pkl', self.test_labels)
      print("getting the labelled training data...")
      train_labelled, self.train_labelled_labels = \
              self.get_labelled_data(self.trainposdir, self.trainnegdir, 'train')
      save_pkl('./data/imdb/train_labelled_labels.pkl', self.train_labelled_labels)
      print("getting the unlabelled training data...")
      train_unlabelled = \
              self.get_unlabelled_data(self.unsupdir)
      all_train = train_labelled + train_unlabelled
      permutation = list(np.random.choice(range(75000), 75000, replace = False))
      all_train = list(np.array(all_train)[permutation])
      train = all_train[:65000]
      valid = all_train[65000:]
      print("creating the vocabulary...")
      self._build_vocab(train, vocab_path)

      self.X_test_data, _ = self._file_to_data(test, data_type='test')
      self.X_train_data, self.Y_train_data = self._file_to_data(train, data_type='train')
      self.X_valid_data, self.Y_valid_data = self._file_to_data(valid, data_type='valid')
      self.X_train_labelled_data, _ = self._file_to_data(train_labelled, data_type='train_labelled')

    self.n_train_docs = math.ceil((len(self.X_train_data) + 0.0))
    self.n_valid_docs = math.ceil((len(self.X_valid_data) + 0.0))
    self.n_test_docs = math.ceil((len(self.X_test_data) + 0.0))

    self.idx2word = {v:k for k, v in self.vocab.items()}

    self.vocab_size = len(self.vocab)
    print("vocabulary size: {}".format(self.vocab_size))
    print("number of training documents: {}".format(self.n_train_docs))
    print("number of validation documents: {}".format(self.n_valid_docs))
    print("number of testing documents: {}".format(self.n_test_docs))

  def get_labelled_data(self, posdir, negdir, data_type):
    pos = os.listdir(posdir)
    neg = os.listdir(negdir)
    sent, tar = [], []
    for i in pos:
        content = open(posdir+i).read()
        sent.append(content)
        tar.append(1)
    for i in neg:
        content = open(negdir+i).read()
        sent.append(content)
        tar.append(0)

    permutation = list(np.random.choice(range(25000), 25000, replace = False))

    sent = np.array(sent)
    sent = list(sent[permutation])
    tar = np.array(tar)
    tar = list(tar[permutation])

    return sent, tar

  def get_unlabelled_data(self, unsupdir):
    loc = os.listdir(unsupdir)
    sent = []
    for i in loc:
        content = open(unsupdir+i).read()
        sent.append(content)

    return sent


  def _read_text(self, file_path):
    with open(file_path) as f:
      return f.read().replace("\n", " %s " % self.eos_token)

  def _build_vocab(self, train, vocab_path):
    # with open('./data/imdb/imdb.vocab') as f:
    #   unique_tokens = f.read().split()

    #get counts of words in train and pick the first vocab_size words put the rest as 'unk'
    #train = ' '.join(train)
    tokenized_train = [nltk.sent_tokenize(x.decode('utf-8').lower()) for x in train]
    tokenized_train = [[nltk.word_tokenize(sent) for sent in doc] for doc in tokenized_train]
    clean_tokenized_train = [[[word for word in sent if word not in ['.', ',', '/', '!', '?', '-', '*', ':', ';', "'",'br', '``', '<', '>', ')', '(', '...', '--', "''"]] for sent in doc if sent != []] for doc in tokenized_train]
  
    train = [word for doc in clean_tokenized_train for sent in doc for word in sent]
    print("train length: ", len(train))
    word_freq = nltk.FreqDist(train)
    vocab = word_freq.most_common(self.vocab_size - 2)
    vocab_words = [x[0] for x in vocab]
    print("vocab words: ", vocab_words)
    print("vocab words length: ", len(vocab_words))

    with open('./data/stop_words.txt', 'rb') as f:
      stops = f.read().split()

    stops_in_train = [x for x in stops if x in vocab_words]
    stops_in_train.append('unk')
    stops_in_train.append('eos')
    print("number of stop words: {}".format(len(stops_in_train)))
    content_in_train = [x for x in vocab_words if x not in stops_in_train]

    vocab = {x:i+1 for i, x in enumerate(content_in_train)}
    for s in stops_in_train:
      vocab[s] = len(vocab)+1

    self.vocab = vocab
    print("official vocab length: ", len(vocab))
    save_pkl(vocab_path, self.vocab)
  
  def _file_to_data(self, data, data_type):
    tokenized_docs = [nltk.sent_tokenize(x.decode('utf-8').lower()) for x in data]
    tokenized_docs = [[nltk.word_tokenize(sent) for sent in doc] for doc in tokenized_docs]
    clean_tokenized_docs = [[[word for word in sent if word not in ['.', ',', '/', '!', '?', '-', '*', ':', ';', "'",'br', '``', '<', '>', ')', '(', '...', '--', "''"]]+['eos'] for sent in doc if sent != []] for doc in tokenized_docs]
    clean_tokenized_docs = [[[word if word in self.vocab else 'unk' for word in sent] for sent in doc] for doc in clean_tokenized_docs]
    X = [[sent[:-1] for sent in doc] for doc in clean_tokenized_docs]
    Y = [[sent[1:] for sent in doc] for doc in clean_tokenized_docs]
    X = np.asarray([[map(self.vocab.get, sent) for sent in doc] for doc in X])
    Y = np.asarray([[map(self.vocab.get, sent) for sent in doc] for doc in Y])
    
    save_pkl(self.data_path+'/'+'X_'+data_type+'.pkl', X)
    save_pkl(self.data_path+'/'+'Y_'+data_type+'.pkl', Y)
    
    return X, Y
  
  def _load(self):
    self.vocab = load_pkl(self.vocab_path)
    self.X_train_data = load_pkl(self.data_path+'/'+'X_train'+'.pkl')
    self.Y_train_data = load_pkl(self.data_path+'/'+'Y_train'+'.pkl')

    self.X_valid_data = load_pkl(self.data_path+'/'+'X_valid'+'.pkl')
    self.Y_valid_data = load_pkl(self.data_path+'/'+'Y_valid'+'.pkl')
    
    self.X_test_data = load_pkl(self.data_path+'/'+'X_test'+'.pkl')
    self.Y_test_data = load_pkl(self.data_path+'/'+'Y_test'+'.pkl')
    
    self.test_labels = load_pkl(self.data_path+'/'+'test_labels' +'.pkl')
    
    self.X_train_labelled_data = load_pkl(self.data_path+'/'+'X_train_labelled'+'.pkl')
    self.Y_train_labelled_data = load_pkl(self.data_path+'/'+'Y_train_labelled'+'.pkl')        
    self.train_labelled_labels = load_pkl(self.data_path+'/'+'train_labelled_labels' +'.pkl')


  def get_data_from_type(self, data_type):
    if data_type == "train":
      X_raw_data = self.X_train_data
      Y_raw_data = self.Y_train_data
    elif data_type == "valid":
      X_raw_data = self.X_valid_data
      Y_raw_data = self.Y_valid_data
    elif data_type == "test":
      X_raw_data = self.X_test_data
      Y_raw_data = self.Y_test_data
    elif data_type == "train_labelled":
      X_raw_data = self.X_train_labelled_data
      Y_raw_data = self.Y_train_labelled_data
      print(X_raw_data.shape)
      print(len(X_raw_data[0]))
      print(len(Y_raw_data))
      
    else:
      raise Exception(" [!] Unkown data type %s: %s" % data_type)

    return X_raw_data, Y_raw_data

  def get_Xc(self, data):
    """data is a document...a list of sentences
    	a list of lists
    """
    doc = [word for sent in data for word in sent]
    counts = np.bincount(doc, minlength=self.vocab_size)
    stops_flag = np.array(list(np.ones([self.lda_vocab_size], dtype=np.int32)) +
    			list(np.zeros([self.n_stops], dtype=np.int32)))

    return counts * stops_flag

  def get_L(self, data):
  	"""
  	data is a document...a list of sentences
  	stop words are tagged 1
  	"""
  	return np.array([[0 if x < self.lda_vocab_size else 1 for x in sent] for sent in data])

  def get_length(self, data):
  	"""
  	data is a document...a list of sentences
  	this returns the length of each sentence in data
  	"""
  	return [len(x) for x in data]

  def pad(self, data):
    """
    data is a document...a list of sentences
    this pads all the shortest sentences to 
    match the length of the longer sequence
    """
    lengths = self.get_length(data)
    maxlen = max(lengths)
    n_rows = len(data)
    padded = np.zeros([n_rows, maxlen], dtype=np.int32)

    for i, length in enumerate(lengths):
    	padded[i, :length] = data[i]

    return padded

  def iterator(self, data_type="train"):
    """
    goes over X and Y and divides them into documents
    returns binary_doc, X, Xc, Y, and L in a round robin
    L is obtained from Y
    Xc and binary_doc are obtained from X
    """
    X_raw_data, Y_raw_data = self.get_data_from_type(data_type)

    x_infos = itertools.cycle(([self.pad(X_doc), 
                  self.get_Xc(X_doc)] 
                              for X_doc in X_raw_data if X_doc != []))
    y_infos = itertools.cycle(([self.pad(Y_doc), 
                  self.pad(self.get_L(Y_doc)), 
                  self.get_length(Y_doc), len(Y_doc)] 
                              for Y_doc in Y_raw_data if Y_doc != []))

    return x_infos, y_infos

# reader = IMDBReader(IMDBConfig())
# iterator = reader.iterator()
# for i in range(25000):
#   X, Xc = iterator[0].next()
#   Y, L, lengths, len_y = iterator[1].next()
#   print("i: {}".format(i))
  # print("X: ", X)
  # print("Y: ", Y)
  # print("L: ", L)
  # print("lengths: ", lengths)
  # print("batch size: ", len_y)
