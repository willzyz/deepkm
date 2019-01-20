## Deep K-means unsupervised learning for dynamic routing 

Repository for code in deep dynamic routing with unsupervised learning. 

## [Dependencies]:

Python 2.7 
FAISS: facebook library for clustering
Tensorflow: LSTM implementations

## [Algorithms]:

# DeepKM code logic (nmt-deepkm/nmt/train.py) 

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

## [Progress]: 

We organized unsup train data (which includes unsup director, and train_labeled in imdb). To align with NMT code, we use a vocabulary with 3 added symbols and .

First deepkm pseudo code implemented in train.py. We can now focus on implementing detailed logic in model_helper.py, train.py, iterator_utils.py.

## Data Processing:

Align the vocabulary, with imdb, we may have to use the vocabulary offered by topic-rnn

1. Wrote scripts in folder ms_topicrnn/scrips to process imdb unsup or other data into line-by-line text format
   this will enable use of nmt code to train deep-km model 
   - all punctuations need to be accounted for, understand how topic-rnn deal with , . 'd 's " '
   - ensure all tokens are space-segmented
   - run a unit-test to ensure correctness of implementation

2. [todo] go through the topic-rnn logic: there might be changes to the actually topic-rnn vocabulary, later we need to transplant trained deep-km model into topic-rnn, this will involve transforming the loaded list of list Xtrain data structures (sample by sample) into numpy arrays, then into the padded and correct format into deep-km model for cluster asignment 


