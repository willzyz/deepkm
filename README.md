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

## Plan for experimental tasks: 

## A. implement the k-means algorithm with LSTM encoder [ half a day to 1 day, delayed: 1 day 3 hours running, due to understand code-base and devise a design for engineering codes as well as data] 

## Algorithm logic for performing k-means is as follows: 

## 1. forward propagate with encoder, a large sub-sample of data into latent space [2 - 3 hours] [ details: need to align the data specs coming out of encoder, feeding into kmeans ] 
##     - build an iterator inside topic-rnn using pickle loaded data 
##     - build a custom lstm encoder borrowing the seq2seq logic 

## 2. k-means clustering with no carry-forward initialization [ implemented ] 

## 3. fine-tune the encoder with NCA objective using cluster [ 2 - 3 hours ] 

## 4. iterate to 1. [ combine, test, and iterate on the algorithm, 2 hours ] 

## 5. visualize and run experiments [ 3 - 4 hours ] 
##    the kmeans algorithm should need to run for 1-3 days to converge 

## B. implement the semi-supervised dynamic routing algorithm for topic-rnn on imdb [delayed 3 hours] 

## 1. implement model: [ 3 - 4 hours ] 
##    use deepkmeans to assign the minibatch data, the labels of assignment 
##    is used to multiply or perform other tensor operations with topic-rnn 
##    classification logic, to select different softmax parameters for classification 
##    this can be compared with topic rnn which compensates softmax weights with
##    added 'b' parameter chosen using a 'topic model' which is actually a variational
##    auto-encoder with some context 

## 2. perform experiments and see how much better perplexity is 
##    note: experiments run for a while - on mac-book pro for 3-5 days, on gpu 1.5-3 days

## 3. to make experiments faster, we can keep the bottom layer fixed and only train the 
##    softmax function using model already trained previously [4 - 5 hours] 

