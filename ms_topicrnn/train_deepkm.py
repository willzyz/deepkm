import numpy as np, tensorflow as tf, cPickle as pkl 
from deepkm_utils import DataIterator 

Xtrain = pkl.load(open('data/imdb/X_train.pkl', 'r')) 

iterator = DataIterator(Xtrain) 

## what's the concrete plan for data processing?

## let's work off the correct dataset

## align the vocabulary 

## so with imdb, we may have to use the vocabulary offered by topic-rnn

## 1. write a few scripts to process imdb unsup or other data into line-by-line text format
##    this will enable use of nmt code to train deep-km model 
##    - all punctuations need to be accounted for, understand how topic-rnn deal with , . 'd 's " '
##    - ensure all tokens are space-segmented
##    - run a unit-test to ensure correctness of implementation

## 2. go through the topic-rnn logic: there might be changes to the actually topic-rnn vocabulary, later we need to transplant trained deep-km model into topic-rnn, this will involve transforming the loaded list of list Xtrain data structures (sample by sample) into numpy arrays, then into the padded and correct format into deep-km model for cluster asignment 

## make a fast sprint 
