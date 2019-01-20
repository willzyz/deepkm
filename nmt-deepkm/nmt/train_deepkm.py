import numpy as np, tensorflow as tf, cPickle as pkl 

Xtrain = pkl.load(open('data/imdb/X_train.pkl', 'r')) 

iterator = DataIterator(Xtrain) 


