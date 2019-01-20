import numpy as np, tensorflow as tf, cPickle as pkl 
from deepkm_utils import DataIterator 

## previous test scripts for training deep-kmeans 
## pivoted to nmt-deepkm/nmt/train.py 

Xtrain = pkl.load(open('data/imdb/X_train.pkl', 'r')) 

iterator = DataIterator(Xtrain) 
