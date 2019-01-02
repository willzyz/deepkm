import os
import cPickle
import numpy as np
import tensorflow as tf
import scipy

def nearest_neighbor(embedding_matrix, word, N):
  """computes the N nearest neighbors of word in embedding_matrix"""
  vocab_size = embedding_matrix.shape[0] #B is V x K
  print("vocab: ", vocab_size)
  dists = []
  for i in range(vocab_size):
    dist = scipy.spatial.distance.cosine(word, embedding_matrix[i, :])
    dists.append(dist)

  arr = np.array(dists) 
  nearest = arr.argsort()[-N:][::-1]

  return nearest

def get_checkpoint_dir(model):
  model_dir = ""#model.dataset
  for attr in model._attrs:
    if hasattr(model, attr):
      model_dir += "/%s=%s" % (attr, getattr(model, attr))

  checkpoint_dir = model.checkpoint_dir + model_dir + "/"

  return checkpoint_dir

def load_session(model):
  print(" [*] Loading checkpoints...")
  checkpoint_dir = get_checkpoint_dir(model)
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    model.saver.restore(model.sess, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Load SUCCESS")
    return True
  else:
    print(" [!] Load failed...")
    return False

def save_session(model, step):
  print(" [*] Saving checkpoints...")
  checkpoint_dir = get_checkpoint_dir(model)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  model.saver.save(model.sess, checkpoint_dir, global_step=step)

def save_pkl(path, obj):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print(" [*] save %s" % path)

def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print(" [*] load %s" % path)
    return obj

def save_npy(path, obj):
  np.save(path, obj)
  print(" [*] save %s" % path)

def load_npy(path):
  obj = np.load(path)
  print(" [*] load %s" % path)
  return obj

def cross_entropy_loss(y, yhat, vocab_size):
  y = tf.one_hot(y, vocab_size)
  res = -tf.multiply(y, tf.log(yhat + 1e-10))
  res = tf.reduce_sum(res, reduction_indices=1)

  return res 

def accuracy():
	pass

