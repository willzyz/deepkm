import sys 
import conviz 
import numpy as np 

sys.path.insert(1, './') 

weights = np.random.randn(16, 16, 3, 32) 
name = 'test' 

conviz.plot_conv_weights(weights, name, channels_all=True)
