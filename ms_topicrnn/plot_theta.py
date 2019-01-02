import matplotlib.pyplot as plt
import numpy as np
import util

seed_text = util.load_pkl('./results/seed_text_gru4.pkl')
print("seed_text: ", seed_text)
theta = util.load_pkl('./results/theta_gru4.pkl')
#theta = np.exp(theta) 
#theta = theta / np.sum(theta)
theta = theta[0]
print("theta: ", theta)
N = len(theta)
x = range(N)
width = 1/2.0
plt.bar(x, theta, width, color="blue")
plt.title("Inferred Topic Distribution from TopicGRU", fontsize=20)
plt.savefig("theta_gru4.pdf")
plt.show()
