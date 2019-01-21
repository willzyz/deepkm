import kmtools 
import numpy as np 

data = np.random.randn(10000, 512).astype(np.float32)

## forward propagate the data 

k = 10 

KmClass = kmtools.Kmeans(k)
L = KmClass.cluster(data, verbose=False)

print(L)

numC = len(KmClass.images_lists)
print(numC)

for i in range(numC):
    print(len(KmClass.images_lists[i]))

