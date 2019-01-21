import time 

import faiss 
import numpy as np 

## toolket of functions used to implement deep-kmeans algorithm 
## and implement dynamic routing 

## subcomponents tested ...
## next steps: test the assignment function
## next next steps: integrate with NMT code

def preprocess_features(npdata, pca=100):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')
    
    # Apply PCA-whitening with Faiss
    print('before pca matrix') 
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    print('before pca train')     
    mat.train(npdata)
    assert mat.is_trained
    print('before pca apply') 
    print(dir(mat)) 
    pw_npdata = mat.apply_py(npdata) 
    
    pcawhiten_mat = mat.apply_py(np.eye(ndim).astype(np.float32))
    
    ## pcawhiten_mat.shape = (ndim, pca_dim) 
    ## linear algebra operation: 
    ## npdata * pcawhiten_mat 
    #print(pcawhiten_mat) 
    #print(pcawhiten_mat.shape) 
    
    # L2 normalization 
    row_sums = np.linalg.norm(pw_npdata, axis=1) 
    pw_npdata = pw_npdata / row_sums[:, np.newaxis] 
    
    return pw_npdata 


"""
Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
""" 
""" 

[step 1]: code up an initial forward pass to obtain all training data for kmeans 
[This is step 2]: code up a label assignment function to build a supervised dataset 
[step 3]: code up the supervised training algorithm with labels 
[step 4]: build the iteration across 1-4 

def cluster_assign(images_lists, dataset):
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
    
    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)
"""

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster

    [Memory for holding input data] 
    
    10 million 512 dim -> 9.5G float16, 19G float, 38G double 
    1 million 512 dim -> 0.95G float16, 1.9G float, 3.8G double 
    
    """
    
    n_data, d = x.shape
    
    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 1
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    
    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


class Kmeans:
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return I, loss
