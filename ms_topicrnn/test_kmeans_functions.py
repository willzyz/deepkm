import kmtools 
import numpy as np 

## Experimental tasks: 

## A. implement the k-means algorithm with LSTM encoder [ half a day to 1 day] 

## Algorithm logic for performing k-means is as follows: 

## 1. forward propagate with encoder, a large sub-sample of data into latent space [2 - 3 hours] [ details: need to align the data specs coming out of encoder, feeding into kmeans ] 

## 2. k-means clustering with no carry-forward initialization [ implemented ] 

## 3. fine-tune the encoder with NCA objective using cluster [ 2 - 3 hours ]

## 4. iterate to 1. [ combine, test, and iterate on the algorithm, 2 hours ] 

## 5. visualize and run experiments [ 3 - 4 hours ] 
##    the kmeans algorithm should need to run for 1-3 days to converge 

## B. implement the semi-supervised dynamic routing algorithm for topic-rnn on imdb 

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

