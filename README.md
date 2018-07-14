# Relation Networks
Relation Networks are a neural network module which are specialized to learn relations, just as convolutional kernels are specialized to process images. RNs are useful for Visual Question Answering, where they hold state-of-the-art results.

"A simple neural network module for relational reasoning"
Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap
https://arxiv.org/pdf/1706.01427.pdf

My Keras implementation uses the Functional API to define the network.  One modification to RNs is the use of a selection kernel which picks k random objects from the processed image tensor instead of all n^2 objects.  This allows for a much smaller number of relation vectors when k << n^2 and speeds up the network.

# CLEVR.py
![Alt text](VQA.png?raw=true "Title")

Visual Question Answering as in https://arxiv.org/pdf/1706.01427.pdf implemented on the CLEVR dataset (https://cs.stanford.edu/people/jcjohns/clevr/).

Above is an example image which could have the following question: "Q: Are there an equal number of large things and metal spheres?"

Images are processed using a CNN, while the questions are processed using an LSTM.  These tensors are then decomposed into objects and fed as input into the RN module.

## Accuracy
Epoch 10 - loss: 1.5208 - acc: 0.3411

# MNIST.py
Implementation of Relation Networks on MNIST.

# RN.py
First working prototype.