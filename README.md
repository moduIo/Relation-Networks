# Relation Networks
Relation Networks are a neural network module which are specialized to learn relations, just as convolutional kernels are specialized to process images. RNs are useful for Visual Question Answering, where they hold state-of-the-art results.

"A simple neural network module for relational reasoning"
Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap
https://arxiv.org/pdf/1706.01427.pdf

# CLEVR.py
Visual Question Answering as in https://arxiv.org/pdf/1706.01427.pdf implemented on the CLEVR dataset (https://cs.stanford.edu/people/jcjohns/clevr/).

![Alt text](VQA.jpg?raw=true "Title")
Above is an example image which could have the following question: "Q: Are there an equal number of large things and metal spheres?"

The image is processed using a CNN, while the question is processed using an LSTM.  These tensors are then decomposed into objects and fed as input into the RN module.

# MNIST.py
Implementation of Relation Networks on MNIST.

# RN.py
First working prototype.
