# Relation Networks
Relation Networks are a neural network module which are specialized to learn relations, just as convolutional kernels are specialized to process images. RNs are useful for Visual Question Answering, where they hold state-of-the-art results.

"A simple neural network module for relational reasoning"
Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap
https://arxiv.org/pdf/1706.01427.pdf

My Keras implementation uses the Functional API to define the network.  One generalization to RNs is the use of a selection kernel which picks k distinct random objects from the processed image tensor instead of all n^2 objects.  This allows for a much smaller number of relation vectors when k << n^2.

# CLEVR.py
Visual Question Answering implemented on the CLEVR dataset (https://cs.stanford.edu/people/jcjohns/clevr/).

![Alt text](VQA.png?raw=true "Title")

Above is an example image which could have the following question: "Q: Are there an equal number of large things and metal spheres?"

## Architecture
Images are processed using a CNN, while the questions are processed using an LSTM.  These tensors are then decomposed into objects and fed as input into the RN module.
![Alt text](CLEVR.png?raw=true "Title")

## Experiments
### 10000 Questions / 1000 Images
* Epochs: 10 / Loss: 1.5208 / Accuracy: 0.3411
* Epochs: 20 / Loss: 1.1206 / Accuracy: 0.4130
* Epochs: 30 / Loss: 1.0256 / Accuracy: 0.4275


# Misc
## MNIST.py
Implementation of Relation Networks on MNIST demonstrating a simpler RN architecture.

## RN.py
First working prototype.
