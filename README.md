##  Bachelor Thesis in Cognitive Science

**Tilte:** Methods for Analyzing the Internal Representations of Convolutional Neural Networks

**University:** Unversität Osnabrück 

**Primary supervisior:** Hristofor Lukanov, M. Sc.

**Second Supervisior:** Prof Dr. Gordon Pipa


## Content

In this thesis we investigate the representation of two convolutional neural networks. Due to their special structure convolutional neural networks learn a feature representation hierarchy.
The representations in the initial layers are rather simple and increase in complexity from layer to layer, resulting in high-level concepts in the upper layers. The two networks under investigation were trained with an identical architecture but different label granularities.  The use of different labels from the same semantic hierarchy results in the emergence of different representations. The goal of this thesis is to identify these differences and to examine the representations of the two convolutional neural networks in terms of their abstractness, complexity and invariance using different analysis methods. First of all, an overview of the general principles of deep learing focus is on convolutional neural networks and their representations is given. In the second part the modelling and analysis techniques are explained and xpectations regarding the differences in the representations of the two models resulting from label granularity are defined.  In the last part the results of the methods are compared with the expectations and put into a scientific context.

## Implementation


[![Build Status](https://api.travis-ci.com/JonaLimp/Ba_Thesis.svg?token=3x3tywg5VyvqsCs7YyK5&branch=master)](https://travis-ci.com/JonaLimp/Ba_Thesis)

The Implementation is written in python. For the training of the CNNs I used tensorflow and keras. Source Code can be found in the Code folder.
