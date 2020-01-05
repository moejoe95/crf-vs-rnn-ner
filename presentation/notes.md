# Presentation notes

## CRF

Formula (1) shows how a CRF is modeled. We want to compute the conditional probability of Y given X, where X is the input sample, which is a word in a sentence in our case, and Y is a member of the predefined categories of named entities. For each category we calculate the probability of the word X belonging to this category.

The main idea is to define so called feature functions in a way that expresses some characteristic that is present in the training data and we want our model to hold. The feature functions are indicated by f_k in formula (1). y_t is the current label, y_{t-1} the previous label, and x_t are input samples. Theta is the parameter vector, whichs values have to be learning in training. 
The outputs of the feature functions multiplied by the parameter vector are summed up. This is computed for every timestep and normalized to be between 0 and 1. The result is then a probability distribution for each label for each sample. You take the maximum then and you have the prediction.

## RNNs 

### Vanishing gradient problem

Downside of RNNs is the inability to deal with long term relationships, which is important in natural language processing. The problem that arises is called *vanishing (and exploding) gradient*. The more timesteps we have, the higher is the chance that the gradient of back-propagation explodes or vanishes down. If we want to calculate the relationship with an input three steps before, we have to multiply the error gradients of each layer, wich is typically a sigmoid function. Sigmoids return values between 0 and 1. If the outputs are all close to 0, the gradient will be almost 0, which is called vanishing gradient. If the gradient is 0, the weights will not be updated and we have no long term relationship.

Exploding gradient happens when the ouputs of the sigmoid function all return close to 1. Yellow nodes indicate a hidden layer, red and green nodes indicate a element-wise operation.

## LSTMs

LSTMs hold an internal *memory* state, which is **added** to the process input. This reduced the effect of the multiplication with the gradient. The time dependence is determined by the *forget* gate.

The data in this picture flows from left to right. x_t is the word at step t in our case. h_{t-1} is the output of the last LSTM cell.

The input is squashed between -1 and 1 by a hyperbolic tanget (tanh) function. The squashed input is then multiplied element wise with the output of the *input* gate. The *input* gate is sigmoid function, squashing between 0 and 1, this can turn on and off parts of the input.

Next step is the *forget* gate. The new variable *s* represents the internal state of the LSTM cell. The interal state is delayed by one step and added to the output of the input gate. The sigmoid function determines which words can be forgotten (close to 0), and which words need to be remembered (output close to 1).


Last step is the *output* gate, which operates in the same way as the input gate. The output is squashed between -1 and 1 by the hyperbolic tangent function, and the output gate with the sigmoid function determines which values to output.


## Features from unsupervised ML algorithms

### brown clusters

Brown clusters are hierarchical clustering algorithms that work based on distributional information. The intuition is that similar words appear in similar context, or more precisely, similar words have similar distributions of words to ther left and right.