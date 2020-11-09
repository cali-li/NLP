import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ParserModel(nn.Module):


    def __init__(self, config, word_embeddings=None, pos_embeddings=None,
                 dep_embeddings=None):
        super(ParserModel, self).__init__()

        self.config = config
        
        # These are the hyper-parameters for choosing how many embeddings to
        # encode in the input layer.  See the last paragraph of 3.1
        n_w = config.word_features_types # 18
        n_p = config.pos_features_types # 18
        n_d = config.dep_features_types # 12

        
        # Copy the Embedding data that we'll be using in the model.  Note that the
        # model gets these in the constructor so that the embeddings can come
        # from anywhere (the model is agnostic to the source of the embeddings).
        #self.word_embeddings = None 
        #self.pos_embeddings = None # TODO
        #self.dep_embeddings = None
        
        self.word_embeddings = word_embeddings
        self.pos_embeddings = pos_embeddings
        self.dep_embeddings = dep_embeddings
        
        embedding_dim = 50
        l1_hidden_size = 200
        l2_hidden_size = 15
        NUM_DEPS = 42
        num_classes = NUM_DEPS * 2 + 1
        batch_size = 2048
        keep_prob = 0.5
        reg_val = 1e-8
        lr = 0.001

        # Create the first layer of the network that transform the input data
        # (consisting of embeddings of words, their corresponding POS tags, and
        # the arc labels) to the hidden layer raw outputs.
        
        ############################################number of dim##############
        ########################### GLOBAL VAR???    TODO######################
        self.hidden = nn.Linear((n_w+n_p+n_d)*embedding_dim, l1_hidden_size)
        #self.hidden = nn.Linear(n_w+n_p+n_d, l1_hidden_size)# first layer
        #############################################TODO######################
        

        
        # TODO
        
        # After the activation of the hidden layer, you'll be randomly zero-ing
        # out a percentage of the activations, which is a process known as
        # "Dropout".  Dropout helps the model avoid looking at the activation of
        # one particular neuron and be more robust.  (In essence, dropout is
        # turning the one network into an *ensemble* of networks).  Create a
        # Dropout layer here that we'll use later in the forward() call.
        self.drop = nn.Dropout(keep_prob)
        # TODO
                
        # Create the output layer that maps the activation of the hidden layer to
        # the output classes (i.e., the valid transitions)
        self.predict = nn.Linear(l1_hidden_size, num_classes)   # output layer
        #output = F.softmax(F.linear(h1, W2))
        # TODO

        # Initialize the weights of both layers
        self.init_weights()
        
        
        
        
        #return output
        
    def init_weights(self):
        # initialize each layer's weights to be uniformly distributed within this
        # range of +/-initrange.  This initialization ensures the weights have something to
        # start with for computing gradient descent and generally leads to
        # faster convergence.
        initrange = 0.1
        self.hidden.weight.data.uniform_(initrange*(-1), initrange)
        if self.hidden.bias is not None:
            self.hidden.bias.data.uniform_(initrange*(-1), initrange)
        self.predict.weight.data.uniform_(initrange*(-1), initrange)
        if self.predict.bias is not None:
            self.predict.bias.data.uniform_(initrange*(-1), initrange)
        #self.hidden.weight = torch.FloatTensor(np.random.uniform(initrange*(-1), initrange))
        #self.predict.weight = torch.FloatTensor(np.random.uniform(initrange*(-1), initrange))     

        
    def lookup_embeddings(self, word_indices, pos_indices, dep_indices, keep_pos = 1):
        
        # Based on the IDs, look up the embeddings for each thing we need.  Note
        # that the indices are a list of which embeddings need to be returned.
        w_embeddings = self.word_embeddings(word_indices)
        p_embeddings = self.pos_embeddings(pos_indices)
        d_embeddings = self.dep_embeddings(dep_indices)

        # TODO
        
        return w_embeddings, p_embeddings, d_embeddings

    def forward(self, word_indices, pos_indices, dep_indices):
        """
        Computes the next transition step (shift, reduce-left, reduce-right)
        based on the current state of the input.
        

        The indices here represent the words/pos/dependencies in the current
        context, which we'll need to turn into vectors.
        """
        w_embeddings, p_embeddings, d_embeddings = self.lookup_embeddings(word_indices, pos_indices, dep_indices)
        #w_embeddings.view(batch_size,-1)
        #p_embeddings.view(batch_size,-1)
        #d_embeddings.view(batch_size,-1)
        self.joint_embeddings = torch.cat([w_embeddings, p_embeddings, d_embeddings], dim=1)
        
        # Look up the embeddings for this prediction.  Note that word_indices is
        # the list of certain words currently on the stack and buffer, rather
        # than a single word
        
        # TODO

        # Since we're converting lists of indices, we're getting a matrix back
        # out (each index becomes a vector).  We need to turn these into
        # single-dimensional vector (Flatten each of the embeddings into a
        # single dimension).  Note that the first dimension is the batch.  For
        # example, if we have a batch size of 2, 3 words per context, and 5
        # dimensions per embedding, word_embeddings should be tensor with size
        # (2,3,5).  We need it to be a tensor with size (2,15), which makes the
        # input just like that flat input vector you see in the network diagram.
        #
        # HINT: you don't need to copy data here, only reshape the tensor.
        # Functions like "view" (similar to numpy's "reshape" function will be
        # useful here.        

        # TODO
        
        # Compute the raw hidden layer activations from the concatentated input
        # embeddings.
        #
        # NOTE: if you're attempting the optional parts where you want to
        # compute separate weight matrices for each type of input, you'll need
        # do this step for each one!
        W1 = self.hidden.weight
        b1 = self.hidden.bias
        #h1 = torch.pow(torch.matmul(self.joint_embeddings, torch.t(W1))+b1, 3)
        h1 = torch.pow(torch.matmul(self.joint_embeddings.view(word_indices.size(0), -1), torch.t(W1))+b1, 3)
        h1 = self.drop(h1)
        

        # TODO
        
        
        # Compute the cubic activation function here.
        #
        # NOTE: Pytorch doesn't have a cubic activation function in the library

        # TODO
        

        # Now do dropout for final activations of the first hidden layer

        # TODO

        # Multiply the activation of the first hidden layer by the weights of
        # the second hidden layer and pass that through a ReLU non-linearity for
        # the final output activations.
        #
        # NOTE 1: this output does not need to be pushed through a softmax if
        # you're going to evaluate the output using the CrossEntropy loss
        # function, which will compute the softmax intrinsically as a part of
        # its optimization when computing the loss.
        W2 = self.predict.weight
        b2 = self.predict.bias
        output = F.relu(torch.matmul(h1, torch.t(W2))+b2)

        # TODO

        return output    

