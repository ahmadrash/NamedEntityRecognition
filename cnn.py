from __future__ import print_function

import sys
import os
import time
import string

import numpy as np
import theano
import theano.tensor as T

import lasagne
import data_process 

#Model Parameters

NUM_EPOCHS = 200
WINDOW = data_process.WINDOW
WORDVEC_LENGTH = data_process.WORDVEC_LENGTH
L1_X = 2
L1_Y = 50
L2_X = 2
L2_Y = 2
L3_X = 1 
L3_Y = 25
L4_X = 1
L4_Y = 2
L5_DROPOUT = 0.5
L6_DROPOUT = 0

def build_cnn(input_var=None):
  
    network = lasagne.layers.InputLayer(shape=(None, 1, WINDOW, WORDVEC_LENGTH),
                                        input_var=input_var)
    lasagne.layers.get_output_shape(network)

    # Convolutional layer with 4 kernels of size 2x50. Strided and padded
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=4, filter_size=(L1_X, L1_Y),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    lasagne.layers.get_output_shape(network)


    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(L2_X, L2_Y))
    lasagne.layers.get_output_shape(network)

    # Another convolution with 8 1x21 kernels, and another 1x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=8, filter_size=(L3_X, L3_Y),
            nonlinearity=lasagne.nonlinearities.rectify)
    lasagne.layers.get_output_shape(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(L4_X, L4_Y))
    lasagne.layers.get_output_shape(network)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=L5_DROPOUT),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 9-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=L6_DROPOUT),
            num_units=9,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]




def main(num_epochs=NUM_EPOCHS):
    # Load the dataset
    print("Loading data...")
    X_1, y_train, X_2, y_test,wordVec,label_set = data_process.getData()

    X_train = np.reshape(X_1,(len(X_1)/WINDOW,WINDOW,WORDVEC_LENGTH))
    X_train = X_train[:,np.newaxis,:,:]
    X_test = np.reshape(X_2,(len(X_2)/WINDOW,WINDOW,WORDVEC_LENGTH))
    X_test = X_test[:,np.newaxis,:,:]
    y_train = y_train[1:len(y_train):WINDOW]
    y_test = y_test[1:len(y_test):WINDOW]
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)
   

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # Create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    test_fn = theano.function([input_var],[test_prediction])

    # Launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, NUM_EPOCHS, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, NUM_EPOCHS, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    response = ''
    while True:

        response = raw_input('Type a query(type e to exit): ')

        if response == 'e':
            sys.exit()
            
        x_q = data_process.getMatrix(response,wordVec)
        X_query= np.reshape(x_q,(len(x_q)/WINDOW,WINDOW,WORDVEC_LENGTH))
        X_query = X_query[:,np.newaxis,:,:]
        result = np.argmax((test_fn(X_query))[0],axis = 1)
        label = data_process.decodeLabel(list(result),list(label_set))

        for i in range(len(label)):
            print(response.split()[i] + '\t' + label[i]) 
            
    
if __name__ == '__main__':
    

        main()
       
        
        
        
