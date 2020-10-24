# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:38:08 2020

@author: Cornelis de Jager - n8891974
"""
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#   import libraries
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import backend as K

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#   Public Functions
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
@tf.function
def triplet_loss(label, embedding, margin = 0.4):
    '''
    Triplet loss function
           
    
    @param a: Anchor vector
    @param p: Positive vector
    @param n: Negetive vector
    @param m: Margin
    
    '''
    a = embedding[0]
    p = embedding[1]
    n = embedding[2]
        
    print(a)
    print(p)
    print(n)
    
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(a - p),axis=1)
    
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(a - n),axis=1)
    
    # compute loss
    basic_loss = pos_dist-neg_dist + margin
    
    loss = tf.maximum(basic_loss, 0.0)
    loss = tf.reduce_mean(loss)     
    
    return loss

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
@tf.function
def contrastive_loss(label, embedding, margin = 0.4):
    '''
    contrastive_loss function
    
    @param p: Positive vector
    @param n: Negetive vector
    
    @returns (float): y - Integer value representing distance
    '''
    # Assign the label
    y = label
    
    # Assign the embeddings
    p1 = embedding[0]
    p2 = embedding[1]
    
    # Get the euclean distance    
    d = tf.norm(p1 - p2, axis=-1)
    
    
    if y == 0:
        return (1/2) * tf.math.sqrt(d)
    else:
        return (1/2) * tf.math.sqrt(tf.math.maximum(0.0, (margin-d)))
    
   
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
def conv_net (ds_info, batch_size = 128, epochs = 12):
    '''
    Create a cnn 
    
    @param x    
    @param y    
    @param ds_info    
    @optional loss    
    @optional batch    
    @optional opochs
    
    '''
    
      # Get the input shape
    image_shape = ds_info.features['image'].shape
    
    model = keras.Sequential(
    [
        Conv2D(32, 3, activation='relu', input_shape=image_shape),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(
            128, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l1(0.01)
            ),
        Dense(ds_info.features['label'].num_classes, activation='softmax')
    ])
    
    
    return model


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
def siamese_model(ds_info, model):
    """
    Define the convolutional network and resulting feature outputs for each glyph in a pair.
    Calculate the absolute difference between them and output a similarity score.
    Arguments:
    input_shape --  shape of the image arrays to be convolved
    Returns:
    siamese_out --  similarity prediction between 0 and 1
    """

    # get the shape from the ds_info
    input_shape =  ds_info.features['image'].shape
    
    # Get the encodings
    encoding_1 = Input(input_shape)
    encoding_2 = Input(input_shape)

    # Define the feature vectors (h(x) of each glyph)
    encoding_1_features = model(encoding_1)
    encoding_2_features = model(encoding_2)
    
    # Find absolute difference between the feature vectors
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoding_1_features, encoding_2_features])
    
    # Final layer generates a similarity score between 0 and 1
    prediction_layer = Dense(1, activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[encoding_1, encoding_2], outputs=prediction_layer)
    
    # Finally return the result
    return siamese_net

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Helper Functions
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def compile_cnn(model, loss = None, optimizer = None):
    '''
    @param model: A kerras model
    @param loss: The loss function to use according to karras standard
    @param optimizer: The optimizer to use.
            
    NOTE: We use Adam over Adadelta as it is a extention of Adadelta 
    
    @returns model: a compiled keras model
    '''

    # Compile the CNN using the specified loss function
    model.compile(loss=loss, optimizer=optimizer)
    
    # return the compiled model
    return model
    

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def UniqueClassIndexes(arr):
    '''
    Convers to unique classes to an array of 0 - num unique classes

    Parameters
    ----------
    arr : numpy array
        Input numpy array.

    Returns
    -------
    output_arr : numpy array
        Output array for the function.

    '''
    unique_list = dict()
    output_arr = []
    index = 0
    
    # We map it to dictionary
    for i in arr:
        if i not in unique_list:
            unique_list[i] = index
            index += 1
    
        i = unique_list[i]
        
        output_arr.append(i)
        
    # now replace the values in list with their index value      
    output = np.array(output_arr)
    return output


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Tasks and Implementations
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

' Load Omniglot dataset ' 
# load the training data
ds, ds_info = tfds.load(name='Omniglot', with_info=True, as_supervised=True)


(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    name = 'kmnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'Specify Testing Values'
epochs = 3 # 3 for testin

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Split the dataset into testing and training '
ds_train, ds_test = ds["train"], ds["test"]

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Tune for performance '

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
ds_test = ds_test.cache().prefetch(buffer_size=AUTOTUNE)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'Create a CNN to help test'
model = conv_net (ds_info)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Test contrastive loss function '
# We use Adam as optimizer
optimizer = keras.optimizers.Adam() 

# Compile the model with the contrastive loss function
cont_loss_model = compile_cnn(model, contrastive_loss, optimizer)

# Fil the results
cont_loss_model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=epochs,
    verbose=1
    )

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Test triplet loss function '
trip_loss_model = compile_cnn(model, triplet_loss, optimizer)

# Fil the results
trip_loss_model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=epochs,
    verbose=1
    )

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Build a siamese network '
## We are going to create two - one for each loss function
# Contrastive Loss
cont_loss_model = siamese_model(ds_info, cont_loss_model)

# Triplet loss
triplet_siamese = siamese_model(ds_info, trip_loss_model)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Traing the siamese network '


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Plot the training and validation error vs time '


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Evaluate performance of network with two different losses by '
# Pairs from training set

# Pairs from two different splits

# Pairs from the test splits

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
' Present findings and Tables and Figures '