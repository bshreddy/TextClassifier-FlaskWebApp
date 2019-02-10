"""
    Movie Review Classifier using Keras - Flask Web App

    Neural Network Model File
      - This is the NN Model trained on IMDb Movie Review Database using Keras and 
        Tensorflow as backend as a simple text classifier
      - 1x Embedding Layer (Shape: 16)
      - 1x GlobalAveragePooling1D Layer (Shape: 16)
      - 1x Dense Layer with 16 nodes with ReLu activation function (Shape: 16)
      - 1x Dense Layer with 1 node with Sigmoid activation function to generate output (Shape: 1)

    Author: Sai Hemanth Bheemreddy
"""

import tensorflow as tf
from tensorflow import keras

# Loading IMDb Dataset
imdb = keras.datasets.imdb

# Getting most common words
word_index = imdb.get_word_index()

# Creating words dict
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

def getRating(review):
    """
    Gets rating from review in [0, 1] which indicated ["Bad", "Good"]

    review - String which contains movie review written by user

    returns rating for a given review
    """

    # Load pretrained model from .h5 file and compiling it
    model = keras.models.load_model("TextClassifier.h5")
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
    review_index = []

    result = None

    try:
        # Preprocessing review for input to model
        for word in review.split(' '):
            try:
                temp = word_index[word.lower()]
                if word in word_index.keys() and temp < 10000:
                    review_index.append(temp)
                elif temp >= 10000:
                    review_index.append(word_index["<UNUSED>"])
                else:
                    raise ValueError('Invalid Word')
            except:
                review_index.append(word_index["<UNK>"])
        
        # Stores the vectorized words converted from "review"
        review_index = keras.preprocessing.sequence.pad_sequences([review_index], value=0, padding='post', maxlen=256)

        # Estimating the rating
        result = model.predict(review_index)[0][0]
    except:
        pass
    
    # Reset Computational Graph
    keras.backend.clear_session()
    
    return result