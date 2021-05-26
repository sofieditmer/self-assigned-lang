#!/usr/bin/env python
"""
Info: This script takes the preprocessed movie data and uses pretrained GloVe word embeddings to train a CNN model to classify the movie plots by genre.

Parameters:
    (optional) input_file: str <name-of-input-file>, default = "clean_movie_data.csv"
    (optional) test_size: float <size-of-test-split>, default = 0.25
    (optional) n_words: int <size-of-vocabulary>, default = 5000
    (optional) n_dimensions: int <number-of-GloVe-embedding-dimensions>, default = 100
    (optional) n_epochs: int <number-of-training-epochs>, default = 15
    (optional) batch_size: int <size-of-batches>, default = 20
    (optional) l2_value: float <regularization-value>, default = 0.0001
    (optional) train_embeddings: str <train-embeddings-true-or-false>, default = "False"

Usage:
    $ python 1-cnn-moviegenre.py
    
Output:
    - cnn_100d_summary.txt: a summary of the model architecture.
    - cnn_100d_architecture.png: a summary of the model architecture.
    - cnn_100d_15epochs_classification_metrics.txt: classification report.
    - cnn_100d_15epochs_loss_accuracy_history.png: loss/accuracy curves.
    - cnn_100d_15epochs_plot_training_test_accuracies.png: training and validation accuracies.
"""

### DEPENDENCIES ###

# System tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, matplotlib, contextlib
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# Embedding utils
import utils.embedding_utils as embed_utils

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam # optimization algorithms
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2 # regularization

# argparse
import argparse

### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to training data
    ap.add_argument("-i", "--input_file",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Name of input file",
                    default = "clean_movie_data.csv") # default is the clean movie data
    
    # Argument 2: Size of test dataset
    ap.add_argument("-ts", "--test_size",
                    type = float,
                    required = False, # the argument is not required 
                    help = "Define the size of the validation dataset as float, e.g. 0.25",
                    default = 0.25) # default test size is 25%
    
    # Argument 3: Size of vocabulary
    ap.add_argument("-n", "--n_words",
                    type = int,
                    required = False, # the argument is not required 
                    help = "Define the size of the vocabulary, i.e. how many words it should consist of",
                    default = 5000) # default vocab size
    
    # Argument 4: Number of word embedding dimensions
    ap.add_argument("-nd", "--n_dimensions",
                    type = int,
                    required = False, # the argument is not required 
                    help = "Define the number of word embedding dimensions to use. Choose between 50, 100, 200, and 300.",
                    default = 100) # default embedding dimensions 
    
    # Argument 5: Number of epochs to
    ap.add_argument("-e", "--n_epochs",
                    type = int,
                    required = False, # the argument is not required 
                    help = "Define the number of epochs to train the CNN model for",
                    default = 15) # default number of epochs
    
    # Argument 6: Batch size 
    ap.add_argument("-b", "--batch_size",
                    type = int,
                    required = False, # the argument is not required 
                    help = "Define the size of the batches",
                    default = 20) # default batch size
    
    # Argument 7: L2 regularization value
    ap.add_argument("-r", "--regularization_value",
                    type = float,
                    required = False, # the argument is not required 
                    help = "Define l2-regularization value. The smaller, the more regularization.",
                    default = 0.0001) # default l2-regularization strength
    
    # Argument 8: Train embeddings
    ap.add_argument("-te", "--train_embeddings",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Specify whether you wish to train the word embeddings: True or False",
                    default = "False") # word embeddings are not trainable by default
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    input_file = os.path.join("..", "data", args["input_file"])
    test_size = args["test_size"]
    n_words = args["n_words"]
    n_dimensions = args["n_dimensions"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    l2_value = args["regularization_value"]
    train_embeddings = args["train_embeddings"]
                        
    # Create output directory if it does not already exist
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
    
    # Start message
    print("\n[INFO] Initializing the construction of the CNN model...")
    
    # Instantiate the CNN_classifier class
    classifier = Classifier(input_file)
    
    # Load data
    print(f"\n[INFO] Loading '{input_file}'...")
    movie_data = classifier.load_data()
    
    # Create train-test split
    print(f"\n[INFO] Creating test-train split using {test_size} as test-size...")
    X_train, X_test, y_train, y_test, plots, labels = classifier.create_test_train_split(movie_data, test_size)
    
    # Binarize labels
    print("\n[INFO] Binarizing train and test labels...")
    y_train_binarized, y_test_binarized = classifier.binarize_labels(y_train, y_test)
    
    # Tokenize training and test data
    print("\n[INFO] Tokenizing traning and validation data...")
    X_train_tokens, X_test_tokens, vocab_size, tokenizer = classifier.tokenize(X_train, X_test, n_words)
    
    # Pad documents to be of equal length
    print("\n[INFO] Padding training and validation documents to be of equal length...")
    maxlen, X_train_pad, X_test_pad = classifier.add_padding(X_train_tokens, X_test_tokens)
    
    # Create embedding matrix
    print(f"\n[INFO] Defining embedding matrix based on pretrained GloVe word embeddings with dimensions of {n_dimensions}...")
    embedding_matrix = classifier.create_embedding_matrix(n_dimensions, tokenizer)
    
    # Define CNN model architecture
    print(f"\n[INFO] Defining CNN model architecture...")
    model = classifier.define_model_architecture(l2_value, n_dimensions, embedding_matrix, maxlen, vocab_size, train_embeddings)
    
    # Train model
    print(f"\n[INFO] Traning and evaluating CNN model...")
    model_history, classification_metrics, train_accuracy, test_accuracy = classifier.train_and_evaluate_model(model, X_train_pad, y_train_binarized, n_epochs, X_test_pad, y_test_binarized, batch_size, labels, n_dimensions)
    
    # Print classification report to terminal
    print(f"\n[INFO] Below are the classification metrics for the CNN model. These can also be found in 'output' directory: \n \n {classification_metrics}")

    # Plot loss/accuracy history
    print(f"\n[INFO] Plotting performance of model and saving to 'output' directory...")
    classifier.plot_model_history(model_history, n_epochs, n_dimensions)
    
    # Plot traning and validation accuracies
    print(f"\n[INFO] Plotting training and validaiton accuracies and saving to 'output' directory...")
    classifier.plot_train_test_accuracy(labels, n_dimensions, train_accuracy, test_accuracy, n_epochs)

    # User message
    print("\n[INFO] Done! Results can be found in the 'output' folder.\n")
    
    
# Creating CNN classifier
class Classifier:
    
    def __init__(self, input_file):
        
        # Receive input
        self.input_file = input_file
   
    
    def load_data(self):
        """
        This method loads the input data.
        """
        # Load data into dataframe
        movie_data = pd.read_csv(self.input_file)
        
        return movie_data
    
    
    def create_test_train_split(self, movie_data, test_size):
        """
        This method creates the training and validation datasets.
        """
        # Define data
        plots = movie_data['Plot'].values
        
        # Define labels
        labels = movie_data['Genre_clean'].values
        
        # Create train and test split using sklearn
        X_train, X_test, y_train, y_test = train_test_split(plots, 
                                                            labels, 
                                                            test_size=test_size, 
                                                            random_state=42)
        
        return X_train, X_test, y_train, y_test, plots, labels
        
        
    def binarize_labels(self, y_train, y_test):
        """
        This method binarizes the labels.
        """
        # Intialize the binarizdr
        lb = LabelBinarizer()
        
        # Binarizer training labels
        y_train_binarized = lb.fit_transform(y_train)
        
        # Binarize the test labels
        y_test_binarized = lb.fit_transform(y_test)
        
        return y_train_binarized, y_test_binarized
    
    
    def tokenize(self, X_train, X_test, n_words):
        """
        This method tokenizes the training and validation data using tf.keras.Tokenizer() to create word embeddings. 
        """
        # Initialize tokenizer
        tokenizer = Tokenizer(num_words=n_words)
        
        # Fit tokenizer to training data
        tokenizer.fit_on_texts(X_train)
        
        # Create sequences of tokens for both training and test data
        X_train_tokens = tokenizer.texts_to_sequences(X_train)
        X_test_tokens = tokenizer.texts_to_sequences(X_test)
        
        # Calcuate overall vocabulary size
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        
        return X_train_tokens, X_test_tokens, vocab_size, tokenizer
    
    
    def add_padding(self, X_train_tokens, X_test_tokens):
        """
        This methods pads the tokenized documents to make sure that they are of equal length. 
        Rather than setting an arbitrary maximum length, padding is done by first computing the maximum length of all 
        documents and then adding 0s to all documents to match the maximum length. 
        Computing the maxlen rather than setting an arbitrary value ensures that we actually consider the data in question. 
        """
        # Find the maximum length of a document
        maxlen = max(len(max(X_train_tokens)), len(max(X_test_tokens, key=len))) 
        
        # Pad the training documents to make them all the same length (maxlen)
        X_train_pad = pad_sequences(X_train_tokens, 
                                    padding='post', # post = puts the pads at end of the sequence. Sequences can be padded "pre" or "post"
           
                                    maxlen=maxlen)
        
        # Pad the testing document to make them all the same length (maxlen)
        X_test_pad = pad_sequences(X_test_tokens, 
                                   padding='post', 
                                   maxlen=maxlen)
        
        return maxlen, X_train_pad, X_test_pad
        
        
    def create_embedding_matrix(self, n_dimensions, tokenizer):
        """
        This method creates the embedding matrix based on the pretrained GloVe embeddings. 
        The word embedding dimensions used can be adjusted by the user. The default size is 100 dimensions. 
        """
        embedding_matrix = embed_utils.create_embedding_matrix(os.path.join("..", "glove", f"glove.6B.{n_dimensions}d.txt"),
                                                               tokenizer.word_index, # keeping only the words that appear in the  data
                                                               embedding_dim=n_dimensions)
        return embedding_matrix
   
    
    def define_model_architecture(self, l2_value, n_dimensions, embedding_matrix, maxlen, vocab_size, train_embeddings):
        """
        This method defines the CNN model architecture.
        """
        # Clearing sessions (making sure that I am not retraining a previous model)
        tf.keras.backend.clear_session()
        
        # Define l2 regularization
        l2 = L2(l2_value)
        
        # Define model
        model = Sequential()
        
        # If the user has specified do train word embeddings
        if train_embeddings != "False":
            
            # Add embedding layer and make word embeddings trainable
            model.add(Embedding(vocab_size,                  # vocab size defined by Tokenizer()
                                n_dimensions,                # embedding input layer size
                                weights=[embedding_matrix],  # pretrained embeddings
                                input_length=maxlen,         # maxlen of padded doc
                                trainable=True))             # embeddings are trainable
        else:
            
            # Add embedding layer in which word embeddings are not trainable
            model.add(Embedding(vocab_size,                  # vocab size defined by Tokenizer()
                                n_dimensions,                # embedding input layer size
                                weights=[embedding_matrix],  # pretrained embeddings
                                input_length=maxlen,         # maxlen of padded doc
                                trainable=False))            # embeddings are not trainable
        
        # Add convolutional layer
        model.add(Conv1D(128, 5, 
                         activation='relu',
                         kernel_regularizer=l2)) # L2 regularization 
        
        # Add maxpooling layer
        model.add(GlobalMaxPool1D())
        
        # Add dense layer
        model.add(Dense(10, 
                        activation='relu', 
                        kernel_regularizer=l2))
        
        # Add output layer: 5 movie genres to predict 
        model.add(Dense(5, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        
        # Save model summary
        output_path_1 = os.path.join("..", "output", f"cnn_{n_dimensions}d_summary.txt")
        with open(output_path_1, 'w') as f:
            with redirect_stdout(f):
                model.summary()
    
        # Plot model architecture with TensorFlow and save to output directory
        output_path_2 = os.path.join("..", "output", f"cnn_{n_dimensions}d_architecture.png")
        plot_LeNet_model = plot_model(model,
                                      to_file = output_path_2,
                                      show_shapes=True,
                                      show_layer_names=True)
    
        return model
    
    
    def train_and_evaluate_model(self, model, X_train_pad, y_train_binarized, n_epochs, X_test_pad, y_test_binarized, batch_size, labels, n_dimensions):
        """
        This method fits the CNN model to the training data.
        """
        # Fit model
        model_history = model.fit(X_train_pad, y_train_binarized,
                                  epochs=n_epochs,
                                  verbose=True, # show progress bars
                                  validation_data=(X_test_pad, y_test_binarized),
                                  batch_size=batch_size)
        
        # Extract predictions
        y_predictions = model.predict(X_test_pad, batch_size=batch_size)
        
        # Extract classification report
        classification_metrics = classification_report(y_test_binarized.argmax(axis=1),
                                                       y_predictions.argmax(axis=1),
                                                       target_names=list(set(labels))) # list unique genres and use as labels
        
        # Extract training accuracy and loss
        train_loss, train_accuracy = model.evaluate(X_train_pad, y_train_binarized, verbose=False)
        train_accuracy = round(train_accuracy, 3) # round to 3 decimals
        
        # Extract testing accuracy and loss
        test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_binarized, verbose=False)
        test_accuracy = round(test_accuracy, 3)
        
        # Save classification report to output directory
        output_path = os.path.join("..", "output", f"cnn_{n_dimensions}d_{n_epochs}epochs_classification_metrics.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Below are the classification metrics for the trained model. Number of epochs = {n_epochs}.\n Training Accuracy: {train_accuracy} \n Testing accuracy: {test_accuracy} \n \n {classification_metrics}")
        
        return model_history, classification_metrics, train_accuracy, test_accuracy

    
    def plot_model_history(self, model_history, n_epochs, n_dimensions):
        """
        This method plots the loss/accuracy curves of the model and saves plot to 'output' directory.
        """
        # Specify path
        plot_out_path = os.path.join("..", "output", f"cnn_{n_dimensions}d_{n_epochs}epochs_loss_accuracy_history.png")
        
        # Plot model history and save to output directory
        plt = embed_utils.plot_history(model_history, n_epochs, plot_out_path)
        
    
    def plot_train_test_accuracy(self, labels, n_dimensions, train_accuracy, test_accuracy, n_epochs):
        """
        This method plots the training and validaiton accuracies for each movie genre in a table plot.
        """
        # Define genre labels
        unique_labels = list(set(labels))
        
        # Define plot
        plt.figure(figsize=(20, 10))                                       # plot size
        p1 = plt.bar(unique_labels, height = train_accuracy)               # training accuracy
        p2 = plt.bar(unique_labels, height = test_accuracy)                # test accuracy
        plt.title("Movies Genre Classification Accuracy")                  # plot title
        plt.ylabel('Accuracy', fontsize=20)                                # y-axis label
        plt.xlabel('Movie Genres', fontsize=20)                            # x-axis label
        plt.legend((p1[0], p2[0]), ('Training accuracy', 'Test Accuracy')) # plot legend
      
        # Save plot
        plt.savefig(os.path.join("..", "output", f"cnn_{n_dimensions}d_{n_epochs}epochs_plot_training_test_accuracies.png"))
        
        
# Define behaviour when called from command line
if __name__=="__main__":
    main()