#!/usr/bin/env python
"""
This script stores utility functions for creating an embedding matrix and plotting loss/accuracy history of CNN model.
"""

# Dependencies
import numpy as np
import matplotlib.pyplot as plt

# Function that creates an embedding matrix.
def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix. 
    This function was developed for use in class and adjusted for this project.
    
    Input:
        - filepath: path to GloVe embedding
        - word_index: indices from keras Tokenizer
        - embedding_dim: dimensions of keras embedding layer
    """
    # Define vocab size
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    
    # Create emebdding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


# Function that plots the loss/accuracy curves of the model
def plot_history(H, epochs, output_path):
    """
    Utility function for plotting model history using matplotlib. 
    This method was developed for use in class and adjusted for this project. 
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


# Define behaviour when called from command line
if __name__=="__main__":
    None