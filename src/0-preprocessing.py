#!/usr/bin/env python
"""
Info: This script takes the 'wiki_movie_plots.csv' and cleans it for use in '1-cnn-moviegenres.py script'.

Parameters:
    (optional) input_file: str <name-of-input-file>, default = "wiki_movie_plots.csv"
    (optional) output_file: str <name-of-output-file>, default = "clean_movie_data.csv"

Usage:
    $ python 0-preprocessing.py
    
Output:
    - clean_movie_data.csv: preprocessed data. 
"""

### DEPENDENCIES ###

# System tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# argparse
import argparse

### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Name of input file
    ap.add_argument("-i", "--input_file",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Name of input file",
                    default = "wiki_movie_plots.csv") # default input file
    
    # Argument 2: Name of output file
    ap.add_argument("-o", "--output_file",
                    type = str,
                    required = False, # the argument is not required
                    help = "Name of output file",
                    default = "clean_movie_data.csv") # default output filename
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    input_file = os.path.join("..", "data", args["input_file"])
    output_file = os.path.join("..", "data", args["output_file"])
    
    # Create output directory if it does not already exist
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
    
    # Instantiate the CNN_classifier class
    clean = Clean(input_file, output_file)
    
    # Load data
    print(f"\n[INFO] Loading '{input_file}'...")
    movie_data = clean.load_data()
    
    # Clean data
    print(f"\n[INFO] Preprocessing '{input_file}'...")
    clean_movie_data_balanced = clean.clean_data(movie_data)
    
    # Save to data folder
    print(f"\n[INFO] Saving clean and balanced data as '{output_file}' to 'data' folder...")
    clean.save_to_csv(clean_movie_data_balanced)

    # User message
    print("\n[INFO] Done! Preprocessed data can be found as '{output_file}' in the 'data' folder.\n")
    
    
# Creating class 
class Clean:
    
    def __init__(self, input_file, output_file):
        
        # Receive input
        self.input_file = input_file
        self.output_file = output_file
        
        
    def load_data(self):
        """
        This method loads the data.
        """
        # Load data into dataframe
        movie_data = pd.read_csv(self.input_file)
        
        # Select relevant columns
        movie_data = movie_data.loc[:, ("Title", "Genre", "Plot")]
        
        return movie_data
        
        
    def clean_data(self, movie_data):
        """
        This method cleans the loaded data.
        """
        # Filter out movies with unknown genre
        movie_data = movie_data[~movie_data['Genre'].isin(['unknown'])]
        
        # Make new column for clean genre descriptions
        movie_data["Genre_clean"] = movie_data["Genre"] 
        
        # Combine genres "sci-fi" and "science fiction" 
        movie_data["Genre_clean"].replace({"sci-fi": "science fiction"}, inplace=True)
        
        # Combine genres "romantic comedy" and "romance"
        movie_data["Genre_clean"].replace({"romantic comedy": "romance"}, inplace=True)
        
        # Choose the top most frequent genres - the genres that are represented more than 1000 times
        chosen_genres = movie_data["Genre_clean"].value_counts().reset_index(name="count").query("count > 1000")["index"].tolist()
        movie_data = movie_data[movie_data["Genre_clean"].isin(chosen_genres)].reset_index(drop=True)
        
        # Balance the data by randomly sampling an equal number of movies for each genre. I choose to sample 1000 for each genre
        clean_movie_data_balanced = movie_data.groupby("Genre_clean").sample(1000).reset_index(drop=True)
        
        # Select relevant columns
        clean_movie_data_balanced = clean_movie_data_balanced.loc[:, ("Title", "Plot", "Genre_clean")]
        
        return clean_movie_data_balanced
    
    
    def save_to_csv(self, clean_movie_data_balanced):
        """
        This method saves the clean dataframe to a CSV in the data folder.
        """
        # Define path
        path = os.path.join("..", "data", self.output_file)
    
        # Save dataframe as CSV in the data folder
        clean_movie_data_balanced.to_csv(path,
                                   index=True, 
                                   encoding="utf-8")
        
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()