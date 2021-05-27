# Self-Assigned Portfolio: Classifying Movie Genre Based on Plot Description Using a CNN Model and Pretrained GloVe Word Embeddings

### Description of Task: Classifying Genre Based on Plot Descriotion <br>
For this assignment I chose to work with movie plot descriptions scraped from Wikipedia; a dataset available on [Kaggle](https://www.kaggle.com/jrobischon/wikipedia-movie-plots ). This dataset contains information about nearly 35,000 movies. I wanted to see whether it would be possible to predict the genre of a movie based on its plot description using pre-trained GloVe word embeddings. In other words, I wanted to see how accurately the relationship between plot and genre can be modelled using pretrained word embeddings. <br>

### Content and Repository Structure <br>
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. It is important to remark that all the code that has been produced has only been tested in Linux and MacOS. Hence, for the sake of convenience, I recommend using a similar environment to avoid potential problems. <br>
The repository follows the overall structure presented below. The two scripts, ```0-preprocessing.py``` and ```1-cnn-moviegenre.py```, are located in the ```src``` folder. The full dataset is provided in the ```data``` folder, and the outputs produced when running the scripts can be found within the ```output```folder. The ```utils``` folder stores a utility script with utility functions for creating the embedding matrix and plotting the loss/accuracy history of the model, and these are called in the main script. The README file contains a detailed run-through of how to engage with the code and reproduce the contents.

| Folder | Description|
|--------|:-----------|
| ```data``` | A folder containing a subset of the full dataset as well as a folder called unseen_images which contains example images that can be used as input for the use-model.py script. 
| ```src``` | A folder containing the python scripts for the particular assignment.
| ```output``` | A folder containing the outputs produced when running the python scripts within the src folder.
| ```utils``` | A folder containing utilty functions to be used in the main scripts.
| ```requirements.txt```| A file containing the dependencies necessary to run the python script.
| ```create_venv.sh```| A bash-file that creates a virtual environment in which the necessary dependencies listed in the ```requirements.txt``` are installed. This script should be run from the command line.
| ```LICENSE``` | A file declaring the license type of the repository.

### Usage and Technicalities <br>
To reproduce the results of this assignment, the user will have to create their own version of the repository by cloning it from GitHub. This is done by executing the following from the command line: 

```
$ git clone https://github.com/sofieditmer/self-assigned-lang.git  
```

Once the user has cloned the repository, a virtual environment must be set up in which the relevant dependencies can be installed. To set up the virtual environment and install the relevant dependencies, a bash-script is provided, which creates a virtual environment and installs the dependencies listed in the requirements.txt file when executed. To run the bash-script that sets up the virtual envi-ronment and installs the relevant dependencies, the user must first navigate to the topic modeling repository:

```
$ cd self-assigned-lang
$ bash create_venv.sh 
```

Once the virtual environment has been set up and the relevant dependencies listed in ```requirements.txt``` have been installed within it, the user is now able to run the scripts provided in the ```src``` folder directly from the command line. In order to run the script, the user must first activate the virtual environment in which the script can be run. Activating the virtual environment is done as follows:

```
$ source movie_venv/bin/activate
```

Once the virtual environment has been activated, the user is now able to run the scripts script within it:

```
(movie_venv) $ cd src

(movie_venv) $ python 0-preprocessing.py

(movie_venv) $ python 1-cnn-moviegenre.py
```

For the ```0-preprocessing.py``` script the user is able to modify the following parameters, however, this is not compulsory:

```
-i, --input_data: str <name-of-input-data>, default = "wiki_movie_plots.csv"
-o, --output_filename: str <name-of-output-file>, default = "clean_movie_data.csv"
```

For the ```1-cnn-moviegenre.py``` script the user is able to modify the following parameters, however, once again this is not compulsory:

```
-i, --input_data: str <name-of-input-data>, default = "clean_movie_data.csv"
-ts, --test_size: float <size-of-test-split>, default = 0.25
-n, --n_words: int <size-of-vocabulary>, default = 5000
-nd, --n_dimensions: int <number-of-embedding-dimensions>, default = 100
-e, --n_epochs: int <number-of-training-epochs>, default = 10
-b, --batch_size: int <size-of-batches>, default = 20
-r, --regularization_value: float <regularization-value>, default = 0.0001
-te, --train_embeddings: str <train-embeddings-true-or-false>, default = “False”
````

The abovementioned parameters allow the user to adjust the pipeline, but default parameters have been set making the script run without explicitly specifying these arguments.The user is able to modify the size of the test-split, the number of words in the vocabulary, the dimensions of the pretrained word embeddings, the number of training epochs, the batch size (the larger the batch size the more efficient processing), the regularization value, and whether to train the embeddings or not. 


### Output <br>
When running the ```0-preprocessing.py```script, the following files will be saved in the ```data``` folder: 
1.  ```clean_movie_data.csv``` Preprocessed data. 

When running the ```1.cnn-moviegenre.py```script, the following files will be saved in the ```output``` folder: 
1. ```cnn_100d_summary.txt``` Summary of the model architecture.
2. ```cnn_100d_architecture.png``` Summary of the model architecture.
3. ```cnn_100d_15epochs_classification_metrics.txt``` Classification report.
4. ```cnn_100d_15epochs_loss_accuracy_history.png``` Loss/accuracy curves.
5. ```cnn_100d_15epochs_plot_training_test_accuracies.png``` Training and validation accuracies.


### Discussion of Results <br>
Three different CNN models were trained. These models differed in the number of epochs and word embedding dimensions. The weighted average accuracy scores of the models were very similar. This section covers the results obtained by the CNN model trained for 20 epochs using pretrained GloVe word embeddings with 300 dimensions (see figure 1). Results for all models are available in the [output folder](https://github.com/sofieditmer/self-assigned-lang/tree/main/output)
 
<img src="https://github.com/sofieditmer/self-assigned-lang/blob/main/output/cnn_300d_architecture.png" width="500">
Figure 1: Summary of CNN model architecture. <br> <br>

This model obtained a weighted average accuracy score of 55% (see [Classification Report](https://github.com/sofieditmer/self-assigned-lang/blob/main/output/cnn_300d_20epochs_classification_metrics.txt)). The model achieved the highest F1-score for the horror genre, suggesting that movie plot descriptions of horror movies contain clear indicators of genre as opposed to comedy movie descriptions for which the model obtained a F1-score of only 29%. Thus, the question of whether movie plot description is a good predictor of genre seems to vary depending on the movie description in question. Overall, the model performs reasonably well. <br>
When assessing the accuracy and loss curves of the model for both the training and validation data, it seems that the model suffers from overfitting (see figure 2). In particular, this is indicated by the loss curves. While the training loss decreases, the validation loss increases substantially, suggesting that the model is overfitting the training data and having a hard time generalizing to the validation data. The training accuracy increases relatively quickly and reaches 100% accuracy already after 7 epochs, which also suggests overfitting. The validation accuracy increases but reaches a plateau of 50% after 5 epochs. 

<img src="https://github.com/sofieditmer/self-assigned-lang/blob/main/output/cnn_300d_20epochs_loss_accuracy_history.png" width="500">
Figure 2: Model history <br> <br>

The conclusions made based on the training and loss curves are likewise supported by the plot below (see figure 3). Here it becomes perhaps even more clear that the model is overfitting the training data, reaching almost 100% accuracy for all movie genres.

<img src="https://github.com/sofieditmer/self-assigned-lang/blob/main/output/cnn_300d_20epochs_plot_training_test_accuracies.png" width="500">
Figure 3: Training and testing accuracies <br> <br>

### License <br>
This project is licensed under the MIT License - see the [LICENSE](https://github.com/sofieditmer/self-assigned-lang/blob/main/LICENSE) file for details.

### Contact Details <br>
If you have any questions feel free to contact me on [201805308@post.au.dk](201805308@post.au.dk)
