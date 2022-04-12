# Assignment 1, CSE 256 UCSD
## By Pratik Ratadiya (PID: A59002413)
This assignment focuses on text classification in Python

### Directory Structure
The directory structure is as follows:

- sentiment.py: Main file to be executed for running the training and evaluation scripts
- classify.py: Utility file containing method to train the classifier
- data: Directory containing data files. Do not modify this, as the sentiment.py script does the needful data loading operations based on the provided tarball

### External Package Dependencies
- nltk
- sklearn

We recommend setting up an Anaconda environment to execute the script

### Execution Methods
To run the program, execute:
 > python sentiment.py
 
 This will execute the script with the default parameters. You have the ability to pass your custom set of parameters. A sample complete execution signature is as follows:
 
 > python sentiment.py --n=5 --c="0.01,0.5,1" --tokenize="True" --stopwords="True" --reduction="Stemming"
 
 where:
 - n (int): The maximum n-grams to extract. Default value is 1.
 - c (str): A string containing all C values to try for the classifier, separated using comma. Default value is "5.0"
 - tokenize (str): A string indicating boolean value of whether to use WordNet tokenization or not. Default value is "False" (note the double quotes)
 - stopwords (str): A string indicating boolean value of whether to remove stopwords or not. Default value is "False" (note the double quotes)
 - reduction (str): A string indicating what word reduction strategy to use. Can be one of "Stemming", "Lemmatization", or "None". Default value is "None"
