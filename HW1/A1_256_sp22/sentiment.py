#!/bin/python
# Parse input arguments
import argparse
import nltk
import warnings

# Download relevant NLTK packages
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

eng_stopwords = set(stopwords.words('english')) - set(['not'])
warnings.filterwarnings("ignore")
lem = WordNetLemmatizer()
ps = PorterStemmer()

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    tar.close()
    return sentiment

def preprocess_text(text, stopword=False, tokenize=False, reduction_method=None):
    '''
    Method to process a given block of text using one or all of stopword removal, tokenization, and stemming/lemmatization based on arguments passed

    stopword: Should stopwords be removed
    tokenize: Should WordNet tokenizer be used
    reduction_method: Do we need to apply stemming or lemmatization
    '''
    if tokenize:
        words = word_tokenize(text)
    else:
        words = text.split(" ")
    if reduction_method == "Stemming":
        words = [ps.stem(w) for w in words]
    elif reduction_method == "Lemmatization":
        words = [lem.lemmatize(word,"v") for word in words]
    if stopword:
        words = [w for w in words if not w in eng_stopwords]
    return " ".join(words)

def clean_data(sentiment, n, stop_word_removal=False, tokenize=False, reduction_method=None):
    '''
    Method to clean the given block of data, including calling the preprocess_text method and vectorization
    
    sentiment: Original data
    n: Maximum value of n to pass to TfIdf vectorizer to extract n-grams
    stop_word_removal, tokenize, reduction_method: Values of conditions to be passed to preprocess_text method
    '''
    print("-- Cleaning data and labels")
    sentiment.train_data = [preprocess_text(x,stop_word_removal, tokenize, reduction_method) for x in sentiment.train_data]
    sentiment.dev_data = [preprocess_text(x,stop_word_removal, tokenize, reduction_method) for x in sentiment.dev_data]
    #sentiment.dev_data = sentiment.dev_data.apply(lambda x: preprocess_text(x,stop_word_removal, tokenize, reduction_method))

    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.tfidf_vect = TfidfVectorizer(ngram_range=(1,n))
    sentiment.trainX = sentiment.tfidf_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.tfidf_vect.transform(sentiment.dev_data)

    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    return sentiment


def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


if __name__ == "__main__":

    # Define arguments to be taken from the user along with their default values
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--c', help='Range of C values to explore', type=str, nargs='?', default="5.0")
    parser.add_argument('-n', '--n', help='Maximum n-gram feature length to explore', type=int, nargs='?', default=1)
    parser.add_argument('-s', '--stopwords', help='Exclude stopwords or not', type=str, nargs='?', default="False")
    parser.add_argument('-t', '--tokenize', help='Use WordNet tokenization', type=str, nargs='?', default="False")
    parser.add_argument('-r', '--reduction', help='Word reduction method', type=str, nargs='?', default="None")

    args = parser.parse_args()
    # Split C into a list through which we will iterate
    c_list = [float(item) for item in args.c.split(',')]
    n = args.n
    st = args.stopwords == "True"
    tk = args.tokenize == "True"
    red = args.reduction

    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    # Iterate through different combinations of n and C
    for i in range(1,n+1):
        sentiment = clean_data(sentiment,i,st, tk, red)
        import classify
        for c_val in c_list:
            print("\nTraining classifier with n=",i," and C=",c_val)
            cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, c_val)
            print("\nEvaluating")
            classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
            classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
