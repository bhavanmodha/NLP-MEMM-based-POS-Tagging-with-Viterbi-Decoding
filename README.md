# NLP-MEMM-based-POS-Tagging-with-Viterbi-Decoding

## Project Overview
This project involves developing and implementing a bigram Maximum Entropy Markov Model (MEMM) for part-of-speech (POS) tagging. The project includes feature engineering, model training, and testing using the Viterbi decoding algorithm.

## Features
- **Feature Generation**: 
  - Generates features at the word level, including n-gram features and word property features.
  - Incorporates tag bigram features to capture word sequence information and general word characteristics.

- **Training Data Preparation**:
  - Creates feature dictionaries and tag dictionaries.
  - Converts corpus tags into numerical format.
  - Represents feature vectors using a sparse matrix (Compressed Sparse Row format).

- **Model Training**:
  - Utilizes a portion of the Brown corpus for training.
  - Instantiates a Logistic Regression model with `class_weight=balanced`, `solver=saga`, and `multi_class=multinomial`.
  - Trains the model using the `fit()` method.

- **Testing and Prediction**:
  - Implements Viterbi decoding for predicting POS tags.
  - Loads test data and generates predictions using the trained model.
  - Converts predicted tag indices into tags using a reverse-tag-dictionary.

## Functions

- **Feature Generation**:
  - `get_features()`: Generates feature lists for words.
  - `remove_rare_features()`: Removes rare features from the feature dictionary.

- **Training Data Preparation**:
  - `build_X(corpus_features, feature_dict)`: Builds the input matrix in CSR format for training.
  - `build_Y(corpus_tags, tag_dict)`: Builds the output matrix for training.

- **Model Training**:
  - `train(proportion=1.0)`: Trains the MEMM model using a specified proportion of the Brown corpus.

- **Testing and Prediction**:
  - `load_test_corpus(corpus_path)`: Loads test data.
  - `viterbi()`: Implements Viterbi decoding.
  - `predict(corpus_path, model, feature_dict, tag_dict)`: Predicts POS tags for test sentences using the trained model.

### Requirements

- Python 3.x
- Scikit-Learn
- Numpy
- NLTK
- Scipy
