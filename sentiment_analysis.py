# https://github.com/alisoltanirad/sentiment-analysis
# Dependencies: numpy, pandas, nltk, sk-learn, keras, reason

import ssl
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.preprocessing import sequence
from keras.datasets import imdb
from reason.stem import PorterStemmer
from reason.tokenize import WordTokenizer
from reason.metrics import accuracy


def main():
    parameters = set_processing_parameters()
    network_weights = train_imdb_network(parameters)
    analyze_dataset(parameters, network_weights)


def set_processing_parameters():
    parameters = {
        'vocabulary_size': 5000,
        'max_words': 500
    }
    return parameters


def train_imdb_network(parameters):
    (x, y), (_, __) = load_imdb_dataset(parameters['vocabulary_size'])
    x = sequence.pad_sequences(x, maxlen=parameters['max_words'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    classifier = Sequential()
    classifier.add(
        Embedding(
            input_dim=parameters['vocabulary_size'],
            output_dim=64,
            input_length=parameters['max_words']
        )
    )
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(Flatten())
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    classifier.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=32,
        epochs=5,
        verbose=2
    )

    weights = [
        classifier.layers[i].get_weights()
        for i in range(1, (len(classifier.layers) - 2))
    ]

    return weights


def load_imdb_dataset(vocabulary_size):
    temp = np.load
    np.load = lambda *a, **k: temp(*a, allow_pickle=True)
    (x, y), (_, __) = imdb.load_data(path='imdb.npz', num_words=vocabulary_size)
    np.load = temp
    return (x, y), (_, __)


def analyze_dataset(parameters, weights):
    dataset = pd.read_csv(
        'https://raw.githubusercontent.com/alisoltanirad'
        '/Sentiment-Analysis-Farsi-Dataset/master'
        '/TranslatedDigikalaDataset.csv',
        sep=','
    )
    y = dataset.iloc[:, 1].values
    x = preprocess_text(dataset['Comment'], parameters)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    corpus = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test
    }
    y_prediction = classify_translated_data(parameters, weights, corpus)

    evaluate_classifier(y_test, y_prediction)


def preprocess_text(corpus, parameters):
    nltk.download('stopwords')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stemmer = PorterStemmer()
    tokenizer = WordTokenizer()
    x = []

    for text in corpus:
        tokenized_text = tokenizer.tokenize(text)
        useful_words = [
            stemmer.stem(word)[0]
            for word in tokenized_text
            if word not in stopwords
        ]
        preprocessed_text = ' '.join(useful_words)
        x.append(preprocessed_text)

    cv = CountVectorizer(max_features=parameters['max_words'])
    x = cv.fit_transform(x).toarray()

    return x


def classify_translated_data(parameters, weights, corpus):
    classifier = Sequential()
    classifier.add(
        Embedding(
            input_dim=parameters['vocabulary_size'],
            output_dim=64,
            input_length=parameters['max_words']
        )
    )
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=32, activation='relu'))

    for i in range(1, len(classifier.layers)):
        classifier.layers[i].set_weights(weights[i-1])
        classifier.layers[i].trainable = False

    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(Flatten())
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    classifier.fit(
        x=corpus['x_train'],
        y=corpus['y_train'],
        batch_size=1,
        epochs=5,
        verbose=2
    )

    y_prediction = (classifier.predict(corpus['x_test']) > 0.5)

    return y_prediction


def evaluate_classifier(y_true, y_prediction):
    print('* Accuracy: {:.2%}'.format(accuracy(y_true, y_prediction)))


if __name__ == '__main__':
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    main()
