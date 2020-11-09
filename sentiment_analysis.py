# https://github.com/alisoltanirad/Sentiment-Analysis-On-Small-Datasets
# Dependencies: numpy, pandas, nltk, sk-learn, keras
import ssl
import re
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.preprocessing import sequence
from keras.datasets import imdb


def train_imdb_network(parameters):
    (x_train, y_train), (x_test, y_test) = load_imdb_dataset(
        parameters['vocabulary_size'])
    x_train = sequence.pad_sequences(x_train, maxlen=parameters['max_words'])
    x_test = sequence.pad_sequences(x_test, maxlen=parameters['max_words'])
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                        test_size=0.15)

    classifier = Sequential()
    classifier.add(Embedding(parameters['vocabulary_size'], 64,
                             input_length=parameters['max_words']))
    classifier.add(Dense(32, activation='relu'))
    classifier.add(Dense(32, activation='relu'))
    classifier.add(Flatten())
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

    classifier.fit(x_train, y_train, validation_data=(x_test, y_test),
                   batch_size=128, epochs=1, verbose=1)

    weights = [
        classifier.layers[1].get_weights(),
        classifier.layers[2].get_weights()
    ]
    return weights


def load_imdb_dataset(vocabulary_size):
    temp = np.load
    np.load = lambda *a, **k: temp(*a, allow_pickle=True, **k)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',
                                                    num_words=vocabulary_size)
    np.load = temp
    return (x_train, y_train), (x_test, y_test)


def train_dataset(parameters, weights):
    dataset = pd.read_csv(
        'https://raw.githubusercontent.com/alisoltanirad'
        '/Sentiment-Analysis-Farsi-Dataset/master'
        '/TranslatedDigikalaDataset.csv', sep=',')
    y = dataset.iloc[:, 1].values

    nltk.download('stopwords')
    x = []
    for i in range(719):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Comment'][i])
        review = review.lower()
        review = review.split()
        ps = nltk.stem.porter.PorterStemmer()
        review = [ps.stem(word) for word in review if
                  word not in set(nltk.corpus.stopwords.words('english'))]
        review = ' '.join(review)
        x.append(review)

    cv = CountVectorizer(max_features=parameters['max_words'])
    x = cv.fit_transform(x).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.15)
    model = Sequential()
    model.add(Embedding(parameters['vocabulary_size'], 64,
                        input_length=parameters['max_words']))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.layers[1].set_weights(weights[0])
    model.layers[2].set_weights(weights[1])

    for layer in model.layers[1:]:
        layer.trainable = False

    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)

    y_prediction = model.predict(x_test)

    y_prediction = (y_prediction > 0.5)

    cm = confusion_matrix(y_test, y_prediction)

    correct_predictions = cm[0][0] + cm[1][1]
    all_predictions = correct_predictions + (cm[0][1] + cm[1][0])
    accuracy = round((correct_predictions / all_predictions) * 100, 3)
    print('\n')
    print(accuracy)


def set_processing_parameters():
    parameters = {
        'vocabulary_size' : 5000,
        'max_words' : 500
    }
    return parameters


def main():
    parameters = set_processing_parameters()
    network_weights = train_imdb_network(parameters)
    train_dataset(parameters, network_weights)


if __name__ == '__main__':
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    main()