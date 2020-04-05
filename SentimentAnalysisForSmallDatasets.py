# Import libraries
import numpy as np
import pandas as pd
import re
import ssl
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from keras import Sequential
from keras.preprocessing import sequence
from keras.layers import Embedding, LSTM, Dense
from keras.datasets import imdb

# Get english stop-words using nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')

# vocabulary_size is the number of most used words in IMDB dataset
vocabulary_size = 5000

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# Get IMDB data
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=vocabulary_size)

# restore np.load for future normal usage
np.load = np_load_old

# Get small dataset (translated from farsi to english)
translated_data = pd.read_csv('https://raw.githubusercontent.com/alisoltanirad'
                              '/Sentiment-Analysis-Farsi-Dataset/master'
                              '/TranslatedDigikalaDataset.csv', sep=',')

f_y = translated_data.iloc[0:719, 1].values

f_x = []
for i in range(719):
    review = re.sub('[^a-zA-Z]', ' ', translated_data['Comment'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if
              word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    f_x.append(review)


# max_words is the maximum number of words in a review
max_words = 500

X_train = sequence.pad_sequences(x_train, maxlen=max_words)
X_test = sequence.pad_sequences(x_test, maxlen=max_words)
cv = CountVectorizer(max_features=500)
f_x = cv.fit_transform(f_x).toarray()


x_train, x_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.15)
f_x_train, f_x_test, f_y_train, f_y_test = train_test_split(f_x, f_y,
                                                            test_size=0.15)


# LSTM neural network for IMDB dataset training
classifier = Sequential()
classifier.add(Embedding(vocabulary_size, 200, input_length=max_words))
classifier.add(LSTM(200))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

classifier.fit(x_train, y_train, validation_data=(x_test, y_test),
               batch_size=64, epochs=1, verbose=1)

# Save layer weights for fine-tuning
weights1 = classifier.layers[0].get_weights()
weights2 = classifier.layers[1].get_weights()


# LSTM neural network for small dataset training
model = Sequential()
model.add(Embedding(vocabulary_size, 200, input_length=max_words))
model.add(LSTM(200))
model.layers[0].set_weights(weights1)
model.layers[1].set_weights(weights2)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(f_x_train, f_y_train, batch_size=1, epochs=1, verbose=1)

# Predict test data of small dataset
f_y_prediction = model.predict(f_x_test)

# Prepare prediction for classification
f_y_predict = (f_y_prediction > 0.5)

# Confusion matrix for test data
cm = confusion_matrix(f_y_test, f_y_predict)

# Print test data accuracy
correct_predictions = cm[0][0] + cm[1][1]
all_predictions = correct_predictions + (cm[0][1] + cm[1][0])
accuracy = round((correct_predictions / all_predictions) * 100, 3)
print('\n')
print(accuracy)