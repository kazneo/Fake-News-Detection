import re
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Embedding
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

def normalize(data):
  normalized = []
  for i in data:
    i = i.lower()
    # get rid of urls
    i = re.sub('https?://\S+|www\.\S+', '', i)
    # get rid of non words and extra spaces
    i = re.sub('\\W', ' ', i)
    i = re.sub('\n', '', i)
    i = re.sub(' +', ' ', i)
    i = re.sub('^ ', '', i)
    i = re.sub(' $', '', i)
    normalized.append(i)
  return normalized

data=pd.read_csv('news.csv')


# data['label'].map({'FAKE':0,'REAL':1})


np.random.seed(42)
np.random.shuffle

train, test = np.split(data.sample(frac=1), [int(.8 * len(data))])

xTrain = train.iloc[:,1:-1]
yTrain = train.iloc[:,-1]

xTest = test.iloc[:,1:-1]
yTest = test.iloc[:,:-1]

# X = data.iloc[:, 1:-1]
# y = data.iloc[:,:-1]

# xTrain, xTest , yTrain , yTest  = train_test_split(X, y, test_size = 0.2, random_state = 18)

max_vocab = 10000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(xTrain)

# Tokenizing the text into vector
xTrain = tokenizer.texts_to_sequences(xTrain)
xTest = tokenizer.texts_to_sequences(xTest)
xTrain = tf.keras.preprocessing.sequence.pad_sequences(xTrain, padding='post', maxlen=256)
xTest = tf.keras.preprocessing.sequence.pad_sequences(xTest, padding='post', maxlen=256)

model = Sequential([
  Embedding(max_vocab, 32),
  Bidirectional(LSTM(64,  return_sequences=True)),
  Bidirectional(LSTM(16)),
  Dense(64, activation='relu'),
  Dropout(0.5),
  Dense(1)
])
# model.summary()

earlyStop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.compile(loss = BinaryCrossentropy(from_logits=True), optimizer = Adam(1e-4), metrics=['accuracy'])
history = model.fit(xTrain, yTrain, epochs=1000, validation_split=0.1, batch_size=30, shuffle=True, callbacks=[earlyStop])