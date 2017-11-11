"""RNN model script"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.models import load_model
from abstract_model import AbstractModel
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

class LSTMModel(AbstractModel):
    """RNNModel"""
    def __init__(
        self,
        train_valid_split=0.1,
        top_words=5000,
        embedding_vector_length=64,
        batch_size=5,
        maxlen=500):
        """__init__"""
        self.model = None
        self.train_valid_split = train_valid_split
        self.top_words = top_words
        self.embedding_vector_length = embedding_vector_length
        self.batch_size = batch_size
        self.maxlen = maxlen

    def fit(self, inputs):
        """fit method"""
        x_all = inputs[0]
        y_all = inputs[1]
        x_all = sequence.pad_sequences(x_all, maxlen=self.maxlen)
        x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=self.train_valid_split)
        
        if self.model is None:
            self.model = Sequential()
            self.model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=500))
            self.model.add(LSTM(100))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(len(y_train[0]), activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        self.model.fit(x_train, y_train, epochs=3, batch_size=self.batch_size)
        # Final evaluation of the model
        scores = self.model.evaluate(x_val, y_val, verbose=0)
        print("Validation Set Accuracy: %.2f%%" % (scores[1]*100))
        return self.model

    def predict(self, inputs):
        """predict method"""
        inputs = sequence.pad_sequences(inputs, maxlen=self.maxlen)
        return self.model.predict(inputs)

    def save(self, file_location):
        """save method"""
        self.model.save(file_location)
        return

    def load(self, file_location):
        """load method"""
        self.model = load_model(file_location)
        return self.model
