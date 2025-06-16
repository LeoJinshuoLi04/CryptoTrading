import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.dummy import DummyClassifier


class LSTMModel:
    def __init__(self, input_shape, n_classes=3, loss="categorical_crossentropy", epochs=30):
        self.model = self._build_model(input_shape, n_classes, loss)
        self.epochs = epochs
        self.encoder = LabelEncoder()
        self.encoder.fit(np.array([-1, 0, 1]))
        self.dummy_clf = DummyClassifier(strategy="stratified", random_state=42)

    def _build_model(self, input_shape, n_classes, loss):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # uncomment this code if you have a more powerful computer / can train for longer
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(n_classes, activation="softmax"))
        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        return model

    def train(self, X, y):
        y_encoded = self.encoder.fit_transform(y)
        y_cat = to_categorical(y_encoded)
        self.model.fit(X, y_cat, batch_size=64, epochs=self.epochs, verbose=1)

    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        return self.encoder.inverse_transform(np.argmax(preds, axis=1))

    def save(self, path="lstm_model.h5"):
        self.model.save(path)

    def load(self, path="lstm_model.h5"):
        self.model = tf.keras.models.load_model(path)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def dummy_train(self, X, y):
        self.dummy_clf.fit(X.reshape(X.shape[0], -1), y)

    def dummy_predict(self, X):
        return self.dummy_clf.predict(X.reshape(X.shape[0], -1))
