import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, recall_score, balanced_accuracy_score


class CnnModel:
    def __init__(self, input_shape, x_train, x_test, x_val, y_train, y_test, y_val, learning_rate=0.0001,
                 model_type="cnn"):
        self.input_shape = input_shape
        if model_type == "cnn_lstm":
            self.model = self.create_lstm_model()
        elif model_type == "cnn_recurrent":
            self.model = self.create_recurrent_plot_model()
        elif model_type == "eegnet":
            self.model = self.create_eegnet_model()
        else:
            self.model = self.create_model()

        self.history = None
        self.X_train = x_train
        self.X_test = x_test
        self.X_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.learning_rate = learning_rate
        self.compile()

    def create_eegnet_model(self):
        model = EEGNet(nb_classes=11, Chans=self.input_shape[0], Samples=self.input_shape[1])
        return model

    def create_lstm_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.ConvLSTM2D(
                64, (3, 3),
                strides=(1, 1), padding='same', activation='relu',
                recurrent_activation="sigmoid", data_format='channels_last',
                input_shape=self.input_shape, return_sequences=True
            ),
            tf.keras.layers.ConvLSTM2D(
                128, (3, 3),
                strides=(1, 1), padding='same', activation='relu',
                recurrent_activation="sigmoid", data_format='channels_last',
                return_sequences=True
            ),
            tf.keras.layers.ConvLSTM2D(
                64, (3, 3),
                strides=(1, 1), padding='same', activation='relu',
                recurrent_activation="sigmoid", data_format='channels_last',
                return_sequences=True
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(11, activation="softmax"),
        ])
        return model

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                64, (3, 3), activation='relu', input_shape=self.input_shape,
                strides=(1, 1), padding='same',
            ),
            tf.keras.layers.Conv2D(
                128, (3, 3),
                activation='relu',
                strides=(1, 1), padding='same',
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(11, activation="softmax"),
        ])
        return model

    def create_recurrent_plot_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(11, activation="softmax"),
        ])
        return model

    def kappa_score(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"Kappa score: {kappa}")

    def recall_score(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        recall = recall_score(y_true, y_pred, average="macro")
        print(f"Recall score: {recall}")

    def balanced_accuracy_score(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        print(f"Balanced accuracy score: {balanced_accuracy}")

    def compile(self):
        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics="accuracy"
        )

    def fit(self, epochs=10):
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs
        )

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test)

    def plot_train_val_accuracy(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def plot_train_val_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.X_val)
        y_true = np.argmax(self.y_val, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(y_true, y_pred)
        # cm = cm / np.sum(cm, axis=0)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='.2f')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()