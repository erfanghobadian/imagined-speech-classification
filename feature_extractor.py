import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pyts.image import RecurrencePlot


class FeatureExtractor:
    def __init__(self, all_trials, all_labels, run_fft=True, run_cov=True, run_normalize=True, output_shape=None, framed=False):

        self.all_trials = all_trials
        self.all_labels = all_labels
        self.classes = np.unique(all_labels)
        self.run_fft = run_fft
        self.run_cov = run_cov
        self.run_normalize = run_normalize
        self.output_shape = output_shape if output_shape else (all_trials.shape[0], all_trials.shape[1], 62)
        self.Y_onehot = self.one_hot()
        self.framed = framed

    def one_hot(self):
        enc = OneHotEncoder(sparse=False)
        y_onehot = enc.fit_transform(self.all_labels[:, np.newaxis])
        return y_onehot

    def get_idx(self, data_size=1913*16, test_size=0.5, val_size=0.3):
        test_idx = np.random.choice(
            np.arange(data_size), int(data_size * test_size), replace=False
        )
        train_idx, val_idx = train_test_split(
            np.arange(int(data_size * (1 - test_size))), test_size=val_size, random_state=36, shuffle=True
        )
        return train_idx, val_idx, test_idx

    def split(self, X_feat, Y_onehot, train_idx, val_idx, test_idx):
        X_train = X_feat[train_idx]
        X_val = X_feat[val_idx]
        X_test = X_feat[test_idx]
        y_train = Y_onehot[train_idx]
        y_val = Y_onehot[val_idx]
        y_test = Y_onehot[test_idx]
        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize(self, X_train, X_val, X_test):
        if not self.framed:
            X_train, X_test, X_val = std(X_train, X_test, X_val)
        else:
            for i in range(X_train.shape[1]):
                X_train[:, i, :, :], X_test[:, i, :, :], X_val[:, i, :, :] = std(
                    X_train[:, i, :, :], X_test[:, i, :, :], X_val[:, i, :, :]
                )
        return X_train, X_val, X_test

    def extract(self):
        X_feat = np.zeros(self.output_shape)
        for i, epoch in enumerate(self.all_trials):
            if self.framed:
                for j, frame in enumerate(epoch):
                    if self.run_fft:
                        frame = fft(frame)
                    if self.run_cov:
                        frame = cov(frame)
                    X_feat[i, j, :, :] = frame
            else:
                if self.run_fft:
                    epoch = fft(epoch)
                if self.run_cov:
                    epoch = cov(epoch)
                X_feat[i] = epoch
        return X_feat

    def extract_recurrent_plot(self):
        X_feat = np.memmap("X_feat",mode="w+",shape=self.output_shape, dtype=bool)
        for i, epoch in enumerate(self.all_trials):
            rp = RecurrencePlot(threshold="distance", percentage=0.3)
            X_rp = rp.fit_transform(epoch)
            X_feat[i] = X_rp
        return X_feat

    def plot_data(self, X_train):
        for cls in self.classes:
            idx = np.where(self.all_labels == cls)[0][0]
            if self.run_cov:
                figure = plt.figure()
                plt.title(f"Class {cls} - Cross Covariance")
                axes = figure.add_subplot(111)
                caxes = axes.matshow(X_train[idx], interpolation ='nearest')
                figure.colorbar(caxes)
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False, labeltop=False)
                plt.show()
                print("\n\n\n")

                figure = plt.figure()
                plt.title(f"Class {cls} - Cross Covariance")
                axes = figure.add_subplot(111)
                caxes = axes.matshow(X_train[idx], interpolation ='nearest', cmap="magma")
                figure.colorbar(caxes)
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False, labeltop=False)
                plt.show()
                print("\n\n\n")
            else:
                plt.title(f"Class {cls} - Time Signal")
                plt.plot(np.arange(250), X_train[idx, :, :].T)
                plt.show()
                print("\n\n\n")

    def get_data(self, rec_plot=False, train_idx=None, val_idx=None, test_idx=None):
        if train_idx is None or val_idx is None or test_idx is None:
            train_idx, val_idx, test_idx = self.get_idx()
        if rec_plot:
            X_feat = self.extract_recurrent_plot()
        else:
            X_feat = self.extract()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X_feat, self.Y_onehot, train_idx, val_idx, test_idx)
        if self.run_normalize:
            X_train, X_val, X_test = self.normalize(
                X_train, X_val, X_test
            )
        return X_train, X_val, X_test, y_train, y_val, y_test