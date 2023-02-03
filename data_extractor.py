import glob
import mne
import scipy.io as sio
import numpy as np

subjects = [
    "MM05",
    "MM08",
    "MM09",
    "MM10",
    "MM11",
    "MM12",
    "MM14",
    "MM15",
    "MM16",
    "MM18",
    "MM19",
    "MM20",
    "MM21",
    "P02"
]
drop_ch = (['M1', 'M2', 'EKG', 'EMG', 'Trigger', "VEO", "HEO"])
channels = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

BASE_PATH = f"drive/Shareddrives/Drive/BCI/p/spoclab/users/szhao/EEG/data/"


class DataExtractor:

    def apply_notch_filter(self, data):
        data.notch_filter(np.array((60, 120, 180, 240)))
        return data

    def get_indexes(self, subject):
        path = f"{BASE_PATH}/{subject}/epoch_inds.mat"
        f = glob.glob(path)[0]
        epoch_inds = sio.loadmat(f, variable_names=('thinking_inds'))["thinking_inds"][0]
        return epoch_inds

    def extract_raw_data(self, subject):
        path = f"{BASE_PATH}/{subject}/*.cnt"
        f = glob.glob(path)[0]
        raw = mne.io.read_raw_cnt(f, eog=['VEO', 'HEO'], ecg=['EKG'], emg=['EMG'], preload=True)
        return raw

    def extract_labels(self, subject):
        path = f"{BASE_PATH}/{subject}/all_features_simple.mat"
        f = glob.glob(path)[0]
        prompts_to_extract = sio.loadmat(f)
        prompts = prompts_to_extract['all_features'][0, 0]["prompts"][0]
        return prompts

    def get_trials(self, raw, epoch_inds):
        raw = self.apply_notch_filter(raw)
        raw.drop_channels(self.drop_ch)
        print(raw.info)
        print(raw.info.ch_names)
        print(1 / 0)
        raw_data = raw.get_data()
        res = np.empty((epoch_inds.shape[0], raw_data.shape[0], 4000))
        for i, t in enumerate(epoch_inds):
            epoch = raw_data[:, t[0][0] + 500:t[0][0] + 4500]
            res[i] = epoch
        return res

    def get_subject_data(self, subject):
        raw = self.extract_raw_data(subject)
        epoch_inds = self.get_indexes(subject)
        labels = self.extract_labels(subject)
        trials = self.get_trials(raw, epoch_inds)
        return trials, labels

    def get_all_subjects_data(self):
        all_trials = np.empty((0, 62, 4000))
        all_labels = np.empty((0,))
        for subject in self.subjects:
            trials, labels = self.get_subject_data(subject)
            all_trials = np.concatenate((all_trials, trials), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
        return all_trials, all_labels

    def load_data(self):
        trials = np.load(f"drive/Shareddrives/Drive/BCI/trials.npy")
        labels = np.load(f"drive/Shareddrives/Drive/BCI/labels.npy", allow_pickle=True)
        return trials, labels

    def window_data(self, trials, labels, window_size=250):
        nr_trials = trials.shape[0] * trials.shape[2] // window_size
        windowed_trials = np.empty((nr_trials, trials.shape[1], window_size))
        windowed_labels = np.empty((nr_trials,), dtype=object)
        i = 0
        for j, trial in enumerate(trials):
            for win in range(0, len(trial[0]), window_size):
                windowed_trials[i, :, :] = trial[:, win:win + window_size]
                windowed_labels[i] = labels[j]
                i += 1
        return windowed_trials, windowed_labels

    def create_frames(self, trials, frame_size=100, overlab=50):
        nr_frames = (trials.shape[2] // overlab) - 1
        framed_trials = np.empty((trials.shape[0], nr_frames, trials.shape[1], frame_size))
        for i, trial in enumerate(trials):
            for j in range(nr_frames):
                framed_trials[i, j, :, :] = trial[:, j * overlab:j * overlab + frame_size]
        return framed_trials

    def __init__(self, window_length, overlap, subjects, drop_ch):
        self.window_length = window_length
        self.overlap = overlap
        self.subjects = subjects
        self.drop_ch = drop_ch
