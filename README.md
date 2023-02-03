# imagined-speech-classification
Imagined Speech based on KaraOne Database

## Introduction
Karaone is a database of imagined speech. It contains 11 classes (4 words and 7 phonemic prompts), each of which is imagined by 14 different subject.
Read more about the database [here](https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html).

## Feature Extraction
The features extracted in different ways like Cross Covariance Matrix of channels in both time and frequency domain.

## Classification
The classification is done using different methods like CNN, CNN+LSTM, EEGNet.

## Results
The results are shown in the table below:

|                           Model                            | Accuracy |
|:----------------------------------------------------------:|:--------:|
|      CNN+Cross Covariance in Time domain+0.25s window      |   9.61   |
|             CNN+Time Signal with 0.25s window              |  15.66   |
|   CNN+Cross Covariance in Frequency domain+0.25s window    |  43.34   |
| CNN+LSTM+Cross Covariance in Frequency domain+0.25s window |  33.24   |
|                     EEGNet+0.25 window                     |  16.73   |

## Contributors
- [Erfan Ghobadian](https://github.com/erfanghobadian/)
- [Sarah Saremi](https://github.com/SarahSaremi)