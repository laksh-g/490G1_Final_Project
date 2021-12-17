# 490G1_Final_Project: Music Genre Classification 

## Motivation
Can we capture abstract creative process of individual music composers? In this project we will try to tackle the problems of classification on musical pieces and, upon successful completion, music generation. In the first leg of the project, we will make a model to predict the genre/ composer of any musical piece. Upon successful completion of this part, we will try to tackle the problem of music generation - generating music in the styles of specific composers/ genres - to emulate human creativity.

## Abstract
This is the final class project for 'CSE 490G1: Deep Learning'. The current version implements a music genre classifier which has been trained on 10 different genres of music. The accuracy on our training and testing data are around 30% and 20% respectively. We are using the spectrogram graph images of the music piece and training our model on those images. Similarly, for testing, we use the spectrogram graph images of the music piece as the input for our trained model.

## Problem Statement
The problem that we solved is "Classifying the genre for a piece of music". 

## Related Work
### Inpiration
We looked at few research papers related to the process of music classification and music generation. For eg. Brunner et al. (MIDI-VAE) from 2018 and Hernandez-Olivan et al., from 2021.

### Dataset [(Link)](http://opihi.cs.uvic.ca/sound/genres.tar.gz)
We are using the GTZAN dataset, which contains 1000 music files. The GTZAN dataset is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions. Dataset has ten types of genres with uniform distribution. Dataset has the following genres: blues, classical, country, disco, hiphop, jazz, reggae, rock, metal, and pop. Each music file is 30 seconds long.

## Methodology
### Training (spectrogram images -> CNN -> Classification (genre))
1. Using a small part of the GTZAN dataset to get spectrogram images of different genres of music.
2. Processing the data to get train and test images as tensors of tensors.
3. Trained a CNN model in Pytorch with the following layers.
   - Defined layers:
        - self.conv1 = nn.Conv2d(4, 8, (3, 3), (2, 2))
        - self.maxpool1 = nn.MaxPool2d(3, 2)
        - self.conv2 = nn.Conv2d(8, 16, (4, 4), (2, 2))
        - self.maxpool2 = nn.MaxPool2d(3, 2)
        - self.conv3 = nn.Conv2d(16, 24, (5, 5), (2, 2))
        - self.maxpool3 = nn.MaxPool2d(3, 2)
        - self.linear1 = nn.Linear(240, 64)
        - self.linear2 = nn.Linear(64, 10)
   - forward pass
        - x = self.conv1(x)
        - x = F.leaky_relu(x)
        - x = self.maxpool1(x)
        - x = self.conv2(x)
        - x = F.leaky_relu(x)
        - x = self.maxpool2(x)
        - x = self.conv3(x)
        - x = F.leaky_relu(x)
        - x = self.maxpool3(x)
        - x = torch.flatten(x, 1)
        - x = self.linear1(x)
        - x = F.leaky_relu(x)
        - x = self.linear2(x)
4. Get training and testing accuracy.

### User Interaction (mp3 -> spectrogram -> Model input -> Model output -> Prediction)
1. Taking mp3 file input from user. (tkinter)
2. Converting mp3 files to wav files. (pydub)
3. Creating spectrogram graph images from .wav files. (librosa)
4. Using the spectrogram image as input in our model. (trained CNN model)
5. Get class label and print prediction on dialogue box. (tkinter)

## Experiments and Evaluation
We are using the training and test accuracies to evaluate our model. Moreover, we tried inputting other mp3 files (recorded and internet sourced) to test the predictions of our model.  

## Results
- Training Accuracy = ~ 30%
- Testing Accuracy = ~ 20%

## Examples
1. Live recording and classifying music [(Video Link)](https://drive.google.com/file/d/1jwlGGtvpO5QxxNBi5Uu47j3WT0ZK3p9v/view?usp=drivesdk)
2. Screen Recorded demo [(Video Link)](https://drive.google.com/file/d/1basZgFVq08BgMmi4refd-uzc3KIt9caK/view?usp=sharing)

## Installation
1. Clone the repo ```(git clone https://github.com/laksh-g/490G1_Final_Project)```
2. Make sure you have python 3.8+ installed.
3. Create new virtual environement.
4. run ```pip install -r requirements.txt``` to install dependencies.
5. You will also need to install ffmpeg [(Link)](https://www.ffmpeg.org/download.html). After downloading the binaries, add the 'bin folder path' to the PATH variable in System Variables.

## Running the classifier
1. Run the front_end.py file.
2. A file browser will appear. Here, select the .mp3 file that you want to classify.
3. An output on the dialogue box will appear showing the predicted musical genre for your mp3 file.

