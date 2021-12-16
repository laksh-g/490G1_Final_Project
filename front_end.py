import os
import matplotlib.pyplot as plt
import librosa.display
import sklearn
from main_test import MusicGenreNet
import torchvision.io
import torchvision.transforms as T
import torch
import multiprocessing

USE_CUDA = False
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())
kwargs = {'num_workers': 6,
          'pin_memory': False} if use_cuda else {}

class application:
    def audio_to_wavelets(self, mp3):
        mp3_title = os.path.basename(mp3)
        mp3_path = os.path.join(os.getcwd(), mp3_title)
        x, sr = librosa.load(mp3_path)
        librosa.display.waveplot(x)
        if not os.path.exists('temp'):
            os.mkdir('temp')
        save_path = os.path.join(os.getcwd(), 'temp', mp3_title + '_wavelet.png')
        plt.savefig(save_path)
        plt.close()
        return save_path


    def audio_to_spectrogram(self, mp3):
        mp3_title = os.path.basename(mp3)
        mp3_path = os.path.join(os.getcwd(), mp3_title)
        x, sr = librosa.load(mp3_path)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb)
        save_path = os.path.join(os.getcwd(), 'temp', mp3_title + '_spectrogram.png')
        plt.savefig(save_path)
        plt.close()
        return save_path

    def get_inputs(self):
        print("Please enter audio file path")
        # audio_file = input()
        audio_file = "still got the blues.mp3"
        print(f"Extracting wavelet information from {audio_file}...")
        spectrogram = app.audio_to_spectrogram(audio_file)
        print(f"Extracting spectrogram information from {audio_file}...")
        wavelet =  app.audio_to_wavelets(audio_file)
        print("Enter model path")
        # model_path = input()
        model_path = "C:\\Users\\laksh\\Desktop\\UW\\Deep Learning\\saved_models\\model_2.0"
        print(f"Loading model...")
        net = MusicGenreNet()
        net.load_model(model_path)
        return audio_file, spectrogram, wavelet, net

    def classify_audio(self, audio_file, wavelet, spectrogram, model):
        img = torchvision.io.read_image(spectrogram)
        transform = T.Compose([
                               T.Resize((288, 432)),
                            ])

        x = transform(img)
        x = torch.unsqueeze(x, 0)
        genre = model.classify_single_input(model, x, device)
        print(f"Classified {audio_file} as {genre}")
        return genre

if __name__ == '__main__':
    app = application()
    audio_file, spectrogram, wavelet, model = app.get_inputs()
    genre = app.classify_audio(audio_file, spectrogram, wavelet, model)

