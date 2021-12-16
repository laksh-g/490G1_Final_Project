import os
import matplotlib.pyplot as plt
import librosa.display
import sklearn
from main_test import MusicGenreNet
import torchvision.io
import torchvision.transforms as T
import torch
import multiprocessing
import GUI

from pydub import AudioSegment

USE_CUDA = False
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())
kwargs = {'num_workers': 6,
          'pin_memory': False} if use_cuda else {}

class application:
    def audio_to_wavelets(self, mp3):
        if not os.path.exists('temp'):
            os.mkdir('temp')
        mp3_title = os.path.basename(mp3)
        mp3_path = os.path.join(os.getcwd(), mp3_title)
        save_path = os.path.join(os.getcwd(), 'temp')
        export_dst = os.path.join(save_path, mp3_title.split(".")[0] + '.wav')
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(export_dst, format="wav")
        x, sr = librosa.load(export_dst)
        librosa.display.waveplot(x)

        wavelet_path = os.path.join(os.getcwd(), 'temp', mp3_title.split(".")[0] + '_wavelet.png')
        plt.savefig(wavelet_path)
        plt.close()
        return wavelet_path

    def audio_to_spectrogram(self, mp3):
        if not os.path.exists('temp'):
            os.mkdir('temp')
        mp3_title = os.path.basename(mp3)
        mp3_path = os.path.join(os.getcwd(), mp3_title)
        save_path = os.path.join(os.getcwd(), 'temp')
        export_dst = os.path.join(save_path, mp3_title.split(".")[0] + '.wav')
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(export_dst, format="wav")

        x, sr = librosa.load(export_dst)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb)
        spectrogram_path = os.path.join(os.getcwd(), 'temp', mp3_title.split(".")[0] + '_spectrogram.png')
        plt.savefig(spectrogram_path)
        plt.close()
        return spectrogram_path

    def get_inputs(self, mp3, model_version=None):
        #print("Please enter audio file path")
        # audio_file = input()
        audio_file = mp3
        print(f"Extracting wavelet information from {audio_file}...")
        spectrogram = app.audio_to_spectrogram(audio_file)
        print(f"Extracting spectrogram information from {audio_file}...")
        wavelet =  app.audio_to_wavelets(audio_file)
        if model_version is not None:
            model_path = model_version
        else:
            model_path = "C:\\Users\\laksh\\Desktop\\UW\\Deep Learning\\saved_models\\model_2.0"
            print(f"Defaulting to model path {model_path}")
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
    mp3_name = GUI.browseFiles()
    app = application()
    model_version = 'saved_models/model_2.0'
    audio_file, spectrogram, wavelet, model = app.get_inputs(mp3_name, model_version)
    genre = app.classify_audio(audio_file, spectrogram, wavelet, model)
