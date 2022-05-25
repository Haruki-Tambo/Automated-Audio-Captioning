import numpy as np
import csv
import pickle
from collections import defaultdict
from librosa.feature import melspectrogram
import librosa
import os
import torch


# CNNの教師データ用のマルチone-hotベクトルを作る
def create_word_index_dict(path):
    word_index_dict = {}
    index = 0
    with open(path) as f:
        reader = csv.reader(f)
        for low in reader:
            if low[0] != 'file_name':
                sentence = low[1].split()
                for word in sentence:
                    if word in word_index_dict:
                        continue
                    else:
                        word_index_dict[word] = index
                        index += 1
    return word_index_dict

def create_one_hot_vec(word_index, path):
    one_hot_vec = defaultdict(int)
    with open(path) as f:
        reader = csv.reader(f)
        for low in reader:
            vec = np.zeros(300)
            if low[0] != 'file_name':
                sentence = low[1].split()
                for word in sentence:
                    vec[word_index[word]] = 1
                one_hot_vec[low[0]] = vec
    return one_hot_vec

def create_ordered_word_vec(path, files_order, name):
    word_index = create_word_index_dict(path)
    one_hot_vec = create_one_hot_vec(word_index, path)
    vecs = np.zeros((1, 1, 300))
    for i, file in enumerate(files_order):
        vec = one_hot_vec[file]
        vec = vec.reshape(1, 1, 300)
        vecs = np.vstack((vecs, vec))
        print(f'キャプション {i+1}個目　{file}')
    np.save(f'data/{name}_word_label.npy', vecs)
    return print('complete')


# オーディオデータの前処理
def feature_extraction(audio_data: np.ndarray,
                       sr: int,
                       nb_fft: int,
                       hop_size: int,
                       nb_mels: int,
                       f_min: float,
                       f_max: float,
                       htk: bool,
                       power: float,
                       norm: bool,
                       window_function: str,
                       center: bool) \
        -> np.ndarray:
    y = audio_data
    mel_bands = melspectrogram(
        y=y, sr=sr, n_fft=nb_fft, hop_length=hop_size, win_length=nb_fft,
        window=window_function, center=center, power=power, n_mels=nb_mels,
        fmin=f_min, fmax=f_max, htk=htk, norm=norm).T
    logmel_spectrogram = librosa.core.power_to_db(
        mel_bands, ref=1.0, amin=1e-10,
        top_db=None)
    logmel_spectrogram = logmel_spectrogram.astype(np.float32)
    return logmel_spectrogram


def wav_to_mel(wav_path, settings, name):
    wav_files = os.listdir(f'{wav_path}')
    mels = np.zeros((1, settings['cut_point'], settings['nb_mels']))
    files_order = []
    for i, wav_file in enumerate(wav_files):
        y = librosa.load(path=f'{wav_path}/{wav_file}', sr=int(settings['sr']), mono=settings['to_mono'])[0]
        mel = feature_extraction(y, sr=settings['sr'],
                           nb_fft=settings['nb_fft'],
                           hop_size=settings['hop_size'],
                           nb_mels=settings['nb_mels'],
                           f_min=settings['f_min'],
                           f_max=settings['f_max'],
                           htk=settings['htk'],
                           power=settings['power'],
                           norm=settings['norm'],
                           window_function=settings['window_function'],
                           center=settings['center'])
        print(mel.shape)
        mel = mel[:settings['cut_point']].reshape(1, settings['cut_point'], settings['nb_mels'])
        mels = np.vstack((mels, mel))
        files_order.append(wav_file)
        print(f'オーディオ {i+1}個目 {wav_file}')
    with open(f'data/{name}_files_order', 'wb') as f:
        pickle.dump(files_order, f)
    np.save(f'data/{name}_processed_wav.npy', mels)
    return print('complete')


settings = {
    'sr': 44100,
    'nb_fft': 1024,
    'hop_size': 512,
    'nb_mels': 64,
    'window_function': 'hann',
    'center': 'Yes',
    'f_min': .0,
    'f_max': None,
    'htk': 'No',
    'power': 1.,
    'norm': 1,
    'to_mono': 'Yes',
    'cut_point': 1200
}


# dev_path = 'data/processed_captions_development.csv'
# wav_path = 'data/development'
name = 'dev'
# wav_to_mel(wav_path, settings, name)
with open(f'data/{name}_files_order', 'rb') as f:
    files_order = pickle.load(f)
# create_ordered_word_vec(dev_path, files_order, name)

# eva_path = 'data/processed_captions_evaluation.csv'
# wav_path = 'data/evaluation'
# name = 'eva'
# wav_to_mel(wav_path, settings, name)
# with open(f'data/{name}_files_order', 'rb') as f:
#     files_order = pickle.load(f)
# create_ordered_word_vec(eva_path, files_order, name)

# Datasetを作る
name = 'dev'
word_label = np.load(f'data/{name}_word_label.npy')
processed_wav = np.load(f'data/{name}_processed_wav.npy')
X = torch.tensor(processed_wav, dtype=torch.float32)
y = torch.tensor(word_label, dtype=torch.int64)
Dataset = torch.utils.data.TensorDataset(X, y)
print(Dataset)