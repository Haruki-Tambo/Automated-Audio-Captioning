import csv
import pickle
import numpy as np
import torch
from gensim.models.word2vec import Word2Vec

def adjust_cap(word_num, file_path):
    data = {}

    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            data[row[0]] = row[1:]
        data.pop('file_name')

    for audio_file_name, cap in data.items():
        for i in range(len(cap)):
            cap[i] = cap[i].replace(',', '')
            cap[i] = cap[i].replace(' .', '')
            cap[i] = cap[i].replace('.', '').lower()
            while len(cap[i].split()) < word_num:
                cap[i] += ' <pad>'

    return data

def create_dataset(wav, caption, files_order, w2v_model_path, word_num, name):
    model = Word2Vec.load(w2v_model_path)
    cap_data = []
    wav_data = []
    ref_data = []

    for i in range(len(files_order)):
        if i == 20:break

        print(f'{i+1}times files order')

        five_cap = caption[files_order[i]]
        ref_list = []
        for cap in five_cap:
            id_list = []
            for key in cap.split():
                word_id = model.wv.get_index(key)
                id_list.append(word_id)
            cap_data.append(id_list)
            wav_data.append(wav[i])
            ref_list.append(id_list)
        for j in range(5):
            ref_data.append(ref_list)

    X = torch.tensor(wav_data, dtype=torch.float32)
    y = torch.tensor(cap_data, dtype=torch.int64)
    ref = torch.tensor(ref_data, dtype=torch.int64)
    Dataset = torch.utils.data.TensorDataset(X, y, ref)

    with open(f'data/{name}_transformer_dataset_{i+1}', 'wb') as f:
        pickle.dump(Dataset, f)

    return Dataset

WORD_NUM = 20
file_path = 'data/clotho_captions_evaluation.csv'
data = adjust_cap(WORD_NUM, file_path)
name = 'eva'
w2v_model_path = 'data/w2v_192_pad.mod'
with open(f'data/{name}_files_order', 'rb') as f:
    files_order = pickle.load(f)
processed_wav = np.load(f'data/{name}_processed_wav.npy')
dataset = create_dataset(processed_wav, data, files_order, w2v_model_path, WORD_NUM, name)
print(dataset)