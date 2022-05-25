import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from eval_metrics import evaluate_metrics_from_lists
import csv

def get_padding(tgt):
    # tgt: (batch_size, max_len)
    device = tgt.device
    batch_size = tgt.size()[0]
    max_len = tgt.size()[1]
    mask = torch.zeros(tgt.size()).type_as(tgt).to(device)
    for i in range(batch_size):
        d = tgt[i]
        # num_pad = max_len-int(tgt_len[i].item())
        num_pad = max_len - i
        mask[i][max_len - num_pad:] = 1
        # tgt[i][max_len - num_pad:] = pad_idx

    # mask:(batch_size,max_len)
    mask = mask.float().masked_fill(mask == 1, True).masked_fill(mask == 0, False).bool()
    return mask

def align_word_embedding(w2v_model_path, ntoken, nhid):
    model = Word2Vec.load(w2v_model_path)
    word_dict = model.wv.index_to_key
    word_emb = torch.zeros((ntoken, nhid)).float()
    word_emb.uniform_(-0.1, 0.1)
    # w2v_vocab = [k for k in model.wv.vocab.keys()]
    for i in range(len(model.wv.index_to_key)):
        word = word_dict[i]
        # if word in w2v_vocab:
        w2v_vector = model.wv[word]
        word_emb[i] = torch.tensor(w2v_vector).float()
    return word_emb

def greedy_decode(model, src, max_len, start_symbol_ind=0):
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    # memory = model.cnn(src)
    memory = model.encode(src)
    ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    for i in range(max_len - 1):
        # ys_i:(batch_size, T_pred=i+1)
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, nhid)
        prob = model.generator(out[-1, :])  # (T_-1, batch_size, nhid)
        next_word = torch.argmax(prob, dim=1)  # (batch_size)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        # ys_i+1: (batch_size,T_pred=i+2)
    return ys, out

def ind_to_str(sentence_ind, special_token, word_dict):
    sentence_str = []
    for s in sentence_ind:
        if word_dict[s] not in special_token:
            sentence_str.append(word_dict[s])
    return sentence_str

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.ignore_index:
                true_dist[:, self.ignore_index] = 0
                mask = torch.nonzero(target.data == self.ignore_index)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def calculate_spider(output_batch, ref_batch, word_dict_pickle_path):
    # word_dict = get_word_dict(word_dict_pickle_path)
    w2v_model_path = 'data/w2v_192_pad.mod'
    model = Word2Vec.load(w2v_model_path)
    word_dict = model.wv.index_to_key
    word_dict[0] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = [ind_to_str(o, special_token, word_dict) for o in output_batch]
    ref_str = [[ind_to_str(r, special_token, word_dict) for r in ref] for ref in ref_batch]
    # ref_str = [ind_to_str(ref, special_token, word_dict) for ref in ref_batch]

    output_str = [' '.join(o) for o in output_str]
    ref_str = [[' '.join(r) for r in ref] for ref in ref_str]
    # ref_str = [' '.join(r) for r in ref_str]

    # 入力はcsvファイルのパスか辞書のリスト？
    metrics, per_file_metrics = evaluate_metrics_from_lists(output_str, ref_str)
    score = metrics['SPIDEr']

    return score, output_str, ref_str

# test

if __name__ == '__main__':
    ntoken = 4369 + 1
    w2v_model_path = 'data/w2v_192_pad.mod'
    nhid = 192
    emb = align_word_embedding(w2v_model_path, ntoken, nhid)
    print(emb)