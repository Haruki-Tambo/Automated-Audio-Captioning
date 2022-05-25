import pickle
import torch
import torch.nn as nn
from utils import get_padding, align_word_embedding, greedy_decode, calculate_spider, LabelSmoothingLoss
from model import TransformerModel
from torch.utils.tensorboard import SummaryWriter
from utils import ind_to_str
from gensim.models.word2vec import Word2Vec
import logging

PAD_IDX = 0
ntoken = 4369 + 1
ninp = 64
nhead = 4
nhid = 192
nlayers = 2
batch_size = 16
clip_grad = 2.5
log_dir = 'models/{name}'.format(name='base')
writer = SummaryWriter(log_dir=log_dir)
pretrain_cnn_path = 'TagModel_50.pt'
w2v_model_path = 'data/w2v_192_pad.mod'
load_pretrain_cnn = True
freeze_cnn = True
pretrain_emb = align_word_embedding(w2v_model_path, ntoken, nhid)
pretrain_cnn = torch.load(pretrain_cnn_path, map_location=torch.device('cpu')) if load_pretrain_cnn else None
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = TransformerModel(ntoken, ninp, nhead, nhid, nlayers, batch_size, dropout=0.2,
                         pretrain_cnn=pretrain_cnn, pretrain_emb=pretrain_emb, freeze_cnn=freeze_cnn).to(device)
# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

# criterion = LabelSmoothingLoss(ntoken, smoothing=0.1, ignore_index=PAD_IDX)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

def train(model, epoch):
    model.train()
    total_loss_text = 0
    batch = 0
    with open(f'data/dev_transformer_dataset_21', 'rb') as f:
        train_dataset = pickle.load(f)
    train_data = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     drop_last=True)
    for src, tgt, ref in train_data:
        src = src.to(device)
        tgt = tgt.to(device)
        # tgt_pad_mask = get_padding(tgt, tgt.shape[1])
        tgt_pad_mask = get_padding(tgt)
        tgt_in = tgt[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_in, target_padding_mask=tgt_pad_mask)
        print(f'output : {output}', output)

        loss_text = criterion(output.contiguous().view(-1, ntoken), tgt_y.transpose(0, 1).contiguous().view(-1))
        loss = loss_text
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss_text += loss_text.item()

        writer.add_scalar('Loss/train-text', loss_text.item(), (epoch - 1) * len(train_data) + batch)

        batch += 1
    print(f'train mean loss : {total_loss_text/batch}')

def eval_all(max_len=20, eos_ind=9, word_dict_pickle_path=None):
    model.eval()
    with open(f'data/eva_transformer_dataset_21', 'rb') as f:
        evaluation_data = pickle.load(f)
    evaluation_data = torch.utils.data.DataLoader(dataset=evaluation_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             drop_last=True)

    w2v_model_path = 'data/w2v_192_pad.mod'
    w2v_model = Word2Vec.load(w2v_model_path)
    word_dict = w2v_model.wv.index_to_key

    with torch.no_grad():
        total_loss = 0
        cnt = 0
        for src, tgt, ref in evaluation_data:
            src = src.to(device)
            output, prob = greedy_decode(model, src, max_len=max_len)
            tgt_y = tgt[:, 1:]
            loss = criterion(prob.contiguous().view(-1, ntoken), tgt_y.transpose(0, 1).contiguous().view(-1))
            total_loss += loss
            cnt += 1

            output_str = [ind_to_str(o, special_token=['<sos>', '<eos>', '<pad>'], word_dict=w2v_model.wv.index_to_key) for o in output]
            tgt_str = [ind_to_str(o, special_token=['<sos>', '<eos>', '<pad>'], word_dict=w2v_model.wv.index_to_key) for o in tgt]

            for i in range(batch_size):
                print('out ---> ' + ' '.join(output_str[i]))
                print('tgt ---> ' + ' '.join(tgt_str[i]) + '\n')
        print(f'test mean loss {total_loss/cnt}')


        # score, output_str, ref_str = calculate_spider(output, ref, word_dict_pickle_path)
        # loss_mean = score
        # writer.add_scalar(f'Loss/eval_greddy', loss_mean, epoch)
        # msg = f'eval_greddy SPIDEr: {loss_mean:2.4f}'
        # logging.info(msg)


# if __name__ == '__main__':
EPOCH = 2
for epoch in range(1, EPOCH):
    print(f'epoch : {epoch}')
    # train(model, epoch)
    eval_all()
    torch.save(model.state_dict(), '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
    scheduler.step(epoch)