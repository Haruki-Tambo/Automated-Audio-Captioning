import torch
import torch.nn as nn
import numpy as np
from encoder import Tag
from tqdm import tqdm

class_num = 300
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
learning_rate=1e-3
model = Tag(class_num).to(device)

# DataloaderとかDatasetを処理するところ
name = 'dev'
word_label = np.load(f'data/{name}_word_label.npy')
processed_wav = np.load(f'data/{name}_processed_wav.npy')
X = torch.tensor(processed_wav, dtype=torch.float32)
y = torch.tensor(word_label, dtype=torch.float32)
train_Dataset = torch.utils.data.TensorDataset(X, y)
training_data = torch.utils.data.DataLoader(dataset=train_Dataset,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=4,
                                     drop_last=True)

name = 'eva'
word_label = np.load(f'data/{name}_word_label.npy')
processed_wav = np.load(f'data/{name}_processed_wav.npy')
X = torch.tensor(processed_wav, dtype=torch.float32)
y = torch.tensor(word_label, dtype=torch.float32)
test_Dataset = torch.utils.data.TensorDataset(X, y)
test_data = torch.utils.data.DataLoader(dataset=train_Dataset,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=4,
                                     drop_last=True)

optimizer =torch.optim.Adam(model.parameters(),
                            lr=learning_rate,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0.,
                            amsgrad=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
tag_loss = nn.BCELoss()

def train(epoch):
    # bar  = tqdm(training_data,total=len(training_data))
    loss_list = []
    model.train()
    with tqdm(training_data,total=len(training_data)) as bar:
        for i, (feature, tag) in enumerate(bar):
            feature = feature.to(device)
            tag = tag.to(device).reshape(16, -1)
            optimizer.zero_grad()
            out_tag = model(feature)
            loss = tag_loss(out_tag,tag)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch, i, np.mean(loss_list)))
    return np.mean(loss_list)

def test(epoch):
    eva_loss = []
    model.eval()
    with torch.no_grad():
        for i, (feature, tag) in enumerate(test_data):
            feature = feature.to(device)
            tag = tag.to(device).reshape(16, -1)
            out_tag = model(feature)
            loss = tag_loss(out_tag, tag)
            eva_loss.append(loss.item())
    mean_loss = np.mean(eva_loss)
    print("epoch:{:d}--testloss:{:.6f}".format(epoch,mean_loss.item()))

    # return  mean_loss


if __name__ == '__main__':
    train_b = True
    epoch_last = 0
    if train_b:
        # model.load_state_dict(torch.load("./models/280/TagModel_{}.pt".format(str(40))))
        for epoch in range(epoch_last+1,epoch_last+2):
            train(epoch)
            scheduler.step(epoch)
            test(epoch)

            torch.save(model.state_dict(), f'./TagModel_{epoch}.pt')
    else:
        for epoch in range(0,225,5):
            model.load_state_dict(torch.load(f'./models/tag_models/TagModel_{epoch}.pt'))
            test(epoch)