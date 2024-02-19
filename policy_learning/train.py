import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from argparse import ArgumentParser
from custom_dataset import TabletopOfflineDataset
from model import PlaceNet

Device = 'cuda'

def train(args):
    dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    pnet = PlaceNet(args.hidden_dim).to(Device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pnet.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs = data['image_after_pick'].permute([0,3,1,2]).to(torch.float32).to(Device)
            actions = data['action']
            labels = np.zeros([len(actions), 10, 13])
            labels[np.arange(len(actions)), actions[:, 0], actions[:, 1]] = 1
            labels = torch.Tensor(labels).to(torch.float32).to(Device)

            probs = pnet(inputs)
            probs_flatten = probs.view(-1, 1)
            labels_flatten = labels.view(-1, 1)
            print('probs:', probs.shape)
            print('labels:', labels.shape)
            loss = criterion(probs_flatten, labels_flatten)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if (i-1) % args.log_freq == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.log_freq))
                running_loss = 0.0
        torch.save(pnet.state_dict(), os.path.join(args.log_dir, 'pnet_e%d.pth'%(epoch+1)))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--data-dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--log-freq', type=int, default=1000)
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    train(args)
    print('Finished Training')
