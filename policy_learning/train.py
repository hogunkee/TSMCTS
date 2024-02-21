import numpy as np
import nonechucks as nc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from custom_dataset import TabletopOfflineDataset
from model import PlaceNet

Device = 'cuda'

def train(args):
    dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size)
    dataset = nc.SafeDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    pnet = PlaceNet(args.hidden_dim).to(Device)
    if args.loss=='ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss=='mse':
        criterion = nn.MSELoss()
    elif args.loss=='sum':
        criterion = None
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

            if args.loss=='sum':
                loss = - (probs * labels).sum()
            else:
                probs_flatten = probs.view(-1, 10 * 13)
                labels_flatten = labels.view(-1, 10 * 13)
                loss = criterion(probs_flatten, labels_flatten)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if (i+1) % args.log_freq == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.log_freq))
                running_loss = 0.0
                if False:
                    plt.subplot(1, 3, 1)
                    plt.imshow(probs.cpu().detach().numpy()[0], vmin=0., vmax=1.)
                    plt.subplot(1, 3, 2)
                    plt.imshow(labels.cpu().detach().numpy()[0], vmin=0., vmax=1.)
                    plt.subplot(1, 3, 3)
                    plt.imshow((probs * labels).cpu().detach().numpy()[0], vmin=0., vmax=1.)
                    plt.show()
        torch.save(pnet.state_dict(), os.path.join(args.log_dir, 'pnet_e%d.pth'%(epoch+1)))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='ce') # mse / sum
    parser.add_argument('--data-dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    train(args)
    print('Finished Training')
