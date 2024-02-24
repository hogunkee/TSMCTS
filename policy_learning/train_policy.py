import os
import time
import numpy as np
import nonechucks as nc
import datetime
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from custom_dataset import TabletopOfflineDataset
from model import PlaceNet, ResNet
from torchvision import transforms
from torchvision.models import resnet18

Device = 'cuda'

def loadRewardFunction(model_path):
    resnet = resnet18
    vNet = resnet(pretrained=False)
    fc_in_features = vNet.fc.in_features
    vNet.fc = nn.Sequential(
        nn.Linear(fc_in_features, 1),
    )
    vNet.load_state_dict(torch.load(model_path))
    vNet.to(Device)
    vNet.eval()
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return vNet, preprocess

def evaluateBatch(data, pnet, vNet, preprocess, H, W, cropSize):
    inputs = preprocess(data['image_after_pick'].permute([0,3,1,2])).to(torch.float32).to(Device)
    probs = pnet(inputs).cpu().detach().numpy()
    gt_actions = data['action'].cpu().detach().numpy()

    imageSize = np.array([360, 480])
    tableSize = np.array([H, W])
    ratio = imageSize // tableSize
    offset = (imageSize - ratio * tableSize + ratio)//2

    images = data['image_after_pick'].detach().numpy()
    patches = data['next_patch'].detach().numpy()
    next_images = data['next_image'].detach().numpy()
    images_after_action = []
    for b in range(len(images)):
        image = images[b]
        patch = patches[b]
        action_prob = probs[b]
        gt_action = gt_actions[b]

        mask = np.zeros([cropSize, cropSize])
        mask[patch.sum(2) > 0] = 1
        next_image = next_images[b]

        image_after_action = np.copy(image)
        index = np.argmax(action_prob)
        py, px = index // W, index % W
        ty, tx = np.array([py, px]) * ratio + offset
        yMin = int(ty - cropSize / 2)
        yMax = int(ty + cropSize / 2)
        xMin = int(tx - cropSize / 2)
        xMax = int(tx + cropSize / 2)
        image_after_action[
                max(0, yMin): min(imageSize[0], yMax),
                max(0, xMin): min(imageSize[1], xMax)
        ] += (patch * mask[:, :, None])[
                            max(0, -yMin): max(0, -yMin) + (min(imageSize[0], yMax) - max(0, yMin)),
                            max(0, -xMin): max(0, -xMin) + (min(imageSize[1], xMax) - max(0, xMin)),
                            ]
        images_after_action.append(image_after_action)
        if False:
            plt.subplot(2, 3, 1)
            plt.imshow(patch)
            plt.subplot(2, 3, 2)
            plt.imshow(action_prob)
            plt.subplot(2, 3, 3)
            gt_action_prob = np.zeros_like(action_prob)
            gt_action_prob[gt_action[0], gt_action[1]] = 1
            plt.imshow(gt_action_prob)
            #plt.imshow(action_prob==action_prob.max())
            plt.subplot(2, 3, 4)
            plt.imshow(image)
            plt.subplot(2, 3, 5)
            plt.imshow(image_after_action)
            plt.subplot(2, 3, 6)
            plt.imshow(next_image)
            plt.show()
    images_after_action = np.concatenate(images_after_action, 0).reshape([-1, 360, 480, 3])
    
    s = preprocess(torch.Tensor(next_images).permute([0,3,1,2])).cuda()
    rewards = vNet(s).cpu().detach().numpy()

    s_prime = preprocess(torch.Tensor(images_after_action).permute([0,3,1,2])).cuda()
    rewards_prime = vNet(s_prime).cpu().detach().numpy()
    return (rewards_prime - rewards).mean()


def train(args, log_name):
    if args.model=='resnet':
        H, W = 12, 15
    else:
        H, W = 10, 13
    train_dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size, H=H, W=W)
    test_dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size, H=H, W=W)
    # train_size = int(0.9 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    indices_train = np.arange(len(train_dataset)).tolist()
    train_dataset.data_rewards = [train_dataset.data_rewards[d] for d in indices_train]
    train_dataset.data_terminals = [train_dataset.data_terminals[d] for d in indices_train]
    train_dataset.data_next_images = [train_dataset.data_next_images[d] for d in indices_train]
    train_dataset.data_images = [train_dataset.data_images[d] for d in indices_train]
    train_dataset.data_next_segs = [train_dataset.data_next_segs[d] for d in indices_train]
    train_dataset.data_segs = [train_dataset.data_segs[d] for d in indices_train]
    train_dataset.data_next_obj_infos = [train_dataset.data_next_obj_infos[d] for d in indices_train]
    train_dataset.data_obj_infos = [train_dataset.data_obj_infos[d] for d in indices_train]

    indices_test = np.arange(len(test_dataset))[::4][-1000:].tolist()
    test_dataset.data_rewards = [test_dataset.data_rewards[d] for d in indices_test]
    test_dataset.data_terminals = [test_dataset.data_terminals[d] for d in indices_test]
    test_dataset.data_next_images = [test_dataset.data_next_images[d] for d in indices_test]
    test_dataset.data_images = [test_dataset.data_images[d] for d in indices_test]
    test_dataset.data_next_segs = [test_dataset.data_next_segs[d] for d in indices_test]
    test_dataset.data_segs = [test_dataset.data_segs[d] for d in indices_test]
    test_dataset.data_next_obj_infos = [test_dataset.data_next_obj_infos[d] for d in indices_test]
    test_dataset.data_obj_infos = [test_dataset.data_obj_infos[d] for d in indices_test]
    
    train_dataset, test_dataset = nc.SafeDataset(train_dataset), nc.SafeDataset(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Train data:", len(train_dataset))
    print("Test data:", len(test_dataset))
    print("-"*60)

    if args.model=='resnet':
        pnet = ResNet().to(Device)
    elif args.model=='pnet':
        pnet = PlaceNet(args.hidden_dim).to(Device)

    if args.loss=='ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss=='mse':
        criterion = nn.MSELoss()
    elif args.loss=='sum':
        criterion = None
    optimizer = optim.Adam(pnet.parameters(), lr=args.learning_rate)

    vNet, preprocess = loadRewardFunction(args.reward_model_path)

    count_steps = 0
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        with tqdm(train_loader) as bar:
            bar.set_description(f"Epoch {epoch + 1}")
            for i, data in enumerate(bar, 0):
                inputs = preprocess(data['image_after_pick'].permute([0,3,1,2])).to(torch.float32).to(Device)
                actions = data['action']
                labels = np.zeros([len(actions), H, W])
                labels[np.arange(len(actions)), actions[:, 0], actions[:, 1]] = 1
                labels = torch.Tensor(labels).to(torch.float32).to(Device)
                probs = pnet(inputs)

                if args.loss=='sum':
                    loss = - (probs * labels).sum()
                else:
                    probs_flatten = probs.view(-1, H * W)
                    labels_flatten = labels.view(-1, H * W)
                    loss = criterion(probs_flatten, labels_flatten)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(
                    loss=float(loss)
                )
                if not args.wandb_off:
                    step_log = {'loss': float(loss)}
                    wandb.log(step_log, count_steps)
                
                running_loss += loss.item()
                if (i+1) % args.log_freq == 0:
                    print('%d steps / loss: %.3f' %
                        (i + 1, running_loss / args.log_freq))
                    if False:
                        plt.subplot(1, 3, 1)
                        plt.imshow(probs.cpu().detach().numpy()[0], vmin=0., vmax=1.)
                        plt.subplot(1, 3, 2)
                        plt.imshow(labels.cpu().detach().numpy()[0], vmin=0., vmax=1.)
                        plt.subplot(1, 3, 3)
                        plt.imshow((probs * labels).cpu().detach().numpy()[0], vmin=0., vmax=1.)
                        plt.show()
                    
                    delta_rewards = []
                    for t_data in test_loader:
                        delta_reward = evaluateBatch(t_data, pnet, vNet, preprocess, H, W, args.crop_size)
                        delta_rewards.append(delta_reward)
                    print('rewards:', np.mean(delta_rewards))
                    if not args.wandb_off:
                        step_log = {
                                'running loss': float(running_loss / args.log_freq),
                                'delta rewards': float(np.mean(delta_rewards)),
                                }
                        wandb.log(step_log, count_steps)
                    running_loss = 0.0
                count_steps += 1
        if not os.path.isdir(os.path.join(args.log_dir, log_name)):
            os.makedirs(os.path.join(args.log_dir, log_name))
        torch.save(pnet.state_dict(), os.path.join(args.log_dir, log_name, 'pnet_e%d.pth'%(epoch+1)))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='ce') # mse / sum / ce:cross-entropy
    parser.add_argument('--model', type=str, default='resnet') # pnet / resnet
    parser.add_argument('--reward-model-path', type=str, default='../mcts/data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--data-dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--wandb-off', action='store_true')
    args = parser.parse_args()

    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    if not args.wandb_off:
        wandb.init(project="Policy learning")
        wandb.config.update(parser.parse_args())
        wandb.run.name = log_name
        wandb.run.save()

    train(args, log_name)
    print('Finished Training')
