import argparse
import copy
import os
import numpy as np
from tqdm import tqdm
from data_loader import PybulletNpyDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50


preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train(args):
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    lrate = args.lr
    save_dir = os.path.join('data', args.out)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_model = True #False
    save_freq = 5

    # dataloader #
    print("Loading data...")
    if args.dataset=='tabletop':
        dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'train'), 
                        remove_bg=args.remove_bg, label_type=args.label_type, view=args.view)
        test_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test'), 
                        remove_bg=args.remove_bg, label_type=args.label_type, view=args.view)
    elif args.dataset=='pybullet':
        dataset = PybulletNpyDataset(data_dir=os.path.join(args.data_dir, 'train'), num_duplication=5)
        test_dataset = PybulletNpyDataset(data_dir=os.path.join(args.data_dir, 'test'), num_duplication=5)
        test_dataset.fsize = 500
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # model #
    print("Create a ResNet model.")
    if args.model=='resnet-18':
        resnet = resnet18
    elif args.model=='resnet-34':
        resnet = resnet34
    elif args.model=='resnet-50':
        resnet = resnet50
    if args.finetune:
        model = resnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model_name = 'finetune'
    else:
        model = resnet(pretrained=False)
        # {view}_{remove_bg}_{label_type}_{loss}
        model_name = args.view.replace('_', '')
        model_name += 'nobg_' if args.remove_bg else 'bg_'
        model_name += args.label_type
        model_name += args.loss
    # replace the last fc layer #
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, 1),
        #nn.Sigmoid()
    )

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)

    # loss function and optimizer #
    if args.loss=='mse':
        loss_fn = nn.MSELoss()
    elif args.loss=='bce':
        loss_fn = BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
 
    # Hold the best model
    best_accuracy = - np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epoch):
        model.train()
        with tqdm(dataloader) as bar:
            bar.set_description(f"Epoch {epoch}")
            for X_batch, Y_batch in bar:
                X_batch = preprocess(X_batch).to(device)
                Y_batch = Y_batch[:, 0].to(device)
                # forward pass
                y_pred = model(X_batch)[:, 0]
                loss = loss_fn(y_pred, Y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                indices = torch.logical_or(Y_batch==0, Y_batch==1)
                acc = (y_pred.round() == Y_batch)[indices].float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )

        # evaluate accuracy at end of each epoch
        model.eval()
        matchings = []
        for X_val, Y_val in test_dataloader:
            X_val = preprocess(X_val).to(device)
            Y_val = Y_val[:, 0].to(device)
            y_pred = model(X_val)[:, 0]
            indices = torch.logical_or(Y_val==0, Y_val==1)
            matching = (y_pred.round()==Y_val)[indices].float().detach().cpu().numpy()
            matchings.append(matching)
        matchings = np.concatenate(matchings, axis=0)
        accuracy = np.mean(matchings)
        print("Validation accuracy:", accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = copy.deepcopy(model.state_dict())

        # optionally save model
        if save_model and (epoch+1)%save_freq==0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_{epoch}.pth"))
            print('saved model at ' + os.path.join(save_dir, f"{model_name}_{epoch}.pth"))

    # restore model and return best accuracy
    #model.load_state_dict(best_weights)
    if save_model:
        torch.save(best_weights, os.path.join(save_dir, f"{model_name}_best.pth"))
        print('saved model at ' + os.path.join(save_dir, f"{model_name}_best.pth"))
    return best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default='classification')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--loss", type=str, default='mse')
    # Dataset
    parser.add_argument("--dataset", type=str, default='tabletop')
    parser.add_argument("--data_dir", type=str, default='/ssd/disk/TableTidyingUp/dataset_template')
    #parser.add_argument("--data_dir", type=str, default='/ssd/disk/ur5_tidying_data/pybullet_single_bg')
    parser.add_argument("--remove_bg", action="store_true") # default: False
    parser.add_argument("--label_type", type=str, default='linspace') # linspace / binary
    parser.add_argument("--view", type=str, default='top') # top / front_top
    # etc
    parser.add_argument("--model", type=str, default='resnet-18')
    args = parser.parse_args()

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print("Training starts.")
    best_accur = train(args)
    print("Training finished.")
    print("Best accuracy:", best_accur)

