import os
import shutil
import argparse

import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torchvision.transforms import Resize

from datasets.datasets import DiffusionDataset
from datasets.transform import Transform
from models import Encoder, Decoder

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pth', type=str, default='/data/codes/tidying_line/train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--updates_per_epoch', type=int, default=10000)
    parser.add_argument('--validation_steps', type=int, default=10)
    args = parser.parse_args()

    exp_name = 'encoder'
    log_dir = os.path.join('logs', exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    data = np.load(os.path.join(args.data_pth, 'resized.npy'))
    dataset = DiffusionDataset(data)
    val_data_size = int(0.05 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, (len(dataset) - val_data_size, val_data_size))
    train_data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_data_loader = DataLoader(val_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    device = torch.device('cuda')

    encoder = Encoder(output_dim=args.latent_dim).to(device)
    decoder = Decoder(input_dim=args.latent_dim).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

    transform = Transform()

    epoch, n_updates, min_val_loss = 0, 0, 10000
    total_updates = args.epochs * args.updates_per_epoch
    pbar = tqdm(total=args.updates_per_epoch, desc='Epoch 0')
    while n_updates < total_updates:
        for batch in train_data_loader:
            x = batch.to(device)
            x = transform(x.transpose(2, 3).transpose(1, 2))
            posterior, prior_loss = encoder(x)
            z = posterior.rsample()
            x_recon = decoder(z)
            recon_loss = (x_recon - x).pow(2).mean()

            loss = recon_loss + args.beta * prior_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            pbar.update(1)

            writer.add_scalar('encoder/train_loss', loss, n_updates)
            writer.add_scalar('encoder/train_recon_loss', recon_loss, n_updates)
            writer.add_scalar('encoder/train_prior_loss', prior_loss, n_updates)

            if n_updates % args.updates_per_epoch == 0:
                pbar.close()
                epoch += 1
                with torch.no_grad():
                    img = make_grid(torch.cat([x[:8], x_recon[:8]], dim=0), normalize=True, range=(-1, 1))
                    writer.add_image('train/img', img, n_updates)

                    validation_losses = []
                    for batch in val_data_loader:
                        x = batch.to(device)
                        x = transform(x.transpose(2, 3).transpose(1, 2))
                        posterior, prior_loss = encoder(x)
                        z = posterior.rsample()
                        x_recon = decoder(z)
                        recon_loss = (x_recon - x).pow(2).mean()

                        loss = recon_loss + args.beta * prior_loss
                        validation_losses.append(loss.item())
                        if len(validation_losses) > args.validation_steps:
                            break
                img = make_grid(torch.cat([x[:8], x_recon[:8]], dim=0), normalize=True, range=(-1, 1))
                writer.add_image('val/img', img, n_updates)
                validation_loss = np.mean(validation_losses)
                writer.add_scalar('diffusion/val_loss', validation_loss, n_updates)

                state_dict = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state_dict, os.path.join(checkpoint_dir, 'checkpoint.pt'))
                if validation_loss < min_val_loss:
                    min_val_loss = validation_loss
                    torch.save(state_dict, os.path.join(checkpoint_dir, 'checkpoint_best.pt'))
                pbar = tqdm(total=args.updates_per_epoch, desc='Epoch %d' % epoch)

            n_updates += 1
            if n_updates == total_updates:
                break


if __name__ == '__main__':
    train()