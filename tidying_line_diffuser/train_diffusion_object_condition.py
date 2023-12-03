import os
import shutil
import argparse
import datetime

import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torchvision.transforms import Resize

from datasets.datasets import CondDiffusionDataset
from datasets.transform import Transform
from models import Encoder, Decoder, ConditionalDiffusion, AttentionMask

import wandb
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pth', type=str, default='data/')
    #'/data/codes/tidying_line/train')
    parser.add_argument('--ckpt_dir', type=str, default='/home/gun/ssd/disk/PreferenceDiffusion/tidying-line-diffusion')
    parser.add_argument('--encoder_pth', type=str, default='1129_1701')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=16) #64
    parser.add_argument('--n_timesteps', type=int, default=1000)
    parser.add_argument('--n_masks', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--updates_per_epoch', type=int, default=10000)
    parser.add_argument('--validation_steps', type=int, default=10)
    parser.add_argument('--wandb_off', action='store_true')
    args = parser.parse_args()

    now = datetime.datetime.now()
    exp_name = 'diffusion_%s' %(now.strftime("%m%d_%H%M"))
    log_dir = os.path.join('logs', exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    wandb_off = args.wandb_off
    if not wandb_off:
        wandb.init(project="instruct-pix2pix")
        wandb.run.name = exp_name
        wandb.config.update(args)
        wandb.run.save()
    #writer = SummaryWriter(log_dir)
    checkpoint_dir = os.path.join(args.ckpt_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    rgb_data = np.load(os.path.join(args.data_pth, 'rgb_128.npy'))
    segmap_data = np.load(os.path.join(args.data_pth, 'segmap_16.npy'))
    dataset = CondDiffusionDataset(rgb_data, segmap_data)
    val_data_size = int(0.05 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, (len(dataset) - val_data_size, val_data_size))
    train_data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_data_loader = DataLoader(val_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    device = torch.device('cuda')
    diffusion = ConditionalDiffusion(input_dim=args.latent_dim, cond_dim=args.latent_dim, n_timesteps=args.n_timesteps).to(device)
    # attention_mask = AttentionMask(input_dim=args.latent_dim, output_dim=args.n_masks).to(device)
    # params = list(diffusion.parameters()) + list(attention_mask.parameters())
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=3e-5)

    encoder = Encoder(output_dim=args.latent_dim).to(device)
    decoder = Decoder(input_dim=args.latent_dim).to(device)
    state_dict = torch.load(os.path.join(args.ckpt_dir, 'encoder_%s/checkpoint_best.pt' %args.encoder_pth))
    encoder.load_state_dict(state_dict['encoder'])
    decoder.load_state_dict(state_dict['decoder'])

    transform = Transform()
    resize = Resize((128, 128))

    epoch, n_updates, min_val_loss = 0, 0, 10000
    total_updates = args.epochs * args.updates_per_epoch
    pbar = tqdm(total=args.updates_per_epoch, desc='Epoch 0')
    while n_updates < total_updates:
        for batch in train_data_loader:
            x, masks = batch
            x = x.to(device)
            masks = masks.to(device)
            x = transform(x.transpose(2, 3).transpose(1, 2))
            with torch.no_grad():
                posterior, _ = encoder(x, compute_loss=False)
                feature = posterior.mean
                assert torch.max(feature) <= 1.

            cond = (masks != 0).to(torch.float32).view(-1, 1, 16, 16) * feature
            # cond = torch.zeros_like(feature)
            # b, y, x = torch.where(masks != 0)
            # for i in range(len(b)):
            #     cond[b[i], :, y[i], x[i]] = feature[b[i], :, y[i], x[i]]
            loss = diffusion.loss(feature, cond)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.update(1)

            if not wandb_off:
                eplog = {'diffusion/train_loss': loss}
                wandb.log(eplog, n_updates)
                #writer.add_scalar('diffusion/train_loss', loss, n_updates)

            if n_updates % args.updates_per_epoch == 0:
                pbar.close()
                epoch += 1
                with torch.no_grad():
                    validation_losses = []
                    for batch in val_data_loader:
                        x, masks = batch
                        x = x.to(device)
                        masks = masks.to(device)
                        x = transform(x.transpose(2, 3).transpose(1, 2))
                        posterior, _ = encoder(x, compute_loss=False)
                        feature = posterior.mean

                        cond = (masks != 0).to(torch.float32).view(-1, 1, 16, 16) * feature
                        mask = (masks != 0).to(torch.float32).view(-1, 1, 16, 16).repeat(1, 3, 1, 1)
                        loss = diffusion.loss(feature, cond)
                        validation_losses.append(loss.item())
                        if len(validation_losses) > args.validation_steps:
                            break
                validation_loss = np.mean(validation_losses)
                with torch.no_grad():
                    feature_recon = diffusion(cond[:1])
                    img_recon = decoder(feature_recon)
                    img = decoder(feature[:1])
                img = make_grid(torch.cat([img, img_recon], dim=0), normalize=True, range=(-1, 1))
                mask_img = resize(mask)
                if not wandb_off:
                    eplog = {
                            'diffusion/val_loss': validation_loss,
                            'diffusion/img': wandb.Image(img),
                            'diffusion/mask': wandb.Image(mask_img),
                            }
                    wandb.log(eplog, n_updates)
                    #writer.add_scalar('diffusion/val_loss', validation_loss, n_updates)
                    #writer.add_image('diffusion/img', img, n_updates)
                    #writer.add_image('diffusion/mask', mask_img, n_updates)

                state_dict = {
                    'diffusion': diffusion.state_dict(),
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'attention_mask': attention_mask.state_dict(),
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
