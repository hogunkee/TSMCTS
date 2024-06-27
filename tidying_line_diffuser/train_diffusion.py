import os
import shutil
import argparse
import datetime

import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.transforms import Resize, Pad

from datasets.tabletop_datasets import TabletopDiffusionDataset
#from datasets.transform import Transform
from models import Encoder, Decoder, ConditionalDiffusion, AttentionMask

import wandb
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pth', type=str, default='data/')
    parser.add_argument('--data_dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--val_data_dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/test-unseen_obj-unseen_template')
    parser.add_argument('--ckpt_dir', type=str, default='/home/gun/ssd/disk/PreferenceDiffusion/tidying-line-diffusion')
    parser.add_argument('--encoder_pth', type=str, default='0626_1752')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=16) #64
    parser.add_argument('--n_timesteps', type=int, default=1000)
    parser.add_argument('--n_masks', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--updates_per_epoch', type=int, default=10000)
    parser.add_argument('--validation_steps', type=int, default=10)
    parser.add_argument('--remove_bg', action='store_true')
    parser.add_argument('--wandb_off', action='store_true')
    args = parser.parse_args()

    now = datetime.datetime.now()
    exp_name = 'diffusion_%s' %(now.strftime("%m%d_%H%M"))
    log_dir = os.path.join('logs', exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    wandb_off = args.wandb_off
    if not wandb_off:
        wandb.init(project="TabletopDiffusion")
        wandb.run.name = exp_name
        wandb.config.update(args)
        wandb.run.save()
    #writer = SummaryWriter(log_dir)
    checkpoint_dir = os.path.join(args.ckpt_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = TabletopDiffusionDataset(args.data_dir, args.remove_bg)
    val_dataset = TabletopDiffusionDataset(args.val_data_dir, args.remove_bg)
    train_data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
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

    #transform = Transform()
    transform = transforms.Compose([Resize([96, 128]), Pad([0, 16, 0, 16])])
    resize = Resize((128, 128))

    epoch, n_updates, min_val_loss = 0, 0, 10000
    total_updates = args.epochs * args.updates_per_epoch
    pbar = tqdm(total=args.updates_per_epoch, desc='Epoch 0')
    while n_updates < total_updates:
        for batch in train_data_loader:
            x, _ = batch
            x = transform(x.transpose(2, 3).transpose(1, 2)).to(device)
            with torch.no_grad():
                posterior, _ = encoder(x, compute_loss=False)
                feature = posterior.mean
                assert torch.max(feature) <= 1.
            # mask = attention_mask(feature)
            cond = torch.zeros_like(feature)
            for i in range(args.n_masks):
                x_i = np.random.randint(0, 12, size=1)
                y_i = np.random.randint(0, 16, size=1)
                cond[:, :, x_i, y_i] = feature[:, :, x_i, y_i]
                # cond_ = (feature * mask[:, i, :, :].unsqueeze(1))
                # cond_ = mask[:, i, :, :].unsqueeze(1) * torch.sum(cond_, dim=(2, 3), keepdim=True)
                # cond += cond_
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
                        x,_ = batch
                        x = transform(x.transpose(2, 3).transpose(1, 2)).to(device)
                        posterior, _ = encoder(x, compute_loss=False)
                        feature = posterior.mean
                        # mask = attention_mask(feature)
                        cond = torch.zeros_like(feature)
                        mask = torch.zeros_like(feature[0, :3, :, :])
                        for i in range(args.n_masks):
                            x_i, y_i = np.random.randint(0, 16, size=2)
                            mask[:, x_i, y_i] = 1.
                            cond[:, :, x_i, y_i] = feature[:, :, x_i, y_i]
                            # cond_ = (feature * mask[:, i, :, :].unsqueeze(1))
                            # cond_ = mask[:, i, :, :].unsqueeze(1) * torch.sum(cond_, dim=(2, 3), keepdim=True)
                            # cond += cond_
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
                torch.save(state_dict, os.path.join(checkpoint_dir, 'checkpoint-%d.pt'%epoch))
                if validation_loss < min_val_loss:
                    min_val_loss = validation_loss
                    torch.save(state_dict, os.path.join(checkpoint_dir, 'checkpoint_best.pt'))
                pbar = tqdm(total=args.updates_per_epoch, desc='Epoch %d' % epoch)

            n_updates += 1
            if n_updates == total_updates:
                break


if __name__ == '__main__':
    train()
