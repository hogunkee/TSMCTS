import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from typing import Dict, Tuple
from tqdm import tqdm
from ddpm import DDPM_Vision_Condition
from models.unet_clip import UNetModel
from clip_custom import clip
from data_loader import TabletopNpyDataset #TabletopDataset


def clip_image(im):
    im_clip = np.clip(im, 0, 1)
    return im_clip

def normalize_image(im):
    v_max = im.max()
    v_min = im.min()
    im_norm = (im - v_min) / (v_max - v_min)
    return im_norm

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def train_tabletop():
    # hardcoding these here
    n_epoch = 30
    batch_size = 50 #64 #256
    n_T = 400 # 500
    device = "cuda:0"
    n_feat = 64 #128 # 128 ok, 256 better (but slower)
    lrate = 2e-5 #1e-4
    save_model = True #False
    save_dir = './data/clipunet_tabletop_output_test/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    unet = UNetModel(
            in_channels=3,
            model_channels=64, #192
            out_channels=3, #6
            num_res_blocks=2, #3
            attention_resolutions=(32,16,8),
            dropout=0.1,
            num_heads=1,
            emb_condition_channels=0,
            encoder_channels=768
            )
    ddpm = DDPM_Vision_Condition(nn_model=unet, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    clip_model, _ = clip.load('ViT-L/14', device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    set_requires_grad(clip_model, False)

    data_dir = "/home/gun/ssd/disk/ur5_tidying_data/tabletop_48x64"
    dataset = TabletopNpyDataset(data_dir=os.path.join(data_dir, 'train'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    test_dataset = TabletopNpyDataset(data_dir=os.path.join(data_dir, 'test'))
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        #for x, c in pbar:
        for x in pbar:
            optim.zero_grad()
            # context #
            # resize & crop, pad
            x_clip = torch.zeros([x.shape[0], 3, 224, 224])
            x_resized = F.interpolate(x, size=(192, 256))
            x_clip[:, :, 16:-16, :] = x_resized[:, :, :, 16:-16]
            x_clip = x_clip.to(device)
            c = clip_model.encode_image(x_clip)
            c = c.to(device)

            x = x.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*2

            x_test = next(iter(test_dataloader))
            x_real = torch.zeros([n_sample, *x.shape[1:]]).to(device)
            for k in range(2):
                for j in range(int(n_sample/2)):
                    idx = k + (j*2)
                    x_real[k+(j*2)] = x_test[idx]

            x_real_clip = torch.zeros([n_sample, 3, 224, 224])
            x_real_resized = F.interpolate(x_real, size=(192, 256))
            x_real_clip[:, :, 16:-16, :] = x_real_resized[:, :, :, 16:-16]
            x_real_clip = x_real_clip.to(device)
            c_real = clip_model.encode_image(x_real_clip)
            x_gen, x_gen_store = ddpm.sample(n_sample, (3, 48, 64), device, c_real)

            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all, nrow=4)
            save_image(grid, save_dir + f"image_ep{ep}.png")
            print('saved image at ' + save_dir + f"image_ep{ep}.png")

            if ep%5==0 or ep == int(n_epoch-1):
                # create gif of images evolving over time, based on x_gen_store
                fig, axs = plt.subplots(ncols=int(n_sample/2), nrows=2,\
                                        sharex=True,sharey=True,figsize=(8,3))
                def animate_diff(i, x_gen_store):
                    print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    plots = []
                    x_gen_clip = clip_image(x_gen_store)
                    #x_gen_norm = normalize_image(x_gen_store)
                    for row in range(2):
                        for col in range(int(n_sample/2)):
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            plots.append(axs[row, col].imshow(x_gen_clip[i,(row*2)+col].transpose([1,2,0])))
                            #plots.append(axs[row, col].imshow(x_gen_norm[i,(row*2)+col].transpose([1,2,0])))
                    return plots
                ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                ani.save(save_dir + f"gif_ep{ep}.gif", dpi=100,writer=PillowWriter(fps=5))
                print('saved image at ' + save_dir + f"gif_ep{ep}.gif")

        # optionally save model
        if save_model:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train_tabletop()

