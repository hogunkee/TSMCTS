import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from typing import Dict, Tuple
from tqdm import tqdm
from ddpm import DDPM
from models.context_unet import ContextUnet
from data_loader import UR5Dataset


def clip_image(im):
    im_clip = np.clip(im, 0, 1)
    return im_clip

def normalize_image(im):
    v_max = im.max()
    v_min = im.min()
    im_norm = (im - v_min) / (v_max - v_min)
    return im_norm


def train_ur5():

    # hardcoding these here
    n_epoch = 30
    batch_size = 64 #256
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 2
    n_feat = 64 #128 # 128 ok, 256 better (but slower)
    lrate = 2e-5 #1e-4
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    save_model = True #False
    save_dir = './data/ur5_outputs2/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) 

    dataset = UR5Dataset(data_dir="/home/gun/ssd/disk/ur5_tidying_data/3blocks_align_ng/")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        #for x, c in pbar:
        for x, _d, _p in pbar:
            c = torch.zeros(x.size()[0]).type(torch.int64)
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
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
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (3, 96, 96), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all, nrow=4)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(ncols=int(n_sample/n_classes), nrows=n_classes,\
                                            sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        x_gen_clip = clip_image(x_gen_store)
                        #x_gen_norm = normalize_image(x_gen_store)
                        for row in range(n_classes):
                            for col in range(int(n_sample/n_classes)):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(axs[row, col].imshow(x_gen_clip[i,(row*n_classes)+col].transpose([1,2,0])))
                                #plots.append(axs[row, col].imshow(x_gen_norm[i,(row*n_classes)+col].transpose([1,2,0])))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100,writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")

        # optionally save model
        if save_model:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train_ur5()

