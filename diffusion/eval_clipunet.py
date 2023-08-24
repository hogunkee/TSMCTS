import argparse
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


def eval():
    n_T = args.n_T
    n_feat = args.n_feat
    lrate = args.lr
    n_eval = args.n_eval
    save_dir = os.path.join('data', args.out)
    model_path = os.path.join('data', args.model)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_model = True #False
    device = "cuda:0"

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
    # load the model #
    print('load model from ' + model_path)
    ddpm.load_state_dict(torch.load(model_path))
    ddpm.to(device)
    ddpm.eval()

    clip_model, _ = clip.load('ViT-L/14', device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    set_requires_grad(clip_model, False)

    if args.dataset=='tabletop-48':
        from data_loader import TabletopNpyDataset
        test_dataset = TabletopNpyDataset(data_dir=os.path.join(args.data_dir, 'test'))
        im_height = 48
        im_width = 64
    elif args.dataset=='tabletop-96':
        from data_loader import TabletopNpyDataset
        test_dataset = TabletopNpyDataset(data_dir=os.path.join(args.data_dir, 'test'))
        im_height = 96
        im_width = 128
    elif args.dataset=='ur5':
        from data_loader import UR5NpyDataset
        test_dataset = UR5NpyDataset(data_dir=os.path.join(args.data_dir, 'test'))
        im_height = 96
        im_width = 96
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)
    test_data_iterator = iter(test_dataloader)

    for ne in range(n_eval):
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        with torch.no_grad():
            n_sample = 4*2

            x_test = next(test_data_iterator)
            x_real = torch.zeros([n_sample, *x_test.shape[1:]]).to(device)
            for k in range(2):
                for j in range(int(n_sample/2)):
                    idx = k + (j*2)
                    x_real[k+(j*2)] = x_test[idx]

            x_real_clip = torch.zeros([n_sample, 3, 224, 224])
            x_real_resized = F.interpolate(x_real, size=(192, 256))
            x_real_clip[:, :, 16:-16, :] = x_real_resized[:, :, :, 16:-16]
            x_real_clip = x_real_clip.to(device)
            c_real = clip_model.encode_image(x_real_clip)
            x_gen, x_gen_store = ddpm.sample(n_sample, (3, im_height, im_width), device, c_real)

            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all, nrow=4)
            save_image(grid, os.path.join(save_dir, f"image_{ne}.png"))
            print('saved image at ' + os.path.join(save_dir, f"image_{ne}.png"))

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
            ani.save(os.path.join(save_dir, f"gif_ep{ne}.gif"), dpi=100,writer=PillowWriter(fps=5))
            print('saved image at ' + os.path.join(save_dir, f"gif_ep{ne}.gif"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_T", type=int, default=400)
    parser.add_argument("--n_feat", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_eval", type=int, default=10)
    parser.add_argument("--out", type=str, default='eval')
    parser.add_argument("--model", type=str, default='clipunet_tabletop')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=['tabletop-48', 'tabletop-96', 'ur5'],
                        default='tabletop-48')
    parser.add_argument("--data_dir", type=str, default='/disk1/hogun/tabletop_48x64')
    args = parser.parse_args()

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    eval()
