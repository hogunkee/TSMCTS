import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from tqdm import tqdm
from clip_custom.model import ModifiedResNet, VisionTransformer


def train():
    model_type = args.model
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    lrate = args.lr
    save_dir = os.path.join('data/encoder/', args.out)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_model = True #False
    device = "cuda:0"

    if args.dataset=='tabletop-48':
        from data_loader import TabletopNpyDataset
        dataset = TabletopNpyDataset(data_dir=os.path.join(args.data_dir, 'train'))
        im_height = 48
        im_width = 64
    elif args.dataset=='tabletop-96':
        from data_loader import TabletopNpyDataset
        dataset = TabletopNpyDataset(data_dir=os.path.join(args.data_dir, 'train'))
        im_height = 96
        im_width = 128
    elif args.dataset=='ur5':
        from data_loader import UR5NpyDataset
        dataset = UR5NpyDataset(data_dir=os.path.join(args.data_dir, 'train'))
        im_height = 96
        im_width = 96
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    vision_layers = 24
    embed_dim = 768
    vision_width = 1024
    image_resoultion = 224
    if model_type=='ViT':
        vision_patch_size = 14
        vision_heads = vision_width // 64
        visual_encoder = VisionTransformer(input_resolution=image_resolution,
                                           patch_size=vision_patch_size,
                                           width=vision_width,
                                           layers=vision_layers,
                                           heads=vision_heads,
                                           output_dim=embed_dim)
    elif model_type=='ResNet':
        vision_heads = vision_width * 32 // 64
        visual_encoder = ModifiedResNet(layers=vision_layers,
                                        output_dim=embed_dim,
                                        heads=vision_heads,
                                        input_resolution=image_resolution,
                                        width=vision_width)
    visual_encoder.to(device)
    optim = torch.optim.Adam(visual_encoder.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        visual_encoder.train()

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
        
            if save_mdoel and ((ep+1)%5==0 or ep == int(n_epoch-1)):
                torch.save(ddpm.state_dict(), os.path.join(save_dir, f"model_{ep}.pth"))
                print('saved model at ' + os.path.join(save_dir, f"model_{ep}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_T", type=int, default=400)
    parser.add_argument("--n_feat", type=int, default=64)
    parser.add_argument("--n_res", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out", type=str, default='clipunet_tabletop')
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

    train()

