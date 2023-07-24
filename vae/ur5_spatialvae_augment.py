import argparse

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
from torch import nn
from data_loader import UR5Dataset

from model import spatial_vae as svae


def draw_spatial_features(numpy_image, poses, image_size=(96, 96), normalised=False):
    image_size_x, image_size_y = image_size
    colors = np.array([
        [1., 0., 0.],
        [0., 0.6, 0.],
        [0., 0., 1.]
        ])
    for c, p in zip(colors, poses):
        x, y = p
        if normalised:
            attend_x_pix = int((x + 1) * (image_size_x - 1) / 2)
            attend_y_pix = int((y + 1) * (image_size_y - 1) / 2)
            numpy_image[attend_y_pix, attend_x_pix] = c
        else:
            numpy_image[int(y), int(x)] = c


def draw_figure(filename, num_images_to_draw, images_to_draw, reconstructed_images_to_draw, 
        poses, reconstructed_sample_images, poses_sample):
    f, axarr = plt.subplots(num_images_to_draw, 3, figsize=(12, 15), dpi=100)
    plt.tight_layout()
    for idx, im in enumerate(reconstructed_images_to_draw[:num_images_to_draw]):
        # original image
        og_image = (images_to_draw[:num_images_to_draw][idx] + 1) / 2
        og_image = og_image.detach().cpu().numpy().transpose([1, 2, 0])
        axarr[idx, 0].imshow(og_image)
        # reconstructed image
        scaled_image = (im.detach().cpu().numpy().transpose([1, 2, 0]) + 1) / 2
        draw_spatial_features(scaled_image, poses[idx])
        axarr[idx, 1].imshow(scaled_image)
        # reconstruct with sampled points
        im_sample = reconstructed_sample_images[idx]
        scaled_image_sample = (im_sample.detach().cpu().numpy().transpose([1, 2, 0]) + 1) / 2
        draw_spatial_features(scaled_image_sample, poses_sample[idx])
        axarr[idx, 2].imshow(scaled_image_sample)

    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--data_dir", default='/home/gun/ssd/disk/ur5_tidying_data/3block', type=str)
    args = parser.parse_args()

    # parameters and miscellaneous
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    # Adam learning rate
    lr = args.learning_rate
    out_file_name = os.path.join('temp/', args.file_name)

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) is visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dataset = UR5Dataset(data_dir=args.data_dir, augmentation=True)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    svae_model = svae.CustomDeepSpatialAutoencoder(in_channels=3, 
                                                   hidden_dims=[32, 64, 128], #[32, 64, 16]
                                                   latent_dimension=390, #64
                                                   latent_height=12, #6
                                                   latent_width=12, #6
                                                   out_channels=3, 
                                                   ).to(device)

    optimiser = torch.optim.Adam(svae_model.parameters(), lr=lr)
    # g_slow does not make sense for non-sequence data such as MNIST
    svae_loss = svae.SVAE_Loss()
    #rec_loss = nn.BCELoss(reduction='sum')

    for epoch in range(num_epochs):
        svae_model.train()
        for batch_idx, (images, poses, images_prime, poses_prime) in enumerate(train_loader):
            images = images.to(device)
            optimiser.zero_grad()
            spatial_features, poses_norm = svae_model.encoder(images, poses)
            output = svae_model.decoder(spatial_features, poses_prime.to(device), normalise=True)

            loss = svae_loss(output, images_prime.to(device))
            loss = loss / len(images)
            loss.backward()
            optimiser.step()
            if batch_idx % 300 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()
                    )
                )

        recon_og = svae_model.decoder(spatial_features, poses.to(device), normalise=True)
        num_images = 5
        _file_name = out_file_name + '_train_%dep' %epoch
        draw_figure(_file_name, num_images, images, recon_og, poses, output, poses_prime)
        torch.save(svae_model.state_dict(), out_file_name + '_%dep.pth'%epoch)


    svae_model.eval()
    with torch.no_grad():
        images, poses, images_prime, poses_prime = next(iter(train_loader)) # test_loader
        images = images.to(device)
        spatial_features, poses_norm = svae_model.encoder(images, poses)
        recon = svae_model.decoder(spatial_features, poses.to(device), normalise=True)
        recon_prime = svae_model.decoder(spatial_features, poses_prime.to(device), normalise=True)

        num_images = 5
        draw_figure(out_file_name, num_images, images, recon, poses, recon_prime, poses_prime)

    torch.save(svae_model.state_dict(), out_file_name + '.pth')
    print('Training done.')
