import argparse

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data_loader import UR5Dataset

from model import dsae


def draw_spatial_features(numpy_image, features, image_size=(28, 28)):
    image_size_x, image_size_y = image_size
    for sp in features:
        x, y = sp
        attend_x_pix = int((x + 1) * (image_size_x - 1) / 2)
        attend_y_pix = int((y + 1) * (image_size_y - 1) / 2)
        numpy_image[attend_y_pix, attend_x_pix] = np.array([1.0, 0.0, 0.0])


def draw_figure(filename, num_images_to_draw, spatial_features_to_draw, images_to_draw, reconstructed_images_to_draw):
    f, axarr = plt.subplots(num_images_to_draw, 2, figsize=(10, 15), dpi=100)
    plt.tight_layout()
    for idx, im in enumerate(reconstructed_images_to_draw[:num_images_to_draw]):
        # original image
        og_image = (images_to_draw[:num_images_to_draw][idx] + 1) / 2
        draw_spatial_features(og_image, spatial_features_to_draw[idx])
        axarr[idx, 0].imshow(og_image)
        # reconstructed image
        scaled_image = (im + 1) / 2
        axarr[idx, 1].imshow(scaled_image.cpu().numpy().reshape(96, 96, 3))

    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()

    # parameters and miscellaneous
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    # Adam learning rate
    lr = args.learning_rate
    out_file_name = os.path.join('temp/', args.file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_dataset = UR5Dataset()
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    # can change these two with your own
    #encoder = dsae.CustomEncoder(in_channels=1, out_channels=(4, 8, 8), normalise=True)
    # image output size can be smaller, not worth it for MNIST images
    #decoder = dsae.CustomDecoder(image_output_size=(28, 28), latent_dimension=16)

    dsae_model = dsae.CustomDeepSpatialAutoencoder(in_channels=3, 
                                                   hidden_dims=[32, 64, 128, 32], 
                                                   latent_dimension=64,
                                                   latent_height=6,
                                                   latent_width=6,
                                                   out_channels=3, 
                                                   temperature=None, 
                                                   normalise=True
                                                   ).to(device)

    optimiser = torch.optim.Adam(dsae_model.parameters(), lr=lr)
    # g_slow does not make sense for non-sequence data such as MNIST
    #dsae_loss = dsae.DSAE_Loss(add_g_slow=False)
    rec_loss = nn.BCELoss(reduce='sum')

    for epoch in range(num_epochs):
        dsae_model.train()
        for batch_idx, (images, _depths, _poses) in enumerate(train_loader):
            images = images.to(device)
            optimiser.zero_grad()
            output = dsae_model(images)
            # we ignore g_slow contribution for MNIST
            #loss, _g_slow_contrib = dsae_loss(output, images)
            loss = rec_loss(output, images)
            loss = loss / len(images)
            loss.backward()
            optimiser.step()
            if batch_idx % 30 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()
                    )
                )

            spatial_features = dsae_model.encoder(images)
            num_images = 5
            _file_name = out_file_name + '_train_%dep' %epoch
            draw_figure(out_file_name, num_images, spatial_features, images, output)

    dsae_model.eval()
    with torch.no_grad():
        images, _depths, _poses = next(iter(train_loader)) # test_loader
        images = images.to(device)
        recon = dsae_model(images)
        spatial_features = dsae_model.encoder(images)
        num_images = 5
        draw_figure(out_file_name, num_images, spatial_features, images, recon)

    torch.save(dsae_model.state_dict(), out_file_name + '.pth')
    print('Training done.')
