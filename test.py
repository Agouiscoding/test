import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from tqdm import tqdm

DEFAULT_CONFIG = {
    'n_epochs': 200,
    'batch_size': 64,
    'lr': 0.0002,
    'latent_dim': 100,
    'save_interval': 500,
    'save_path': 'images',
}


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.shape[0], -1)
        score = self.model(img)
        return score


def train(config, dataloader, discriminator, generator, optimizer_G, optimizer_D):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()

    for epoch in tqdm(range(config['n_epochs']), desc="Training Epochs"):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            # Soft label smoothing
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # 平滑标签 0.9 而非 1.0
            fake_labels = torch.zeros((batch_size, 1), device=device)

            real_imgs = imgs.to(device)
            z = torch.randn(batch_size, config['latent_dim'], device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            # Real image loss
            real_loss = criterion(discriminator(real_imgs), real_labels)
            # Fake image loss
            gen_imgs = generator(z)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            # Generator loss: fool discriminator
            g_loss = criterion(discriminator(gen_imgs), real_labels)  # 目标是让生成图片被判为真实
            g_loss.backward()
            optimizer_G.step()

            # Save Images
            batches_done = epoch * len(dataloader) + i
            if batches_done % config['save_interval'] == 0:
                save_image(gen_imgs[:25],
                           os.path.join(config['save_path'], f"{batches_done}.png"),
                           nrow=5, normalize=True, value_range=(-1, 1))




   


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output image directory
    os.makedirs(config['save_path'], exist_ok=True)

    # Load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=config['batch_size'], shuffle=True
    )

    # Initialize models and optimizers
    generator = Generator(config['latent_dim']).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['lr'])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr'])

    # Start training
    train(config, dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # Save generator model
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    main(DEFAULT_CONFIG)
