# models.py - GHRoxo

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from PIL import Image

class ButterflyDataset(Dataset):
    """
    Dataset principal para imagens reais.
    """

    def __init__(
        self,
        df,
        img_dir,
        transform=None,
        class_to_idx=None
    ):

        self.img_labels = df.reset_index(drop=True)

        self.img_dir = img_dir

        self.transform = transform

        # Mantém o mesmo mapeamento entre train/val/test
        if class_to_idx is None:

            self.classes = sorted(
                self.img_labels['label'].unique()
            )

            self.class_to_idx = {
                cls_name: idx
                for idx, cls_name in enumerate(self.classes)
            }

        else:

            self.class_to_idx = class_to_idx

            self.classes = list(class_to_idx.keys())

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        row = self.img_labels.iloc[idx]

        img_path = os.path.join(
            self.img_dir,
            row['filename']
        )

        image = Image.open(img_path).convert("RGB")

        label_name = row['label']

        label_idx = self.class_to_idx[label_name]

        label = torch.tensor(
            label_idx,
            dtype=torch.long
        )

        if self.transform:
            image = self.transform(image)

        return image, label


class AugmentedButterflyDataset(Dataset):
    """
    Dataset híbrido:
    - imagens reais
    - imagens sintéticas
    """

    def __init__(
        self,
        df,
        original_img_dir,
        synthetic_img_dir,
        transform=None,
        class_to_idx=None
    ):

        self.img_labels = df.reset_index(drop=True)

        self.original_img_dir = original_img_dir

        self.synthetic_img_dir = synthetic_img_dir

        self.transform = transform

        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        row = self.img_labels.iloc[idx]

        img_name = row['filename']

        is_synth = row.get(
            'is_synthetic',
            0
        )

        # Escolhe pasta dinamicamente
        if is_synth == 1:

            img_path = os.path.join(
                self.synthetic_img_dir,
                img_name
            )

        else:

            img_path = os.path.join(
                self.original_img_dir,
                img_name
            )

        image = Image.open(img_path).convert("RGB")

        label_name = row['label']

        label_idx = self.class_to_idx[label_name]

        label = torch.tensor(
            label_idx,
            dtype=torch.long
        )

        if self.transform:
            image = self.transform(image)

        return image, label


class HybridButterflyDataset(Dataset):
    """
    Dataset híbrido avançado:
    - imagens reais
    - imagens cGAN
    - imagens VAE
    """

    def __init__(
        self,
        df,
        img_dir,
        cgan_img_dir,
        vae_img_dir,
        transform=None,
        class_to_idx=None
    ):

        self.df = df.reset_index(drop=True)

        self.img_dir = img_dir

        self.cgan_img_dir = cgan_img_dir

        self.vae_img_dir = vae_img_dir

        self.transform = transform

        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Define origem da imagem
        if row['folder_type'] == 'cgan':

            target_dir = self.cgan_img_dir

        elif row['folder_type'] == 'vae':

            target_dir = self.vae_img_dir

        else:

            target_dir = self.img_dir

        img_path = os.path.join(
            target_dir,
            row['filename']
        )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            self.class_to_idx[row['label']],
            dtype=torch.long
        )

        return image, label

class BaselineCNN(nn.Module):

    def __init__(self, num_classes=75):

        super().__init__()

        def conv_block(
            in_channels,
            out_channels
        ):

            return nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),

                nn.BatchNorm2d(out_channels),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2, 2)
            )

        self.features = nn.Sequential(

            conv_block(3, 64),

            conv_block(64, 128),

            conv_block(128, 256),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Linear(256, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(512, 1024),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(1024, 2048),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(2048, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        return self.classifier(x)


class CGANGenerator(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        num_classes=75,
        embed_dim=100,
        ngf=64
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.label_embed = nn.Embedding(
            num_classes,
            embed_dim
        )

        self.net = nn.Sequential(

            nn.ConvTranspose2d(
                latent_dim + embed_dim,
                ngf * 8,
                4,
                1,
                0,
                bias=False
            ),

            nn.BatchNorm2d(ngf * 8),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf * 8,
                ngf * 4,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(ngf * 4),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(ngf * 2),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(ngf),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf,
                3,
                4,
                2,
                1,
                bias=False
            ),

            nn.Tanh()
        )

    def forward(self, z, labels):

        lbl_emb = self.label_embed(labels)

        lbl_emb = lbl_emb.view(
            labels.size(0),
            -1,
            1,
            1
        )

        z = z.view(
            z.size(0),
            self.latent_dim,
            1,
            1
        )

        x = torch.cat(
            [z, lbl_emb],
            dim=1
        )

        return self.net(x)


class CGANDiscriminator(nn.Module):

    def __init__(
        self,
        num_classes=75,
        ndf=64,
        img_size=64
    ):

        super().__init__()

        self.img_size = img_size

        self.label_embed = nn.Embedding(
            num_classes,
            img_size * img_size
        )

        self.net = nn.Sequential(

            nn.utils.spectral_norm(
                nn.Conv2d(4, ndf, 4, 2, 1, bias=False)
            ),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.25),

            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            ),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.25),

            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            ),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.25),

            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            ),

            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            )
        )

    def forward(self, x, labels):

        lbl_emb = self.label_embed(labels)

        lbl_emb = lbl_emb.view(
            labels.size(0),
            1,
            self.img_size,
            self.img_size
        )

        x_cond = torch.cat(
            [x, lbl_emb],
            dim=1
        )

        return self.net(x_cond).view(-1)


class ConvVAE(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        num_classes=75,
        embed_dim=100
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.label_embed = nn.Embedding(
            num_classes,
            embed_dim
        )

        # Encoder
        self.enc_conv = nn.Sequential(

            nn.Conv2d(3, 64, 4, 2, 1),

            nn.BatchNorm2d(64),

            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2)
        )

        self.flatten_size = 512 * 4 * 4

        self.fc_mu = nn.Linear(
            self.flatten_size,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            self.flatten_size,
            latent_dim
        )

        self.dec_linear = nn.Linear(
            latent_dim + embed_dim,
            self.flatten_size
        )

        # Decoder
        self.dec_conv = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                4,
                2,
                1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                64,
                3,
                4,
                2,
                1
            ),

            nn.Sigmoid()
        )

    def encode(self, x):

        h = self.enc_conv(x)

        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z, labels):

        label_embedding = self.label_embed(labels)

        z_cond = torch.cat(
            [z, label_embedding],
            dim=1
        )

        h = self.dec_linear(z_cond)

        h = h.view(
            h.size(0),
            512,
            4,
            4
        )

        return self.dec_conv(h)

    def forward(self, x, labels):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(
            z,
            labels
        )

        return reconstruction, mu, logvar