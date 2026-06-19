# models.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image


class ButterflyDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.img_labels['label'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_name = self.img_labels.iloc[idx]['label']
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
            *[layer for size in [256, 512, 1024]
            for layer in (nn.Linear(size, size*2), nn.ReLU(inplace=True), nn.Dropout(p=0.5))],
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ButterflyAutoencoder64(nn.Module):
    def __init__(self, color_channels=3):
        super().__init__()        
        # Batch para melhorar a estabilidade(dentro do batch) e LEaky para manter vivo o gradiente (evitar morrer ou explodir)
        self.encoder = nn.Sequential(
            nn.Conv2d(color_channels, 32, kernel_size=3, stride=2, padding=1), # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),             # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),            # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=8)                                 # 8x8 -> 1x1 (Gargalo latente de 256)
        )
        
        # Decoder: Reconstrói 1x1 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=8),                        # 1x1 -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, color_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32 -> 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
class ButterflyVAE64(nn.Module):
    def __init__(self, color_channels=3, stage="global"):
        super().__init__()
        self.stage = stage
        
        # Encoder (Gargalo 128 - mais estável que 512)
        self.encoder = nn.Sequential(
            nn.Conv2d(color_channels, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Camadas do VAE (Média e Variância)
        self.fc_mu = nn.Linear(128 * 8 * 8, 128)
        self.fc_var = nn.Linear(128 * 8 * 8, 128)
        self.fc_decode = nn.Linear(128, 128 * 8 * 8)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, color_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder(x)
        mu, logvar = self.fc_mu(features), self.fc_var(features)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.fc_decode(z)), mu, logvar

# Loss do VAE (Reconstrução + Divergência KL para organizar o espaço latente)
def vae_loss(recon_x, x, mu, logvar):
    mse = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld
        

class Generator64(nn.Module):
    def __init__(self, latent_dim=128):
        super(Generator64, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # Entrada: Vetor latente Z (batch, 128, 1, 1) -> Saída: (batch, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Saída: (batch, 256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Saída: (batch, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Saída: (batch, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Saída final: (batch, 3, 64, 64)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.dim() == 2:
            input = input.view(-1, self.latent_dim, 1, 1)
        return self.main(input)


class Discriminator64(nn.Module):
    def __init__(self):
        super(Discriminator64, self).__init__()
        self.model = nn.Sequential(
            # Entrada: (batch, 3, 64, 64) -> Saída: (batch, 64, 32, 32)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Saída: (batch, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Saída: (batch, 256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Saída: (batch, 512, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Saída final: (batch, 1, 1, 1) -> Reduz o bloco 4x4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input).view(-1,1)