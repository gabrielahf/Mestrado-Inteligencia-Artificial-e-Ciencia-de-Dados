import torch
import os

class ProjectConfig:
    EXP_NAME = "teste_3"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    _cores_detetados = os.cpu_count() if os.cpu_count() is not None else 0
    
    if _cores_detetados > 2:
        NUM_WORKERS = _cores_detetados // 2  # local: 8 // 2 = 4 
    else:
        NUM_WORKERS = _cores_detetados       # No Colab: usar 2

    NUM_CLASSES = 75
    IMAGE_SIZE = 64
    BATCH_SIZE = 32

    # CNN
    CNN_EPOCHS = 120
    CNN_LR = 0.0005
    CNN_PATIENCE = 15

    # GAN
    GAN_EPOCHS = 200
    GAN_LR = 0.0001
    GAN_LATENT_DIM = 256
    GAN_SAMPLE_PER_CLASS = 80

    # VAE
    VAE_EPOCHS = 80
    VAE_LR = 0.0003
    VAE_LATENT_DIM = 256

    # KL Annealing
    VAE_BETA = 0.1
    VAE_PATIENCE = 10