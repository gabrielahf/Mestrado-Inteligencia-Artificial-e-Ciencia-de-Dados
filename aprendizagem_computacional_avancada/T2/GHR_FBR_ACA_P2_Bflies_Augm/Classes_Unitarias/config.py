import torch
import os

class ProjectConfig:
    EXP_NAME = "teste_FBR_1"

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
    CNN_LR = 0.001
    CNN_PATIENCE = 15

    # VAE
    AE_EPOCHS = 120
    AE_LR = 0.0005         # RECOMENDAÇÃO: Reduzir de 0.001 para 0.0005. 
    AE_LATENT_DIM = 128    # Perfeito para imagens 64x64 de 3 classes.
    AE_PATIENCE = 10

    # GAN
    GAN_EPOCHS = 80       
    GAN_LR = 0.0002        # RECOMENDAÇÃO: Mudar para 0.0002.
    GAN_LATENT_DIM = 128

    SAMPLE_PER_CLASS = 47 # (105-48) - (teto - nossas classes)