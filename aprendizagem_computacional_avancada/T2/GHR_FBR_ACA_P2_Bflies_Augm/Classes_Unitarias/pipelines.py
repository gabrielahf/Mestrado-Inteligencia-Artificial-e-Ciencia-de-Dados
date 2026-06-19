import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os, json
from torchvision.io import read_image
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from torchmetrics.functional.image import structural_similarity_index_measure
import torchvision.utils as vutils


from tqdm.auto import tqdm

def evaluate_model(model, loader, criterion, device):
    """
    Função de avaliação para validação e teste.

    O modelo entra em modo de avaliação:
    - desativa Dropout
    - estabiliza BatchNorm
    - desliga gradientes

    Retorna:
    - loss média
    - acurácia média
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            # Acumula loss ponderada pelo batch
            running_loss += loss.item() * images.size(0)

            # Classe predita
            _, preds = torch.max(outputs, 1)

            # Contagem correta
            correct += (preds == labels).sum().item()

            total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc


def train_classifier(model, train_loader, val_loader, test_loader, optimizer, criterion, config, force_train=False):
    """
    Pipeline completo de treinamento da CNN.

    Fluxo:
    1. Verifica se já existe modelo treinado (se force_train=False, carrega direto)
    2. Treinamento / Validação / Early Stopping
    3. Salvamento do melhor modelo
    4. Avaliação final no conjunto de teste
    """

    checkpoint_path = 'best_classifier.pth'

    # SE O MODELO JÁ EXISTE E NÃO FORÇAMOS O TREINO: CARREGA DIRETO
    if os.path.exists(checkpoint_path) and not force_train:
        print(f"\n[INFO] Checkpoint '{checkpoint_path}' encontrado! Carregando pesos existentes...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        
        print("[INFO] Avaliando modelo carregado no conjunto de teste local...")
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, config.DEVICE)
        
        print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}%\n")
        
        # Retorna histórico vazio (ou dummy) para manter a compatibilidade de desempacotamento da célula
        dummy_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        return model, dummy_history, test_acc, test_acc

    
    # FLUXO DE TREINO ORIGINAL (Caso não exista o arquivo ou queira retreinar)
  
    best_acc = 0.0
    patience_counter = 0

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())

    print("\n[INFO] Iniciando treinamento da CNN...\n")

    for epoch in range(config.CNN_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, config.DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{config.CNN_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.CNN_PATIENCE:
                print("\n[INFO] Early Stopping ativado!\n")
                break

    print("\n[INFO] Carregando melhor modelo salvo...\n")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, config.DEVICE)
    print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}%")

    return model, history, best_acc, test_acc
    

def vae_loss(
    xhat, x, mu, logvar, beta=1.0):
    """
    ELBO Loss:
    - Reconstruction Loss (BCE)
    - KL Divergence
    """

    recon_loss = F.binary_cross_entropy(xhat, x, reduction='sum')

    kl_loss = -0.5 * torch.sum( 1 + logvar - mu.pow(2) - logvar.exp() )

    return recon_loss + beta * kl_loss


def train_autoencoder(model, train_loader, val_loader, optimizer, config, target_class, force_train=False):
    """
    Pipeline de treinamento do Autoencoder Não-Condicional.
    Métricas salvas: Loss (MSE) e Acurácia de Reconstrução Baseada em L1 Relativo.
    """
    class_suffix = target_class.replace(" ", "_")
    checkpoint_path = f"best_autoencoder_{class_suffix}.pth"

    # Se o modelo já existe e não forçamos o treino: carrega direto
    if os.path.exists(checkpoint_path) and not force_train:
        print(f"\n[INFO] Checkpoint do Autoencoder '{checkpoint_path}' encontrado! Carregando pesos existentes...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        model.eval()
        # Retorna histórico vazio para compatibilidade de desempacotamento
        return model, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
    }

    print("\n[INFO] Iniciando treinamento do Autoencoder.\n")

    for epoch in range(config.AE_EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0.0  # Métrica de fidelidade baseada em 1 - L1_Loss_normalizada
        total_samples = 0

        for images, _ in train_loader:
            images = images.to(config.DEVICE)
            batch_size = images.size(0)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            
            # Métrica de proximidade de pixel (1.0 = reconstrução perfeita)
            with torch.no_grad():
                l1_dist = F.l1_loss(outputs, images, reduction='mean').item()
                running_acc += (1.0 - min(1.0, l1_dist)) * batch_size
                
            total_samples += batch_size

        train_loss = running_loss / total_samples
        train_acc = running_acc / total_samples

        # Validação
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        val_total_samples = 0

        with torch.no_grad():
            for v_images, _ in val_loader:
                v_images = v_images.to(config.DEVICE)
                v_batch_size = v_images.size(0)
                
                v_outputs = model(v_images)
                v_loss = criterion(v_outputs, v_images)
                
                val_running_loss += v_loss.item() * v_batch_size
                v_l1_dist = F.l1_loss(v_outputs, v_images, reduction='mean').item()
                val_running_acc += (1.0 - min(1.0, v_l1_dist)) * v_batch_size
                val_total_samples += v_batch_size

        val_loss = val_running_loss / val_total_samples
        val_acc = val_running_acc / val_total_samples

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"AE Epoch [{epoch+1}/{config.AE_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc (Fidelity): {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc (Fidelity): {val_acc*100:.2f}%"
        )

        # Salvamento estrito do melhor modelo (Early Stopping por Val Loss mínimo)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.AE_PATIENCE:
                print("\n[INFO] Early Stopping ativado no Autoencoder!\n")
                break

    print("\n[INFO] Carregando melhor modelo do Autoencoder salvo...\n")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()
    return model, history
    
def train_vae(model, train_loader, val_loader, optimizer, config, target_class, force_train=False):

    if target_class == 'MASTER':
        epochs = config.AE_EPOCHS
    else:
        epochs = 15
        
    class_suffix = target_class.replace(" ", "_")
    checkpoint_path = f"best_vae_{class_suffix}.pth"
    history_path = f"history_vae_{class_suffix}.json"

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Early Stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10 

    if os.path.exists(checkpoint_path) and not force_train:
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        if os.path.exists(history_path):
            with open(history_path, 'r') as f: return model, json.load(f)
    
    for epoch in range(epochs):
        model.train()
        running_loss = running_acc = total = 0.0
        
        for images, _ in train_loader:
            images = images.to(config.DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss = vae_loss(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                running_loss += loss.item()
                l1 = F.l1_loss(recon, images, reduction='mean').item()
                running_acc += (1.0 - min(1.0, l1)) * images.size(0)
                total += images.size(0)
        
        # Validação
        model.eval()
        val_running_loss = val_total = 0.0
        val_running_acc = 0.0
        with torch.no_grad():
            for v_images, _ in val_loader:
                v_images = v_images.to(config.DEVICE)
                v_recon, v_mu, v_logvar = model(v_images)
                val_running_loss += vae_loss(v_recon, v_images, v_mu, v_logvar).item()
                v_l1 = F.l1_loss(v_recon, v_images, reduction='mean').item()
                val_running_acc += (1.0 - min(1.0, v_l1)) * v_images.size(0)
                val_total += v_images.size(0)

        epoch_val_loss = val_running_loss / val_total
        
        # Histórico
        history['train_loss'].append(running_loss / total)
        history['train_acc'].append((running_acc / total) * 100)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append((val_running_acc / val_total) * 100)
        
        # Early Stopping Logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path) # Salva apenas quando melhora
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early Stopping atingido na época {epoch+1}. Interrompendo.")
                break
        
        with open(history_path, 'w') as f: json.dump(history, f, indent=4)
        print(f"VAE Epoch {epoch+1} | Train_Loss: {history['train_loss'][-1]:.2f} | Val_Loss: {epoch_val_loss:.2f} | Train_Acc: {history['train_acc'][-1]:.2f}| Val_Acc: {history['val_acc'][-1]:.2f}%")
        
    return model, history
    
    

def train_pure_gan(generator, discriminator, loader, opt_g, opt_d, config, target_class, force_train=False):
    
    if target_class == 'MASTER':
        epochs = config.GAN_EPOCHS
    else:
        epochs = 15

    class_suffix = target_class.replace(" ", "_")
    g_checkpoint = f"best_gan_gen_{class_suffix}.pth"
    d_checkpoint = f"best_gan_disc_{class_suffix}.pth"

    # Noise fixo para monitorar a evolução visual das mesmas "borboletas"
    fixed_noise = torch.randn(16, config.GAN_LATENT_DIM, 1, 1, device=config.DEVICE)

    if os.path.exists(g_checkpoint) and os.path.exists(d_checkpoint) and not force_train:
        print(f"\n[INFO] Checkpoints encontrados! Carregando...")
        generator.load_state_dict(torch.load(g_checkpoint, map_location=config.DEVICE))
        discriminator.load_state_dict(torch.load(d_checkpoint, map_location=config.DEVICE))
        generator.eval()
        discriminator.eval()
        return generator, discriminator, {'g_loss': [], 'd_loss': [], 'd_acc_real': [], 'd_acc_fake': []}

    generator.train()
    discriminator.train()
    history = {'g_loss': [], 'd_loss': [], 'd_acc_real': [], 'd_acc_fake': []}
    best_g_loss = float('inf')

    print("\n[INFO] Iniciando treinamento da GAN Pura...\n")

    for epoch in range(epochs):
        g_running, d_running = 0.0, 0.0
        d_correct_real, d_correct_fake, total_samples, n_batches = 0, 0, 0, 0

        for real, _ in loader:
            real = real.to(config.DEVICE)
            batch_size = real.size(0)

            # --- Treino Discriminador ---
            opt_d.zero_grad()
            real_logits = discriminator(real)
            d_loss_real = torch.mean(F.relu(1.0 - real_logits))
            d_correct_real += (real_logits > 0.0).sum().item()

            z = torch.randn(batch_size, config.GAN_LATENT_DIM, 1, 1, device=config.DEVICE)
            fake = generator(z)
            fake_logits = discriminator(fake.detach())
            d_loss_fake = torch.mean(F.relu(1.0 + fake_logits))
            d_correct_fake += (fake_logits <= 0.0).sum().item()

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # --- Treino Gerador ---
            opt_g.zero_grad()
            fake_logits_for_g = discriminator(fake)
            g_loss = -torch.mean(fake_logits_for_g)
            g_loss.backward()
            opt_g.step()

            g_running += g_loss.item()
            d_running += d_loss.item()
            total_samples += batch_size
            n_batches += 1

        # Métricas
        avg_g, avg_d = g_running / n_batches, d_running / n_batches
        acc_real, acc_fake = d_correct_real / total_samples, d_correct_fake / total_samples
        history['g_loss'].append(avg_g)
        history['d_loss'].append(avg_d)
        history['d_acc_real'].append(acc_real)
        history['d_acc_fake'].append(acc_fake)

        print(f"GAN Epoch [{epoch+1}/{epochs}] | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | "
              f"D Acc R: {acc_real*100:.2f}% | D Acc F: {acc_fake*100:.2f}%")

        # Salva Grid de Amostras
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                gen_images = generator(fixed_noise).detach().cpu()
                vutils.save_image((gen_images + 1.0) / 2.0, f"samples_epoch_{epoch+1}.png", nrow=4)

        # Early Stop: Convergência estável (70-80% de acurácia)
        if epoch > 20 and 0.70 < acc_real < 0.80 and 0.70 < acc_fake < 0.80:
            print(f"\n[INFO] Equilíbrio atingido (Epoch {epoch+1}). Early Stop acionado.")
            break

        if avg_g < best_g_loss and acc_real > 0.5:
            best_g_loss = avg_g
            torch.save(generator.state_dict(), g_checkpoint)
            torch.save(discriminator.state_dict(), d_checkpoint)
            
    print("\n[INFO] Carregando melhores checkpoints da GAN salvos...\n")
    generator.load_state_dict(torch.load(g_checkpoint, map_location=config.DEVICE))
    discriminator.load_state_dict(torch.load(d_checkpoint, map_location=config.DEVICE))
    generator.eval()
    discriminator.eval()

    return generator, discriminator, history
    
    
def gerar_samples(model, target_class, num_samples, config, output_base_dir, model_type='vae', loader=None):
    model.eval()
    class_suffix = target_class.replace(" ", "_")
    output_dir = os.path.join(output_base_dir, class_suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFERENCE] Gerando {num_samples} amostras para [{target_class}] via {model_type.upper()}...")
    
    with torch.no_grad():
        if model_type.lower() == 'vae':
            z = torch.randn(num_samples, config.AE_LATENT_DIM, device=config.DEVICE)
            fake_images = model.decoder(model.fc_decode(z))
        elif model_type.lower() == 'gan':
            z = torch.randn(num_samples, config.GAN_LATENT_DIM, 1, 1, device=config.DEVICE)
            fake_images = model(z)
            # Tanh -> [0, 1]
            fake_images = (fake_images + 1.0) / 2.0
        elif model_type.lower() == 'ae':
            if loader is None:
                raise ValueError("AE requer um 'loader' para extrair a base latente.")
            
            # Pegamos um lote do loader para ter mais de 1 amostra
            real_imgs, _ = next(iter(loader))
            real_imgs = real_imgs.to(config.DEVICE)
            
            if real_imgs.size(0) < num_samples:
                # Se precisar de mais que o batch, ajustamos para o que temos
                real_imgs = real_imgs.repeat(num_samples // real_imgs.size(0) + 1, 1, 1, 1)
            
            real_imgs = real_imgs[:num_samples]
            z = model.encoder(real_imgs)
            z = z + torch.randn_like(z) * 0.1
            fake_images = model.decoder(z)
        else:
            raise ValueError("model_type deve ser 'vae', 'gan' ou 'ae'.")
            
        final_tensor = torch.clamp(fake_images, 0.0, 1.0)
        
        for i in range(num_samples):
            filename = f"synthetic_{class_suffix}_{i:03d}.jpg"
            # vutils agora disponível
            vutils.save_image(final_tensor[i], os.path.join(output_dir, filename))
            
    print(f"[STATUS] Sucesso: {num_samples} amostras salvas em '{output_dir}'.")
    

def avaliar_metricas_por_classe(val_loader_real, output_base_dir, target_class, device):
    class_suffix = target_class.replace(" ", "_")
    folder_sinteticas = os.path.join(output_base_dir, class_suffix)
    
    arquivos = [f for f in os.listdir(folder_sinteticas) if f.lower().endswith(('.jpg', '.png'))]
    if not arquivos: return None
    
    list_fakes_8bit = []
    list_fakes_float = [] 
    
    for f in arquivos:
        img = read_image(os.path.join(folder_sinteticas, f)).to(device)
        if img.shape[0] == 4: img = img[:3, :, :]
        # Garantir que a imagem sintética esteja em [0, 255] uint8 para FID/IS
        list_fakes_8bit.append(img.unsqueeze(0).to(torch.uint8))
        list_fakes_float.append(img.unsqueeze(0).to(torch.float32) / 255.0)
        
    fakes_8bit_tensor = torch.cat(list_fakes_8bit, dim=0)
    fakes_float_tensor = torch.cat(list_fakes_float, dim=0)

    # FID/IS exigem modelos de percepção. feature=2048 usa o Inception-v3 original.
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore(feature=2048).to(device) # CORREÇÃO 4
    
    ssim_valores = []
    idx_fake = 0
    num_fakes = fakes_float_tensor.size(0)
    
    print(f"\n[METRICS] Processando métricas para [{target_class}]...")
    for reais, _ in val_loader_real:
        reais = reais.to(device)
        reais_8bit = (reais.clamp(0, 1) * 255).to(torch.uint8)
        fid_metric.update(reais_8bit, real=True)
        
        for i in range(reais.size(0)):
            img_fake_single = fakes_float_tensor[idx_fake % num_fakes].unsqueeze(0)
            ssim_val = structural_similarity_index_measure(img_fake_single, reais[i].unsqueeze(0))
            ssim_valores.append(ssim_val.item())
            idx_fake += 1

    batch_size_metric = 16
    for i in range(0, fakes_8bit_tensor.size(0), batch_size_metric):
        batch_fakes = fakes_8bit_tensor[i:i+batch_size_metric]
        fid_metric.update(batch_fakes, real=False)
        is_metric.update(batch_fakes)

    fid_resultado = fid_metric.compute().item()
    is_mean, _ = is_metric.compute()
    ssim_medio = sum(ssim_valores) / len(ssim_valores) if ssim_valores else 0.0
    
    return {'FID': fid_resultado, 'IS': is_mean.item(), 'SSIM': ssim_medio}