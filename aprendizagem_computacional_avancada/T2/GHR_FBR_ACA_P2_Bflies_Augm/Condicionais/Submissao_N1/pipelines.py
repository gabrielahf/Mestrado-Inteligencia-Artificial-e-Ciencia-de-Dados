# pipelines.py - GHRoxo

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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


def train_classifier(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    config
):
    """
    Pipeline completo de treinamento da CNN.

    Fluxo:
    1. Treinamento
    2. Validação
    3. Early Stopping
    4. Salvamento do melhor modelo
    5. Avaliação final no conjunto de teste
    """

    best_acc = 0.0
    patience_counter = 0

    # Histórico das métricas
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Backup dos melhores pesos
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

            # Limpa gradientes anteriores
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            loss = criterion(outputs, labels)

            # Backward
            loss.backward()

            # Atualiza pesos
            optimizer.step()

            # Estatísticas
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        # Métricas de treino
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(
            model,
            val_loader,
            criterion,
            config.DEVICE
        )

        # Salva histórico
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

            # Salva melhor modelo
            torch.save(
                best_model_wts,
                'best_classifier.pth'
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= config.CNN_PATIENCE:

                print("\n[INFO] Early Stopping ativado!\n")

                break

    print("\n[INFO] Carregando melhor modelo salvo...\n")

    model.load_state_dict(
        torch.load(
            'best_classifier.pth',
            map_location=config.DEVICE
        )
    )

    test_loss, test_acc = evaluate_model(
        model,
        test_loader,
        criterion,
        config.DEVICE
    )

    print(
        f"[TEST] Loss: {test_loss:.4f} | "
        f"Acc: {test_acc*100:.2f}%"
    )

    return model, history, best_acc, test_acc


def init_dcgan_weights(m):
    """
    Inicialização baseada no artigo DCGAN.

    Conv:
    - média 0
    - std 0.02

    BatchNorm:
    - média 1
    - std 0.02
    """

    classname = m.__class__.__name__

    if 'Conv' in classname:

        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass

    elif 'BatchNorm' in classname:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)


def train_conditional_gan(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    config
):
    """
    Pipeline de treinamento da cGAN.

    Estratégias:
    - Hinge Loss
    - Gradient Clipping
    - Conditional Generation
    """

    generator.train()
    discriminator.train()

    history = {
        'g_loss': [],
        'd_loss': []
    }

    print("\n[INFO] Iniciando treinamento da cGAN...\n")

    for epoch in range(config.GAN_EPOCHS):

        g_running = 0.0
        d_running = 0.0
        n_batches = 0

        for real, labels in tqdm(
            loader,
            desc=f"GAN Epoch {epoch+1}",
            leave=False
        ):

            real = real.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            batch_size = real.size(0)

            opt_d.zero_grad()

            # Imagens reais
            real_logits = discriminator(
                real,
                labels
            )

            d_loss_real = torch.mean(
                F.relu(1.0 - real_logits)
            )

            # Ruído aleatório
            z = torch.randn(
                batch_size,
                config.GAN_LATENT_DIM,
                device=config.DEVICE
            )

            # Labels falsas
            fake_labels = torch.randint(
                0,
                config.NUM_CLASSES,
                (batch_size,),
                device=config.DEVICE
            )

            # Gera imagens falsas
            fake = generator(
                z,
                fake_labels
            )

            # Avalia imagens falsas
            fake_logits = discriminator(
                fake.detach(),
                fake_labels
            )

            d_loss_fake = torch.mean(
                F.relu(1.0 + fake_logits)
            )

            # Loss total do discriminador
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=1.0
            )

            opt_d.step()

            opt_g.zero_grad()

            fake_logits = discriminator(
                fake,
                fake_labels
            )

            # Queremos maximizar logits
            g_loss = -torch.mean(fake_logits)

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=1.0
            )

            opt_g.step()

            g_running += g_loss.item()
            d_running += d_loss.item()

            n_batches += 1

        avg_g = g_running / n_batches
        avg_d = d_running / n_batches

        history['g_loss'].append(avg_g)
        history['d_loss'].append(avg_d)

        print(
            f"GAN Epoch [{epoch+1}/{config.GAN_EPOCHS}] | "
            f"D Loss: {avg_d:.4f} | "
            f"G Loss: {avg_g:.4f}"
        )

    return history

def vae_loss(
    xhat,
    x,
    mu,
    logvar,
    beta=1.0
):
    """
    ELBO Loss:
    - Reconstruction Loss (BCE)
    - KL Divergence
    """

    recon_loss = F.binary_cross_entropy(
        xhat,
        x,
        reduction='sum'
    )

    kl_loss = -0.5 * torch.sum(
        1 +
        logvar -
        mu.pow(2) -
        logvar.exp()
    )

    return recon_loss + beta * kl_loss


def train_vae_conditional(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    config
):
    """
    Pipeline completo de treinamento do C-VAE.

    Inclui:
    - KL Annealing
    - Early Stopping
    - Validação
    - Avaliação final em teste
    """

    best_val_loss = float('inf')

    counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'beta': []
    }

    print("\n[INFO] Iniciando treinamento do C-VAE...\n")

    for epoch in range(config.VAE_EPOCHS):

        beta = min(
            1.0,
            (epoch + 1) / 30
        )

        history['beta'].append(beta)

        model.train()

        train_loss = 0.0
        n = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"VAE Epoch {epoch+1}",
            leave=False
        ):

            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            # Forward
            xhat, mu, logvar = model(
                images,
                labels
            )

            # Loss
            loss = vae_loss(
                xhat,
                images,
                mu,
                logvar,
                beta=beta
            )

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss += loss.item()
            n += images.size(0)

        avg_train_loss = train_loss / n

        model.eval()

        val_loss_total = 0.0
        val_n = 0

        with torch.no_grad():

            for v_imgs, v_lbls in val_loader:

                v_imgs = v_imgs.to(config.DEVICE)
                v_lbls = v_lbls.to(config.DEVICE)

                v_xhat, v_mu, v_logvar = model(
                    v_imgs,
                    v_lbls
                )

                v_loss = vae_loss(
                    v_xhat,
                    v_imgs,
                    v_mu,
                    v_logvar,
                    beta=beta
                )

                val_loss_total += v_loss.item()
                val_n += v_imgs.size(0)

        avg_val_loss = val_loss_total / val_n

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(
            f"VAE Epoch [{epoch+1}/{config.VAE_EPOCHS}] | "
            f"Beta: {beta:.3f} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                'best_vae_model.pth'
            )

            counter = 0

        else:

            counter += 1

            if counter >= config.VAE_PATIENCE:

                print("\n[INFO] Early Stopping no VAE!\n")

                break

    print("\n[INFO] Carregando melhor modelo do VAE...\n")

    model.load_state_dict(
        torch.load(
            'best_vae_model.pth',
            map_location=config.DEVICE
        )
    )

    model.eval()

    test_loss_total = 0.0
    test_n = 0

    with torch.no_grad():

        for t_imgs, t_lbls in test_loader:

            t_imgs = t_imgs.to(config.DEVICE)
            t_lbls = t_lbls.to(config.DEVICE)

            t_xhat, t_mu, t_logvar = model(
                t_imgs,
                t_lbls
            )

            t_loss = vae_loss(
                t_xhat,
                t_imgs,
                t_mu,
                t_logvar,
                beta=1.0
            )

            test_loss_total += t_loss.item()
            test_n += t_imgs.size(0)

    avg_test_loss = test_loss_total / test_n

    print(
        f"[TEST VAE] Loss: {avg_test_loss:.4f}"
    )

    return history, avg_test_loss