# The Butterfly Effect — Entrega do Projecto

Este repositório contém o relatório e os materiais de suporte do segundo projecto prático da unidade de Advanced Machine Learning (2025/2026), Universidade de Coimbra.

O trabalho investiga a utilização de modelos generativos profundos como estratégia de aumento de dados para um classificador de 75 espécies de Lepidoptera, seguindo uma trajectória em dois estágios: abordagem condicional sobre todas as classes e abordagem isolada sobre as três classes mais sub-representadas.

## Estrutura da entrega

- `ACA_TP2_Report.pdf` — relatório final no formato Springer LNCS
- `Condicionais/` — código relativo à Fase 1: treino da cGAN condicional, C-VAE condicional e configuração híbrida sobre as 75 classes
- `Classes_Unitarias/` — código relativo à Fase 2: treino do Autoencoder, VAE e GAN clássica isoladamente sobre as classes Gold Banded, Malachite e Crimson Patch

## Notas

Os dois notebooks são independentes entre si e seguem a mesma estrutura interna: `config.py` para centralização de hiperparâmetros, `models.py` para definição de arquitecturas e `pipeline.py` para o fluxo de execução.

O dataset não está incluído nesta entrega por restrições de tamanho, descarregado do kaggle antes de executar (instrucoes no notebook).