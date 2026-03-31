Objetivo: Desenvolver uma solução para detectar e quantificar características em imagens de plantas

Dataset sugerido: abdallahalidev/plantvillage-dataset — Kaggle

este repo possui dois notebooks

O notebook /notebooks/01_eda.ipynb abrange:

  1. Download do conjunto de dados e exploração da estrutura

  2. Visualização da distribuição das classes

  3. Grade de imagens de amostra por classe

  4. Comparação entre imagens coloridas e segmentadas

  5. Detalhamento por tipo de cultura (plantio) e por doença

  6. Distribuição do tamanho das imagens

O notebook notebooks/02_segmentation.ipynb aborda o problema da seguinte forma:

  - Arquitetura: U-Net com um encoder leve MobileNetV2 (via segmentation_models_pytorch)

  - Ground truth: Máscaras de 3 classes derivadas de pares de imagens coloridas + segmentadas via limiarização (thresholding) HSV

  - Fase 1: Congelar o encoder → treinar apenas o decoder (convergência rápida)

  - Fase 2: Descongelar todas as camadas → ajuste fino (fine-tuning) com uma taxa de aprendizado 10 vezes menor (fase não realizada)

  - Função de perda (Loss): Combinação de Dice + Cross-Entropy para lidar com o o desbalanceamento severo de classes no nível de pixel

Entregável: Porcentagem de severidade da doença por imagem (área afetada / área total da planta)
