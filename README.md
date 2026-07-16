# Detección de Imágenes Generadas por IA mediante Deep Learning

Trabajo de Fin de Máster (TFM) — Comparativa de paradigmas supervisados de clasificación binaria y no supervisados de detección de anomalías para la clasificación de imágenes reales vs. generadas por inteligencia artificial.

---

## Descripción

Este repositorio contiene los experimentos desarrollados en el TFM, cuyo objetivo es evaluar y comparar diferentes arquitecturas de deep learning para detectar si una imagen ha sido generada por un modelo generativo (IA) o es una fotografía real.

Se exploran dos paradigmas principales:

- **Clasificación supervisada**: modelos entrenados con etiquetas binarias (real / fake).
- **Detección de anomalías no supervisada**: modelos VAE que aprenden la distribución de imágenes reales y detectan las generadas como anomalías, sin necesidad de etiquetas de clase durante el entrenamiento.

---

## Dataset

El dataset utilizado contiene 60 000 imágenes organizadas en la siguiente estructura:

```
dataset/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Las imágenes se redimensionan a **224 × 224 px** (modelos Keras/TF) o **256 × 256 px** (modelos PyTorch/VAE).

> **Nota metodológica**: el dataset procede de Kaggle (tristanzhang32, *AI Generated Images vs Real Images*) 

---

## Arquitecturas experimentadas

### Paradigma supervisado (TensorFlow / Keras)

| Notebook | Modelo | Etapa |
|---|---|---|
| `class_cnn_preentrenada.ipynb` | EfficientNetB0 (backbone congelado) | Evaluación zero-shot |
| `class_cnn_finetuning.ipynb` | EfficientNetB0 (fine-tuning) | Ajuste fino sobre dataset |
| `class_cnn_salida.ipynb` | EfficientNetB0 (solo capa de salida) | Entrenamiento únicamente del clasificador |
| `class_cnn_vit_preentrenado.ipynb` | CNN (EfficientNetB0) + ViT Base 16×16 (congelado) | Híbrido CNN+ViT, evaluación inicial |
| `class_cnn_vit_finetuning.ipynb` | CNN (EfficientNetB0) + ViT Base 16×16 (fine-tuning) | Ajuste fino del modelo híbrido |
| `class_cnn_vit_salida.ipynb` | CNN (EfficientNetB0) + ViT Base 16×16 (solo salida) | Entrenamiento únicamente del clasificador |
| `class_vit_preentrenado.ipynb` | ViT Base 16×16 (congelado) | Evaluación zero-shot |
| `class_vit_finetuning.ipynb` | ViT Base 16×16 (fine-tuning) | Ajuste fino completo |
| `class_vit_salida.ipynb` | ViT Base 16×16 (solo salida) | Entrenamiento únicamente del clasificador |

### Paradigma no supervisado — Detección de anomalías con VAE (PyTorch)

| Notebook | Modelo | Enfoque |
|---|---|---|
| `vae_pretrained.ipynb` | VAE preentrenado (`REPA-E/e2e-invae-hf`) | Error MSE |
| `vae_pretrained_finetuning.ipynb` | VAE preentrenado + fine-tuning del decoder | Error de reconstrucción píxel (MSE) |
| `cnn_vae_pretrained.ipynb` | ResNet-50 (extractor de características) + VAE | Características CNN combinadas con error VAE |

---

## Stack tecnológico

| Componente | Herramienta |
|---|---|
| Framework supervisado | TensorFlow 2.18 / Keras 3.14.1 |
| Framework VAE | PyTorch + HuggingFace `diffusers` |
| Modelo VAE preentrenado | `REPA-E/e2e-invae-hf` (AutoencoderKL) |
| Backbone supervisado | EfficientNetB0 (ImageNet), ViT Base 16×16 (via `keras-hub`) |
| Backbone CNN-VAE | ResNet-50 (ImageNet V2, via `torchvision`) |
| Métricas | scikit-learn: AUC-ROC, F1, accuracy, matriz de confusión |
| Visualización | matplotlib, seaborn |
| Infraestructura | RunPod (RTX 4090 24 GB) |

---

## Configuración de entrenamiento

**Modelos supervisados (Keras)**

```python
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 100       # con Early Stopping (patience=5)
LR_pretrain = 1e-3
LR_finetune = 1e-5
DROPOUT    = 0.2
VAL_SPLIT  = 0.2
```

**Modelos VAE (PyTorch)**

```python
IMG_SIZE   = 256
BATCH_SIZE = 16        # (64 en CNN-VAE)
NITS       = 50
LR         = 1e-3      # (5e-5 en fine-tuning)
PATIENCE   = 5
VAL_SPLIT  = 0.2
```

Data augmentation aplicado al conjunto de entrenamiento: flip horizontal aleatorio, rotación (±36°), random resized crop, autocontrast aleatorio.

---

## Métricas y evaluación

La métrica principal es el **AUC-ROC**, elegida por ser independiente del umbral de decisión y por permitir diagnosticar la separabilidad de distribuciones incluso cuando éstas se solapan. Se complementa con F1-score, accuracy y matriz de confusión.

Para los modelos VAE, la detección de anomalías se implementa comparando el error de reconstrucción de cada imagen contra un umbral óptimo calculado sobre el conjunto de validación. Se exploran dos señales de error:

- **Error en espacio latente**: distancia de Mahalanobis simplificada sobre `mu` respecto a la distribución aprendida de imágenes reales. Utilizado simplemente para ilustrar el impacto de diferentes métricas de error.
- **Error de reconstrucción píxel (MSE)**: diferencia cuadrática media entre imagen original y reconstruida; métrica válida para el VAE con fine-tuning del decoder (encoder congelado). Métrica utilizada para la comparativa del impacto de los reentrenamientos

---

## Estructura del repositorio

```
├── class_cnn_preentrenada.ipynb
├── class_cnn_finetuning.ipynb
├── class_cnn_salida.ipynb
├── class_cnn_vit_preentrenado.ipynb
├── class_cnn_vit_finetuning.ipynb
├── class_cnn_vit_salida.ipynb
├── class_vit_preentrenado.ipynb
├── class_vit_finetuning.ipynb
├── class_vit_salida.ipynb
├── vae_pretrained.ipynb
├── vae_pretrained_finetuning.ipynb
├── cnn_vae_pretrained.ipynb
└── README.md
```

---

## Referencia

> Este repositorio forma parte del Trabajo de Fin de Máster en Desarrollo de Inteligencia Artificial. Los experimentos han sido ejecutados en infraestructura cloud GPU (RunPod) sobre un dataset privado no incluido en este repositorio.
