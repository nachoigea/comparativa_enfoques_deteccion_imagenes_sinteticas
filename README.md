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

> **Nota metodológica**: el dataset procede de Kaggle (tristanzhang32, *AI Generated Images vs Real Images*) y carece de documentación metodológica indexada en publicaciones científicas. Esta limitación se señala explícitamente en la memoria del TFM, contrastándola con datasets mejor documentados como CIFAKE (Bird & Lotfi, 2024, *IEEE Access*).

---

## Arquitecturas experimentadas

### Paradigma supervisado (TensorFlow / Keras)

| Notebook | Modelo | Etapa |
|---|---|---|
| `class_cnn_preentrenada.ipynb` | EfficientNetB0 (frozen backbone) | Evaluación zero-shot / feature extraction |
| `class_cnn_finetuning.ipynb` | EfficientNetB0 (fine-tuning) | Ajuste fino sobre dataset |
| `class_cnn_salida.ipynb` | EfficientNetB0 (solo capa de salida) | Entrenamiento únicamente del clasificador |
| `class_cnn_vit_preentrenado.ipynb` | CNN (EfficientNetB0) + ViT Base 16×16 (frozen) | Híbrido CNN+ViT, evaluación inicial |
| `class_cnn_vit_finetuning.ipynb` | CNN (EfficientNetB0) + ViT Base 16×16 (fine-tuning) | Ajuste fino del híbrido |
| `class_cnn_vit_salida.ipynb` | CNN (EfficientNetB0) + ViT Base 16×16 (solo salida) | Entrenamiento únicamente del clasificador |
| `class_vit_preentrenado.ipynb` | ViT Base 16×16 (frozen) | Evaluación zero-shot |
| `class_vit_finetuning.ipynb` | ViT Base 16×16 (fine-tuning) | Ajuste fino completo |
| `class_vit_salida.ipynb` | ViT Base 16×16 (solo salida) | Entrenamiento únicamente del clasificador |

### Paradigma no supervisado — Detección de anomalías con VAE (PyTorch)

| Notebook | Modelo | Enfoque |
|---|---|---|
| `vae_pretrained.ipynb` | VAE preentrenado (`REPA-E/e2e-invae-hf`) | Error en espacio latente (distancia de Mahalanobis simplificada) |
| `vae_pretrained_finetuning.ipynb` | VAE preentrenado + fine-tuning del decoder | Error de reconstrucción píxel (MSE) |
| `cnn_vae_pretrained.ipynb` | ResNet-50 (extractor de características) + VAE | Características CNN combinadas con error VAE |

---

## Stack tecnológico

| Componente | Herramienta |
|---|---|
| Framework supervisado | TensorFlow 2.18 / Keras 3.x |
| Framework VAE | PyTorch + HuggingFace `diffusers` |
| Modelo VAE preentrenado | `REPA-E/e2e-invae-hf` (AutoencoderKL) |
| Backbone supervisado | EfficientNetB0 (ImageNet), ViT Base 16×16 (via `keras-hub`) |
| Backbone CNN-VAE | ResNet-50 (ImageNet V2, via `torchvision`) |
| Métricas | scikit-learn: AUC-ROC, F1, accuracy, matriz de confusión |
| Perceptual loss | `lpips` |
| Visualización | matplotlib, seaborn |
| Infraestructura | RunPod (RTX A6000 48 GB para PyTorch; RTX 4090 24 GB para TensorFlow) |

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

- **Error en espacio latente**: distancia de Mahalanobis simplificada sobre `mu` respecto a la distribución aprendida de imágenes reales.
- **Error de reconstrucción píxel (MSE)**: diferencia cuadrática media entre imagen original y reconstruida; métrica válida para el VAE con fine-tuning del decoder (encoder congelado).

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

## Hallazgos principales

- Los modelos supervisados (EfficientNetB0, CNN+ViT) alcanzan AUC-ROC cercano a **0.98** tras fine-tuning, con rendimiento prácticamente idéntico entre arquitecturas.
- Los modelos VAE de detección de anomalías obtienen AUC-ROC en el rango **0.60–0.71**, una brecha atribuida a la debilidad del MSE píxel como señal discriminativa para imágenes generadas por IA modernas, diseñadas para ser perceptualmente indistinguibles de fotografías reales.
- La pérdida KL empuja todos los vectores `mu` hacia N(0, I), reduciendo el poder discriminativo de métricas basadas en espacio latente.
- La comparación pretrained vs. fine-tuned sobre el VAE requiere métricas distintas: el error latente solo es válido cuando el encoder puede actualizarse; para el decoder fine-tuned con encoder congelado, el MSE píxel es la métrica correcta.
- Los resultados negativos del paradigma no supervisado se interpretan como contribución analítica válida, ilustrando los límites estructurales de la detección de anomalías frente a generadores adversariales modernos.

---

## Referencia

> Este repositorio forma parte del Trabajo de Fin de Máster en [nombre del Máster]. Los experimentos han sido ejecutados en infraestructura cloud GPU (RunPod) sobre un dataset privado no incluido en este repositorio.
