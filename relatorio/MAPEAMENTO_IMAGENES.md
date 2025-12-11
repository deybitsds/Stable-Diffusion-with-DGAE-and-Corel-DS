# 🗺️ Mapeamento de Imágenes Disponibles

Este documento mapea las imágenes que tienes a las que necesita el relatorio.

## ✅ Imágenes que Tienes y Dónde Usarlas

### 1. Dataset Corel ✓
- **Tienes**: `codes/figs/corel_dataset_samples.png`
- **Usar en**: `figs/corel_dataset_samples.png` (ya está correcto)
- **Acción**: Copiar a `relatorio/figs/corel_dataset_samples.png`

### 2. LoRA Generated ✓
- **Tienes**: `codes/corel_generated_all.png`
- **Usar en**: Ya actualizado en el LaTeX como `../codes/corel_generated_all.png`
- **Acción**: El LaTeX ya está configurado para usar esta ruta

### 3. VAE Reconstructions ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓
- **Tienes**: `codes/vae_models/reconstructions/reconstruction_epoch_300.png` (y otras épocas)
- **Usar en**: Ya actualizado como `../codes/vae_models/reconstructions/reconstruction_epoch_300.png`
- **Acción**: El LaTeX ya está configurado

### 4. VAE Training Loss ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓
- **Tienes**: `codes/vae_models/training_losses.png`
- **Usar en**: Ya actualizado como `../codes/vae_models/training_losses.png`
- **Acción**: El LaTeX ya está configurado

### 5. Diffusion Generated ✓
- **Tienes**: `codes/diffusion_models/samples/epoch_360.png` (y otras épocas)
- **Usar en**: Ya actualizado como `../codes/diffusion_models/samples/epoch_360.png`
- **Acción**: El LaTeX ya está configurado

### 6. DGAE Samples ✓
- **Tienes**: `codes/dgae_models/samples/samples_epoch_300.png` (y otras épocas)
- **Usar en**: Puedes agregar una figura adicional si quieres mostrar DGAE samples

### 7. DGAE Reconstructions ✓
- **Tienes**: `codes/dgae_models/reconstructions/reconstruction_epoch_300.png`
- **Usar en**: Puedes agregar una figura adicional si quieres mostrar DGAE reconstructions

### 8. DGAE Training Loss ✓
- **Tienes**: `codes/dgae_models/training_losses.png`
- **Usar en**: Puedes agregar una figura adicional si quieres mostrar DGAE loss

## ❌ Imágenes que NO Tienes (Comentadas en LaTeX)

### 1. UMAP SimCLR
- **Estado**: Comentado en LaTeX, texto descriptivo agregado
- **Solución**: Cuando ejecutes `5A-simclr_corel.py --evaluate-clustering`, se generará

### 2. UMAP CNN-JEPA
- **Estado**: Comentado en LaTeX, texto descriptivo agregado
- **Solución**: Cuando ejecutes `5D-cnn-jepa_corel.py --evaluate-clustering`, se generará

### 3. UMAP DGAE
- **Estado**: Comentado en LaTeX, texto descriptivo agregado
- **Solución**: Cuando ejecutes `5E-compare-clustering.py`, se generará

### 4. Clustering Comparison
- **Estado**: Comentado en LaTeX, referencia a tablas agregada
- **Solución**: Cuando ejecutes `5E-compare-clustering.py` correctamente, se generará

## 📋 Comandos para Copiar Imágenes a `relatorio/figs/`

```bash
cd /home/nando/Semestre/unicamp/no_supervisionado/Stable-Diffusion-with-DGAE-and-Corel-DS

# Crear carpeta
mkdir -p relatorio/figs

# 1. Dataset samples
cp codes/figs/corel_dataset_samples.png relatorio/figs/

# Listo! Las demás imágenes se referencian directamente desde codes/
```

## 🔄 Opción Alternativa: Copiar Todas las Imágenes

Si prefieres copiar todas las imágenes a `relatorio/figs/` para tener todo en un solo lugar:

```bash
cd /home/nando/Semestre/unicamp/no_supervisionado/Stable-Diffusion-with-DGAE-and-Corel-DS

mkdir -p relatorio/figs

# Dataset
cp codes/figs/corel_dataset_samples.png relatorio/figs/

# LoRA generated
cp codes/corel_generated_all.png relatorio/figs/corel_lora_generated_grid.png

# VAE reconstructions (usar la última época)
cp codes/vae_models/reconstructions/reconstruction_epoch_300.png relatorio/figs/vae_reconstruction_samples.png

# VAE loss
cp codes/vae_models/training_losses.png relatorio/figs/vae_training_loss.png

# Diffusion generated (usar la última época)
cp codes/diffusion_models/samples/epoch_360.png relatorio/figs/diffusion_generated_samples.png

echo "✅ Imágenes copiadas a relatorio/figs/"
```

Si haces esto, necesitarás actualizar las rutas en el LaTeX de `../codes/...` a `figs/...`.

## 📝 Nota sobre Rutas en LaTeX

El LaTeX ahora usa rutas relativas `../codes/...` para acceder directamente a las imágenes en la carpeta `codes/`. Esto funciona si compilas desde `relatorio/`.

Si prefieres tener todo en `relatorio/figs/`, copia las imágenes y actualiza las rutas en el LaTeX.


