# 📸 Imagens Necessárias para o Relatório

Este documento lista todas as imagens referenciadas no relatório que precisam ser geradas ou adicionadas.

## Estrutura de Diretórios

Crie a seguinte estrutura no diretório `relatorio/`:

```
relatorio/
└── figs/
    ├── corel_dataset_samples.png
    ├── corel_lora_generated_grid.png
    ├── vae_reconstruction_samples.png
    ├── vae_training_loss.png
    ├── diffusion_generated_samples.png
    ├── umap_simclr.png
    ├── umap_cnn_jepa.png
    ├── umap_dgae.png
    └── clustering_comparison.png
```

## Lista de Imagens

### 1. Dataset Corel (Figura \ref{fig:corel_samples})
**Arquivo**: `figs/corel_dataset_samples.png`  
**Gerado por**: `2A-prepare_corel_dataset.py`  
**Localização**: `figs/corel_dataset_samples.png` (gerado automaticamente)  
**Descrição**: Grid mostrando exemplos de imagens da base Corel organizadas por classe.

**Como gerar**:
```bash
cd codes
python 2A-prepare_corel_dataset.py --data-dir data/corel --base-dir .
# A imagem será gerada em figs/corel_dataset_samples.png
```

---

### 2. Imagens Geradas por LoRA (Figura \ref{fig:lora_generated})
**Arquivo**: `figs/corel_lora_generated_grid.png`  
**Gerado por**: `2C-generate-lora-corel.py`  
**Localização**: `generated_images/corel_all_grid_*.png`  
**Descrição**: Grid de imagens geradas usando modelo LoRA treinado.

**Como gerar**:
```bash
cd codes
python 2C-generate-lora-corel.py --lora-dir corel_models --num-images 16
# Copiar o grid gerado para relatorio/figs/
cp generated_images/corel_all_grid_*.png ../relatorio/figs/corel_lora_generated_grid.png
```

---

### 3. Reconstruções VAE (Figura \ref{fig:vae_recon})
**Arquivo**: `figs/vae_reconstruction_samples.png`  
**Gerado por**: `3A-train-vae-corel.py` (durante treinamento)  
**Localização**: Verificar saída do script durante treinamento  
**Descrição**: Comparação lado a lado de imagens originais e reconstruídas pelo VAE.

**Como gerar**: O script `3A-train-vae-corel.py` deve salvar visualizações durante treinamento. Verificar diretório de saída.

---

### 4. Perda de Treinamento VAE (Figura \ref{fig:vae_loss})
**Arquivo**: `figs/vae_training_loss.png`  
**Gerado por**: Plotar perda durante treinamento do VAE  
**Descrição**: Gráfico mostrando evolução da perda durante treinamento.

**Como gerar**: Adicionar código para plotar perda durante treinamento ou usar dados do log.

---

### 5. Imagens Geradas por Difusão (Figura \ref{fig:diffusion_generated})
**Arquivo**: `figs/diffusion_generated_samples.png`  
**Gerado por**: `3C-generate-samples-corel.py`  
**Localização**: Verificar saída do script  
**Descrição**: Grid de imagens geradas pelo modelo de difusão treinado do zero.

**Como gerar**:
```bash
cd codes
python 3C-generate-samples-corel.py \
    --diffusion-checkpoint diffusion_models/best_model.pt \
    --vae-checkpoint vae_models/best_model.pt \
    --num-images 16
# Organizar em grid e salvar em relatorio/figs/
```

---

### 6. UMAP SimCLR (Figura \ref{fig:umap_simclr})
**Arquivo**: `figs/umap_simclr.png`  
**Gerado por**: `5A-simclr_corel.py` (com opção `--evaluate-clustering`)  
**Localização**: Verificar saída do script  
**Descrição**: Projeção UMAP dos features aprendidos por SimCLR.

**Como gerar**:
```bash
cd codes
python 5A-simclr_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir simclr_models \
    --evaluate-clustering
# O script deve gerar visualização UMAP
```

---

### 7. UMAP CNN-JEPA (Figura \ref{fig:umap_cnnjepa})
**Arquivo**: `figs/umap_cnn_jepa.png`  
**Gerado por**: `5D-cnn-jepa_corel.py` (com opção `--evaluate-clustering`)  
**Localização**: Verificar saída do script  
**Descrição**: Projeção UMAP dos features aprendidos por CNN-JEPA.

**Como gerar**:
```bash
cd codes
python 5D-cnn-jepa_corel.py \
    --data-dir training_data/corel/corel_all \
    --output-dir cnn_jepa_models \
    --evaluate-clustering
```

---

### 8. UMAP DGAE (Figura \ref{fig:umap_dgae})
**Arquivo**: `figs/umap_dgae.png`  
**Gerado por**: `5E-compare-clustering.py` ou script separado  
**Localização**: Verificar saída do script de comparação  
**Descrição**: Projeção UMAP dos features aprendidos por DGAE.

**Como gerar**: Usar features extraídas por `4B-extract-features-corel.py` e plotar UMAP.

---

### 9. Comparação de Clustering (Figura \ref{fig:clustering_comparison})
**Arquivo**: `figs/clustering_comparison.png`  
**Gerado por**: `5E-compare-clustering.py`  
**Localização**: `clustering_results/clustering_comparison.png`  
**Descrição**: Gráficos comparando métricas de clustering entre técnicas.

**Como gerar**:
```bash
cd codes
python 5E-compare-clustering.py \
    --features-dir features \
    --output-dir clustering_results \
    --compare-both
# Copiar para relatorio/figs/
cp clustering_results/clustering_comparison.png ../relatorio/figs/
```

---

## Tabelas Necessárias

As tabelas são geradas automaticamente no LaTeX, mas você precisa preencher os valores com os resultados reais:

### Tabela 1: Métricas Sem Aumentação (Tabela \ref{tab:clustering_baseline})
- Preencher valores de ARI, NMI, Silhouette, V-measure para SimCLR, CNN-JEPA, DGAE

### Tabela 2: Métricas Com Aumentação (Tabela \ref{tab:clustering_augmented})
- Preencher valores comparando com e sem aumentação

---

## Notas Importantes

1. **Resolução**: Todas as imagens devem ter resolução adequada (mínimo 300 DPI para impressão)
2. **Formato**: PNG é preferível para gráficos, JPG pode ser usado para fotos
3. **Nomenclatura**: Use exatamente os nomes listados acima para que as referências no LaTeX funcionem
4. **Placeholders**: Se uma imagem ainda não foi gerada, você pode usar um placeholder temporário ou comentar a figura no LaTeX

## Script para Copiar Imagens

Crie um script para facilitar a cópia das imagens:

```bash
#!/bin/bash
# Copiar imagens geradas para relatorio/figs/

mkdir -p relatorio/figs

# Dataset samples
cp codes/figs/corel_dataset_samples.png relatorio/figs/ 2>/dev/null

# LoRA generated
cp codes/generated_images/corel_all_grid_*.png relatorio/figs/corel_lora_generated_grid.png 2>/dev/null

# Clustering comparison
cp codes/clustering_results/clustering_comparison.png relatorio/figs/ 2>/dev/null

echo "Imagens copiadas para relatorio/figs/"
```

