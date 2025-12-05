#!/bin/bash
# Script para instalar dependencias para entrenamiento LoRA con Corel
# Ejecutar con: bash install_dependencies.sh

echo "Instalando dependencias para Corel LoRA Training..."

# Verificar si estamos en un entorno conda
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Entorno conda detectado: $CONDA_DEFAULT_ENV"
    echo "Instalando con pip en el entorno conda..."
    
    pip install datasets
    pip install diffusers
    pip install accelerate
    pip install peft
    pip install transformers
    pip install bitsandbytes
    pip install tqdm
    
    echo "✓ Dependencias instaladas"
else
    echo "⚠ No se detectó entorno conda activo"
    echo "Por favor activa tu entorno conda primero:"
    echo "  conda activate mo433"
    echo "Luego ejecuta: pip install -r requirements_corel.txt"
fi

