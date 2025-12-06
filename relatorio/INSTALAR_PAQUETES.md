# 📦 Instalación de Paquetes LaTeX Necesarios

## Problema

El relatório necesita paquetes adicionales de LaTeX que no están instalados por defecto.

## Solución Rápida

### Arch Linux

```bash
# Instalar soporte para português y otros idiomas
sudo pacman -S texlive-lang

# Si falta algún otro paquete, instalar el grupo completo
sudo pacman -S texlive-most
```

### Ubuntu/Debian

```bash
# Instalar soporte para português
sudo apt-get install texlive-lang-portuguese

# O instalar el paquete completo
sudo apt-get install texlive-full
```

## Después de Instalar

1. **Cambiar babel a português**: Edita `content/includes.tex` y cambia:
   ```latex
   \usepackage[english]{babel}
   ```
   por:
   ```latex
   \usepackage[portuguese]{babel}
   ```

2. **Compilar de nuevo**:
   ```bash
   ./compile.sh
   ```

## Solución Temporal (Sin Instalar Paquetes)

Si no puedes instalar paquetes ahora, el documento compilará con `english` en lugar de `portuguese`. El contenido seguirá siendo en português, solo que algunas características de babel (como separación silábica) no funcionarán perfectamente.

Para compilar sin instalar nada adicional:
```bash
cd relatorio
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

