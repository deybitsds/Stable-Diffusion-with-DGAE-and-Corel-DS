# 🔧 Solución de Errores de Compilación

## Error: "Unknown option 'portuguese'"

**Solución**: Ya está corregido. El archivo `includes.tex` ahora usa `english` por defecto.

Para usar português, instala primero:
```bash
sudo pacman -S texlive-lang
```

Luego cambia en `content/includes.tex`:
```latex
\usepackage[english]{babel}  % Cambiar esto
```
por:
```latex
\usepackage[portuguese]{babel}
```

## Error: "File 'breakcites.sty' not found"

**Solución**: Ya está comentado en `includes.tex`. No es crítico para la compilación.

## Error: "I can't find the format file 'Cursor-2.1.42-x86_64.AppImage.fmt'"

Este es un problema de configuración del sistema, no del código LaTeX.

### Solución 1: Usar ruta completa

```bash
cd relatorio
/usr/bin/pdflatex main.tex
/usr/bin/pdflatex main.tex
/usr/bin/pdflatex main.tex
```

### Solución 2: Actualizar PATH

```bash
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd relatorio
./compile.sh
```

### Solución 3: Compilar manualmente

```bash
cd relatorio

# Compilar 3 veces
/usr/bin/pdflatex -interaction=nonstopmode main.tex
/usr/bin/pdflatex -interaction=nonstopmode main.tex
/usr/bin/pdflatex -interaction=nonstopmode main.tex
```

## Verificar que Compiló Correctamente

Después de compilar, verifica que el PDF se generó:

```bash
ls -lh main.pdf
```

Si el archivo existe y tiene tamaño > 0, ¡compiló correctamente! 🎉

## Si Persisten los Problemas

1. **Instalar paquetes faltantes**:
   ```bash
   sudo pacman -S texlive-most texlive-lang
   ```

2. **Verificar instalación de LaTeX**:
   ```bash
   pdflatex --version
   ```

3. **Compilar con salida completa para ver errores**:
   ```bash
   cd relatorio
   /usr/bin/pdflatex main.tex
   ```
   (Esto mostrará todos los errores en detalle)

