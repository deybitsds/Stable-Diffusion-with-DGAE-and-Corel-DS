#!/bin/bash
# Script para compilar el relatorio LaTeX

cd "$(dirname "$0")"

echo "=========================================="
echo "Compilando relatorio LaTeX..."
echo "=========================================="

# Primera pasada: genera referencias cruzadas
echo ""
echo "Pasada 1/3: Compilando documento..."
# Intentar diferentes rutas de pdflatex
PDFLATEX_CMD=$(which pdflatex 2>/dev/null || echo "/usr/bin/pdflatex")
$PDFLATEX_CMD -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error en primera pasada. Revisa los errores:"
    $PDFLATEX_CMD -interaction=nonstopmode main.tex
    exit 1
fi
echo "✓ Primera pasada completada"

# Segunda pasada: procesa bibliografía (si usa natbib/bibtex)
echo ""
echo "Pasada 2/3: Procesando bibliografía..."
$PDFLATEX_CMD -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error en segunda pasada. Revisa los errores:"
    $PDFLATEX_CMD -interaction=nonstopmode main.tex
    exit 1
fi
echo "✓ Segunda pasada completada"

# Tercera pasada: resuelve todas las referencias
echo ""
echo "Pasada 3/3: Resolviendo referencias finales..."
$PDFLATEX_CMD -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error en tercera pasada. Revisa los errores:"
    $PDFLATEX_CMD -interaction=nonstopmode main.tex
    exit 1
fi
echo "✓ Tercera pasada completada"

# Limpiar archivos auxiliares (opcional)
echo ""
read -p "¿Limpiar archivos auxiliares (.aux, .log, .toc, etc.)? [s/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    rm -f *.aux *.log *.toc *.lof *.lot *.out *.bbl *.blg *.nav *.snm *.vrb
    echo "✓ Archivos auxiliares eliminados"
fi

echo ""
echo "=========================================="
echo "✅ Compilación completada!"
echo "=========================================="
echo "PDF generado: main.pdf"
echo ""

