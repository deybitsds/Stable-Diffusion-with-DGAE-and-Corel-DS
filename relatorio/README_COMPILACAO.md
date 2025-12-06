# Como Compilar o Relatório

## Método 1: Script Automático (Recomendado)

```bash
cd relatorio
./compile.sh
```

O script faz automaticamente as 3 passadas necessárias para compilar corretamente o documento.

## Método 2: Compilação Manual

Se preferir compilar manualmente:

```bash
cd relatorio

# Passada 1: Compila o documento
pdflatex main.tex

# Passada 2: Resolve referências cruzadas
pdflatex main.tex

# Passada 3: Resolve todas as referências finais
pdflatex main.tex
```

## Método 3: Compilação com Visualização de Erros

Se quiser ver os erros em tempo real:

```bash
cd relatorio
pdflatex main.tex
# Se houver erros, corrija e repita
pdflatex main.tex
pdflatex main.tex
```

## Limpar Arquivos Auxiliares

Para limpar arquivos temporários gerados durante a compilação:

```bash
cd relatorio
rm -f *.aux *.log *.toc *.lof *.lot *.out *.bbl *.blg *.nav *.snm *.vrb
```

Ou use o script com a opção de limpeza automática.

## Solução de Problemas

### Erro: "File not found: content/includes.tex"
- Verifique se está no diretório `relatorio/`
- Verifique se o arquivo `content/includes.tex` existe

### Erro: "Missing \begin{document}"
- Verifique se há erros de sintaxe no `main.tex`
- Verifique se todos os arquivos `content/*.tex` existem

### Erro: "Citation undefined"
- Execute múltiplas passadas (3x) para resolver referências
- Verifique se as citações em `biblio.tex` correspondem às usadas no texto

### Erro: "Undefined control sequence"
- Pode faltar algum pacote LaTeX
- Instale pacotes faltantes: `sudo pacman -S texlive-most` (Arch Linux)

## Dependências LaTeX

O documento usa os seguintes pacotes principais:
- `babel` (português)
- `graphicx` (imagens)
- `natbib` (bibliografia)
- `amsmath` (fórmulas matemáticas)
- `tikz` (diagramas)
- E outros definidos em `content/includes.tex`

Se faltar algum pacote, instale com:
```bash
# Arch Linux
sudo pacman -S texlive-most

# Ubuntu/Debian
sudo apt-get install texlive-full

# Ou instale pacotes específicos conforme necessário
```

