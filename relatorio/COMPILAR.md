# 📄 Como Compilar o Relatório

## 🚀 Método Rápido (Recomendado)

```bash
cd relatorio
./compile.sh
```

## 📝 Método Manual

Se o script não funcionar, compile manualmente:

```bash
cd relatorio

# Compilar 3 vezes (necessário para referências cruzadas)
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

O PDF será gerado como `main.pdf` no diretório `relatorio/`.

## 🔧 Solução de Problemas

### Se aparecer erro de "format file not found"

Tente usar o caminho completo do pdflatex:

```bash
/sbin/pdflatex main.tex
/sbin/pdflatex main.tex
/sbin/pdflatex main.tex
```

### Se faltar algum pacote LaTeX

Instale os pacotes necessários:

```bash
# Arch Linux
sudo pacman -S texlive-most texlive-lang

# Ubuntu/Debian
sudo apt-get install texlive-full texlive-lang-portuguese
```

### Se houver erros de compilação

1. Verifique se todos os arquivos em `content/` existem
2. Verifique se há erros de sintaxe (parênteses, chaves, etc.)
3. Compile com saída completa para ver os erros:
   ```bash
   pdflatex main.tex
   ```

## 📊 Estrutura de Compilação

O LaTeX precisa de múltiplas passadas para:
- **Passada 1**: Compila o documento e gera referências
- **Passada 2**: Resolve referências cruzadas
- **Passada 3**: Resolve todas as referências finais

## ✅ Verificação

Após compilar, verifique se o arquivo `main.pdf` foi criado:

```bash
ls -lh main.pdf
```

Se o PDF foi gerado, está tudo certo! 🎉

