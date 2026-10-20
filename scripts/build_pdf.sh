#!/bin/bash
# docs/RAPORTI.md -> docs/RAPORTI.pdf (needs pandoc + xelatex)
cd "$(dirname "$0")/../docs"
pandoc RAPORTI.md -o RAPORTI.pdf --pdf-engine=xelatex -V fontfamily=fontspec \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V mathfont="DejaVu Math TeX Gyre" \
  -V colorlinks=true -V linkcolor=blue --resource-path=.:..
