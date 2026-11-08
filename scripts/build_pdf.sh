#!/bin/bash
# docs/RAPORTI.md -> docs/RAPORTI.pdf
# Uses pandoc + xelatex when available; otherwise pandoc + Typst
# (pip install pypandoc_binary typst; fonts: DejaVu Serif / Sans Mono / Math TeX Gyre).
cd "$(dirname "$0")/../docs"
if command -v xelatex > /dev/null; then
  pandoc RAPORTI.md -o RAPORTI.pdf --pdf-engine=xelatex -V fontfamily=fontspec \
    -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V mathfont="DejaVu Math TeX Gyre" \
    -V colorlinks=true -V linkcolor=blue --resource-path=.:..
else
  PANDOC=$(command -v pandoc || python3 -c "import pypandoc; print(pypandoc.get_pandoc_path())")
  printf 'margin:\n  x: 2.3cm\n  y: 2.3cm\nmainfont: DejaVu Serif\nmathfont: DejaVu Math TeX Gyre\ncodefont: DejaVu Sans Mono\npapersize: a4\nfontsize: 11pt\nsection-numbering: "1.1"\n' > _meta.yaml
  "$PANDOC" RAPORTI.md -s -t typst --metadata-file=_meta.yaml -o _raporti.typ
  python3 -c "import typst, os; typst.compile('_raporti.typ', output='RAPORTI.pdf', root='..', font_paths=[os.path.expanduser('~/Library/Fonts'), '/usr/share/fonts'])"
  rm -f _meta.yaml _raporti.typ
fi
