"""Build bangla_token_tax.ipynb from the percent-format source src/bangla_token_tax.py."""
import os, re, sys
import nbformat

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'bangla_token_tax.py'), encoding='utf-8').read()
out = os.path.join(here, '..', 'bangla_token_tax.ipynb')

cells = []
for block in re.split(r'^# %%', src, flags=re.M)[1:]:
    header, _, body = block.partition('\n')
    body = body.strip('\n')
    if header.strip() == '[markdown]':
        md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body))

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'language_info': {'name': 'python'}})
nbformat.write(nb, out)
print(f'wrote {os.path.normpath(out)} with {len(cells)} cells')
