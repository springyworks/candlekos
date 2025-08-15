#!/usr/bin/env python3
import json
import sys
from pathlib import Path

LANG_COMMENT = {
    'rust': '// Cell {n} -',
    'python': '# Cell {n} -',
    'javascript': '// Cell {n} -',
    'markdown': '### Cell {n} -',
}

def annotate(path: Path):
    data = json.loads(path.read_text())
    cells = data.get('cells', [])
    # backup
    backup = path.with_suffix(path.suffix + '.bak')
    backup.write_text(json.dumps(data, indent=2))
    for i, cell in enumerate(cells, start=1):
        ctype = cell.get('cell_type')
        meta = cell.get('metadata', {})
        lang = meta.get('language') or ( 'markdown' if ctype=='markdown' else 'python')
        prefix = LANG_COMMENT.get(lang, '// Cell {n} -')
        line = prefix.format(n=i)
        src = cell.get('source', [])
        # avoid double-annotating if already annotated
        if src and isinstance(src, list) and src[0].lstrip().startswith(line.split()[0].rstrip('-')):
            continue
        if ctype == 'markdown':
            # insert heading as first line
            src.insert(0, line + '\n')
        else:
            # for code cells insert comment line
            src.insert(0, line + '\n')
        cell['source'] = src
    data['cells'] = cells
    path.write_text(json.dumps(data, indent=2))
    print(f"Annotated {len(cells)} cells in {path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: annotate_notebook_cells.py path/to/notebook.ipynb')
        sys.exit(2)
    p = Path(sys.argv[1])
    if not p.exists():
        print('file not found:', p)
        sys.exit(2)
    annotate(p)
