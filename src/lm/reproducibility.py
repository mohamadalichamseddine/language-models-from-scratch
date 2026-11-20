"""Portable report inputs and a deterministic bundle of project source code."""

import csv
import json
from pathlib import Path


def read_numeric_csv(path, text_columns, integer_columns=()):
    """Read saved report rows, preserving labels and integer counts."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return [{key: value if key in text_columns else int(value) if key in integer_columns else float(value)
             for key, value in row.items()} for row in rows]


def bundle_sources(root, out):
    """Bundle Python, notebook code cells and public configs; omit outputs and private files.

    Only explicit source directories and root packaging files are considered.
    Symlinks are rejected, and all text uses LF line endings in the bundle.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    paths = []
    for directory, suffix in (("src", ".py"), ("scripts", ".py"), ("tests", ".py"),
                              ("notebooks", ".ipynb"), ("configs", ".json")):
        base = root / directory
        if base.is_symlink():
            raise ValueError(f"Source directory is a symlink: {base}")
        paths.extend(path for path in base.rglob(f"*{suffix}") if path.is_file())
    paths.extend(root / name for name in ("pyproject.toml", "requirements.txt") if (root / name).is_file())
    if not paths:
        raise ValueError("No source files found")
    if out in [path.resolve() for path in paths]:
        raise ValueError("Bundle output must not overwrite a source file")
    sections = ["Language models from scratch — source bundle\n",
                "Includes Python, notebook code cells, configs and packaging. Notebook outputs are omitted.\n"]
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Source must be a regular file within the repository: {path}")
        name = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            notebook = json.loads(text)
            cells = []
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = cell["source"]
                    cells.append(f"# Cell {index}\n" + (source if isinstance(source, str) else "".join(source)))
            text = "\n\n".join(cells)
        sections.append(f"\n{'=' * 72}\nFILE: {name}\n{'=' * 72}\n{text.rstrip()}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(sections), encoding="utf-8", newline="\n")
    return len(paths)
