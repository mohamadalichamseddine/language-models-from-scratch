import json

import pytest

from lm.reproducibility import bundle_sources, read_numeric_csv


def test_bundle_is_sorted_repeatable_and_excludes_private_data_and_notebook_outputs(tmp_path):
    for folder in ("src/lm", "scripts", "notebooks", "docs/internal", "results", ".venv"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "src/lm/z.py").write_bytes(b"print('z')\r\n")
    (tmp_path / "scripts/a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy==2.4.6\n", encoding="utf-8")
    for name in (".env", "AGENTS.md", "docs/internal/private.py", ".venv/ignored.py", "results/ignored.py"):
        (tmp_path / name).write_text("PRIVATE_SENTINEL", encoding="utf-8")
    notebook = {"cells": [
        {"cell_type": "markdown", "source": ["OMIT_MARKDOWN"]},
        {"cell_type": "code", "source": ["print('notebook')\n"], "outputs": [{"text": "OMIT_OUTPUT"}]},
    ]}
    (tmp_path / "notebooks/demo.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    first, second = tmp_path / "one.txt", tmp_path / "two.txt"
    assert bundle_sources(tmp_path, first) == 4
    assert bundle_sources(tmp_path, second) == 4
    assert first.read_bytes() == second.read_bytes()
    assert b"\r" not in first.read_bytes()
    text = first.read_text(encoding="utf-8")
    assert "print('notebook')" in text
    assert all(word not in text for word in ("PRIVATE_SENTINEL", "OMIT_MARKDOWN", "OMIT_OUTPUT"))
    assert text.index("FILE: scripts/a.py") < text.index("FILE: src/lm/z.py")
    with pytest.raises(ValueError, match="overwrite"):
        bundle_sources(tmp_path, tmp_path / "scripts/a.py")


def test_numeric_csv_keeps_labels_and_counts(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text('group,tokens,gain\n"word, rare",3,-0.125\n', encoding="utf-8")
    assert read_numeric_csv(path, {"group"}, {"tokens"}) == [{"group": "word, rare", "tokens": 3, "gain": -0.125}]
    path.write_text("group,tokens,gain\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No rows"):
        read_numeric_csv(path, {"group"})
