import pytest
import os
from src.utils.format_all_documentation import (
    align_gfm_table,
    sanitize_cell_pipes,
    normalize_links,
    format_markdown_content,
    verify_documentation_integrity,
)


def test_sanitize_cell_pipes():
    raw = "| `<|im_start|>assistant` | `<|im_end|>` |"
    san = sanitize_cell_pipes(raw)
    assert "<&#124;im_start&#124;>" in san
    assert "<&#124;im_end&#124;>" in san


def test_align_gfm_table_vertical_alignment():
    table = """| Track | Category | Dialogues | Accuracy |
| :--- | :---: | ---: | :--- |
| Track 1 | Reasoning | 130 | 50.77% |
| Track 2 | Multi-Turn Replay | 130 | 88.64% |
| Track 3 | Autonomous Exploit Trajectory | 390 | 97.60% |"""

    aligned = align_gfm_table(table)
    lines = aligned.strip().split("\n")
    assert len(lines) == 5

    # Check pipe positions match across all rows
    pipe_indices = [[i for i, ch in enumerate(l) if ch == "|"] for l in lines]
    first = pipe_indices[0]
    for idx, indices in enumerate(pipe_indices):
        assert indices == first, f"Row {idx} pipe positions mismatch: {indices} != {first}"

    # Verify separator row
    sep_cells = [c.strip() for c in lines[1].split('|')[1:-1]]
    assert sep_cells[0].startswith(':') and not sep_cells[0].endswith(':')  # left
    assert sep_cells[1].startswith(':') and sep_cells[1].endswith(':')      # center
    assert sep_cells[2].endswith(':') and not sep_cells[2].startswith(':')  # right
    assert sep_cells[3].startswith(':') and not sep_cells[3].endswith(':')  # left


def test_align_gfm_table_unescaped_pipes_in_cells():
    table = """| Field | Format |
| :--- | :--- |
| Token | `<|im_start|>assistant\\n` |"""

    aligned = align_gfm_table(table)
    lines = aligned.strip().split("\n")
    assert len(lines) == 3
    # Check that row 2 has exactly 2 columns, not broken by pipe
    row_cells = [c.strip() for c in lines[2].split("|")[1:-1]]
    assert len(row_cells) == 2
    assert "<&#124;im_start&#124;>" in row_cells[1]


def test_normalize_links():
    # If file exists, it should convert to relative path
    abs_clean_readme = os.path.abspath("data/clean/README.md").replace("\\", "/")
    content = f"[Clean Data](file:///{abs_clean_readme})"
    norm = normalize_links(content, "data/README.md")
    assert "file:///" not in norm
    assert "[Clean Data](clean/README.md)" in norm or "[Clean Data](clean\\README.md)" in norm

    # If file does not exist, it should render as code span
    content_fake = "[missing_script](file:///non_existent_folder/missing.py)"
    norm_fake = normalize_links(content_fake, "data/README.md")
    assert "file:///" not in norm_fake
    assert "`missing_script`" in norm_fake


def test_format_markdown_preserves_code_blocks():
    doc = """# Header

```bash
| Not a table | Inside code |
```

| Genuine | Table |
| :--- | :--- |
| Key | Val |
"""
    formatted = format_markdown_content(doc)
    assert "| Not a table | Inside code |" in formatted
    assert "| Genuine | Table |" in formatted


def test_verify_documentation_integrity():
    total, issues = verify_documentation_integrity(".")
    assert total > 0
    assert issues == [], f"Found integrity issues: {issues}"
