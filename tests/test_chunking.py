"""Chunker tests on tricky inputs. Token count = whitespace words, for determinism."""

from __future__ import annotations

import itertools
import re

from docrag.ingest.chunking import Chunk, ChunkingConfig, chunk_document


def words(s: str) -> int:
    return len(s.split())


def run(text: str, cfg: ChunkingConfig, title: str | None = "Doc") -> list[Chunk]:
    return chunk_document(
        text=text,
        title=title,
        doc_id="content/x.mdx",
        doc_slug="x",
        url="https://example.org/docs/x",
        source_url="https://github.com/o/r/blob/c/content/x.mdx",
        cfg=cfg,
        count=words,
    )


def fences_balanced(text: str) -> bool:
    return len(re.findall(r"^\s*```", text, flags=re.M)) % 2 == 0


def test_heading_paths_follow_nesting() -> None:
    text = "# Doc\n\nIntro.\n\n## Pool\n\nA.\n\n### Supply\n\nB.\n\n## Router\n\nC.\n"
    chunks = run(text, ChunkingConfig(max_tokens=50, overlap_tokens=0))
    paths = [c.heading_str for c in chunks]
    assert paths == ["Doc", "Doc > Pool", "Doc > Pool > Supply", "Doc > Router"]


def test_small_code_block_kept_whole_even_if_over_budget() -> None:
    code = "```solidity\n" + "\n".join(f"uint x{i} = {i};" for i in range(40)) + "\n```"
    text = f"## Code\n\nBefore.\n\n{code}\n\nAfter.\n"
    cfg = ChunkingConfig(max_tokens=30, overlap_tokens=0, max_code_tokens=500)
    chunks = run(text, cfg)
    code_chunks = [c for c in chunks if "```solidity" in c.text]
    assert len(code_chunks) == 1
    assert code.strip() in code_chunks[0].text
    assert all(fences_balanced(c.text) for c in chunks)


def test_huge_code_block_split_only_at_blank_lines_and_refenced() -> None:
    funcs = [f"function f{i}() {{\n  return {i};\n}}\n" for i in range(30)]
    code = "```js\n" + "\n".join(funcs) + "```"
    cfg = ChunkingConfig(max_tokens=40, overlap_tokens=0, max_code_tokens=40)
    chunks = run(f"## Big\n\n{code}\n", cfg)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("```js")
        assert c.text.rstrip().endswith("```")
        assert fences_balanced(c.text)
        # every function body is complete inside one chunk
        assert c.text.count("{") == c.text.count("}")


def test_fence_containing_heading_like_lines_is_not_a_heading() -> None:
    text = "## Real\n\n```bash\n# not a heading\necho hi\n```\n"
    chunks = run(text, ChunkingConfig(max_tokens=100))
    assert [c.heading_str for c in chunks] == ["Doc > Real"]
    assert "# not a heading" in chunks[0].text


def test_large_table_split_by_rows_with_header_repeated() -> None:
    header = "| Network | Address |\n| --- | --- |"
    rows = "\n".join(f"| chain{i} | 0x{i:040x} |" for i in range(60))
    cfg = ChunkingConfig(max_tokens=60, overlap_tokens=0)
    chunks = run(f"## Deployments\n\n{header}\n{rows}\n", cfg)
    assert len(chunks) > 1
    seen = []
    for c in chunks:
        lines = c.text.split("\n")
        assert lines[:2] == header.split("\n")
        assert all(ln.startswith("|") and ln.endswith("|") for ln in lines)
        seen += lines[2:]
    assert seen == rows.split("\n")  # every row exactly once, in order


def test_small_table_kept_intact() -> None:
    table = "| a | b |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |"
    chunks = run(f"## T\n\nText.\n\n{table}\n", ChunkingConfig(max_tokens=100))
    assert table in chunks[0].text


def test_nested_list_splits_only_between_top_level_items() -> None:
    items = []
    for i in range(12):
        items.append(
            f"- Item {i} has words here\n  - child {i}a with more words\n"
            f"    - grandchild {i}a deep\n  - child {i}b"
        )
    cfg = ChunkingConfig(max_tokens=40, overlap_tokens=0)
    chunks = run("## List\n\n" + "\n".join(items) + "\n", cfg)
    assert len(chunks) > 1
    for c in chunks:
        # each top-level item appears with all its children in the same chunk
        for i in range(12):
            if f"- Item {i} " in c.text:
                assert f"child {i}a" in c.text
                assert f"grandchild {i}a" in c.text
                assert f"child {i}b" in c.text


def test_overlap_stays_within_section_and_respects_budget() -> None:
    para = " ".join(f"Sentence {i} is here." for i in range(30))
    text = f"## A\n\n{para}\n\n## B\n\nOther section text.\n"
    cfg = ChunkingConfig(max_tokens=40, overlap_tokens=8)
    chunks = run(text, cfg)
    a = [c for c in chunks if c.heading_str == "Doc > A"]
    b = [c for c in chunks if c.heading_str == "Doc > B"]
    assert len(a) > 1 and len(b) == 1
    assert "Sentence" not in b[0].text
    for prev, nxt in itertools.pairwise(a):
        last_sentence = prev.text.split(". ")[-1]
        assert last_sentence in nxt.text  # overlap carried forward
    assert all(c.n_tokens <= cfg.max_tokens for c in a)


def test_every_chunk_within_budget_except_atomic_code() -> None:
    para = " ".join(["word"] * 200) + "."
    chunks = run(f"## P\n\n{para}\n", ChunkingConfig(max_tokens=50, overlap_tokens=0))
    # a single giant sentence cannot be split further; it is kept rather than cut mid-sentence
    assert "".join(c.text for c in chunks).count("word") == 200


def test_chunk_ids_are_deterministic_and_unique() -> None:
    text = "## A\n\nOne.\n\n## B\n\nTwo.\n"
    c1 = run(text, ChunkingConfig())
    c2 = run(text, ChunkingConfig())
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2] == ["x#000", "x#001"]


def test_duplicate_title_h1_not_repeated_in_path() -> None:
    chunks = run("# Doc\n\nBody.\n", ChunkingConfig())
    assert chunks[0].heading_path == ["Doc"]


def test_lead_in_sentence_glued_to_following_code() -> None:
    code = "```js\n" + "\n".join(f"const a{i} = {i};" for i in range(30)) + "\n```"
    text = (
        f"## Review\n\n{' '.join(['filler'] * 25)}.\n\nYour file should look like this:\n\n{code}\n"
    )
    chunks = run(text, ChunkingConfig(max_tokens=40, overlap_tokens=0, max_code_tokens=500))
    code_chunk = next(c for c in chunks if "```js" in c.text)
    assert code_chunk.text.startswith("Your file should look like this:")
    assert not any(c.text.strip() == "Your file should look like this:" for c in chunks)


def test_small_sibling_sections_merged_with_inline_subheadings() -> None:
    text = (
        "## Steps\n\nIntro text here.\n\n### Step 1\n\nPick a flow.\n\n"
        "### Step 2\n\nBuild the URL.\n\n## Other\n\n" + " ".join(["long"] * 60) + ".\n"
    )
    cfg = ChunkingConfig(max_tokens=100, overlap_tokens=0, min_section_tokens=20)
    chunks = run(text, cfg)
    steps = [c for c in chunks if "Pick a flow." in c.text]
    assert len(steps) == 1
    merged = steps[0]
    assert merged.heading_path == ["Doc", "Steps"]
    assert "**Step 1**\n\nPick a flow." in merged.text
    assert "**Step 2**\n\nBuild the URL." in merged.text
    assert "Intro text here." in merged.text
    # unrelated, large section stays separate
    assert any(c.heading_path == ["Doc", "Other"] for c in chunks)


def test_cousin_sections_not_merged() -> None:
    text = "## A\n\n### X\n\nx.\n\n## B\n\n### Y\n\ny.\n"
    cfg = ChunkingConfig(max_tokens=100, overlap_tokens=0, min_section_tokens=50)
    chunks = run(text, cfg)
    x = next(c for c in chunks if "x." in c.text)
    assert "y." not in x.text


def test_merging_disabled_by_default() -> None:
    text = "## A\n\nOne.\n\n## B\n\nTwo.\n"
    assert len(run(text, ChunkingConfig())) == 2
