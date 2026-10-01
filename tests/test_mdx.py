from docrag.ingest.mdx import normalize_mdx


def test_frontmatter_extracted() -> None:
    doc = normalize_mdx("---\ntitle: Hooks\ndescription: About hooks.\n---\n\nBody.\n")
    assert doc.title == "Hooks"
    assert doc.description == "About hooks."
    assert doc.text.strip() == "Body."


def test_callout_becomes_labelled_text() -> None:
    doc = normalize_mdx('<Callout type="warn">\n\nBe careful.\n\n</Callout>\n')
    assert "**Warning:**" in doc.text
    assert "Be careful." in doc.text
    assert "Callout" not in doc.text


def test_multiline_card_becomes_bullet() -> None:
    src = (
        '<Cards>\n  <Card\n    title="Uniswap v3"\n    href="/docs/protocols/v3/overview"\n'
        '    description="Concentrated liquidity."\n  />\n</Cards>\n'
    )
    doc = normalize_mdx(src)
    assert "- **Uniswap v3**: Concentrated liquidity. (/docs/protocols/v3/overview)" in doc.text
    assert "<" not in doc.text


def test_tab_headings_get_tab_suffix() -> None:
    src = (
        '<ImplementationTabs>\n<ImplementationTab value="api">\n\n### Step 1\n\nA.\n\n'
        '</ImplementationTab>\n<ImplementationTab value="sdk">\n\n### Step 1\n\nB.\n\n'
        "</ImplementationTab>\n</ImplementationTabs>\n\n### After\n"
    )
    doc = normalize_mdx(src)
    assert "### Step 1 (api)" in doc.text
    assert "### Step 1 (sdk)" in doc.text
    assert "### After\n" in doc.text  # outside tabs: unchanged


def test_code_fences_untouched() -> None:
    src = '```tsx\nimport x from "y";\n<Callout type="warn">\n{/* keep */}\n```\n'
    doc = normalize_mdx(src)
    assert doc.text == src


def test_mdx_comments_and_layout_tags_removed() -> None:
    src = "{/* hidden */}\n<div style={{ a: 1 }}>\nVisible.\n</div>\n<summary>Q?</summary>\n"
    doc = normalize_mdx(src)
    assert "hidden" not in doc.text
    assert "div" not in doc.text
    assert "Visible." in doc.text
    assert "**Q?**" in doc.text


def test_unterminated_fence_closed_at_eof() -> None:
    doc = normalize_mdx("Text:\n\n```solidity\nfunction f() external;\n")
    assert doc.text.rstrip().endswith("```")
