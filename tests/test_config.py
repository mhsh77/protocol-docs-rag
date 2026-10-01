from docrag.config import CorpusConfig


def make_cfg() -> CorpusConfig:
    return CorpusConfig(
        name="t",
        repo="Org/docs",
        commit="abc123",
        license="MIT",
        include_prefixes=["content/protocols/", "content/sdks/"],
        exclude_prefixes=["content/protocols/archive/"],
        strip_prefix="content/",
        published_url_template="https://example.org/docs/{path}",
    )


def test_wants_filters_prefix_extension_and_exclusions() -> None:
    cfg = make_cfg()
    assert cfg.wants("content/protocols/v4/overview.mdx")
    assert cfg.wants("content/sdks/v3/guide.md")
    assert not cfg.wants("content/protocols/v4/image.png")
    assert not cfg.wants("content/changelog/x.mdx")
    assert not cfg.wants("content/protocols/archive/old.mdx")


def test_published_url_strips_prefix_extension_and_index() -> None:
    cfg = make_cfg()
    assert cfg.published_url("content/protocols/v4/overview.mdx") == (
        "https://example.org/docs/protocols/v4/overview"
    )
    assert cfg.published_url("content/protocols/v3/deployments/index.mdx") == (
        "https://example.org/docs/protocols/v3/deployments"
    )


def test_source_url_is_pinned_to_commit() -> None:
    assert make_cfg().source_url("content/a.mdx") == (
        "https://github.com/Org/docs/blob/abc123/content/a.mdx"
    )
