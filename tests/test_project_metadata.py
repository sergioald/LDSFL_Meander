from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import ldsfl


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _match(pattern: str, text: str, *, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match is not None, f"Could not find {label}"
    return match.group(1)


def _local_markdown_targets(text: str) -> list[str]:
    targets: list[str] = []
    for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = raw.strip().split()[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0])
        if target:
            targets.append(target)
    return targets


def test_version_metadata_is_consistent():
    pyproject = _read("pyproject.toml")
    citation = _read("CITATION.cff")
    readme = _read("README.md")

    project_version = _match(
        r'^version\s*=\s*"([^"]+)"',
        pyproject,
        label="pyproject version",
    )
    citation_version = _match(
        r"^version:\s*([^\s]+)",
        citation,
        label="CITATION.cff version",
    )
    readme_version = _match(
        r"version-v([0-9][0-9A-Za-z_.-]*)-blue",
        readme,
        label="README version badge",
    )

    assert ldsfl.__version__ == project_version
    assert citation_version == project_version
    assert readme_version == project_version


def test_readme_has_no_literal_newline_escape_artifact():
    readme = _read("README.md")
    assert r"\n\n## Documentation" not in readme
    assert r"\n- [Theory-to-code mapping]" not in readme


def test_readme_local_links_exist():
    readme = _read("README.md")
    missing = [
        target
        for target in _local_markdown_targets(readme)
        if not (ROOT / target).exists()
    ]
    assert missing == []


def test_documentation_index_local_links_exist():
    index_path = ROOT / "docs" / "index.md"
    text = index_path.read_text(encoding="utf-8")
    missing = [
        target
        for target in _local_markdown_targets(text)
        if not (index_path.parent / target).exists()
    ]
    assert missing == []


def test_citation_metadata_includes_repository_and_concept_doi():
    citation = _read("CITATION.cff")
    assert 'repository-code: "https://github.com/sergioald/LDSFL_Meander"' in citation
    assert 'value: "10.5281/zenodo.19945291"' in citation
    for given_name in ("Sergio", "Alessandro", "Stefano"):
        assert f'given-names: "{given_name}"' in citation


def test_source_distribution_manifest_keeps_key_research_software_files():
    manifest = _read("MANIFEST.in")
    required = {
        "include LICENSE",
        "include CITATION.cff",
        "include USER_MANUAL.md",
        "recursive-include Input *.csv",
        "recursive-include docs *.md *.tex *.pdf",
        "recursive-include examples *.md *.json *.csv",
        "prune Output",
    }
    assert required.issubset(set(manifest.splitlines()))
