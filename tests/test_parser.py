from pathlib import Path

from skill_rag.parser import parse_skill_file


def _write(tmp_path: Path, name: str, content: str) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    p = d / "SKILL.md"
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_valid(tmp_path):
    p = _write(
        tmp_path,
        "foo",
        "---\nname: foo\ndescription: does foo\n---\nBody here.\n",
    )
    r = parse_skill_file(p)
    assert r is not None
    assert r.name == "foo"
    assert r.description == "does foo"
    assert r.body == "Body here.\n"
    assert r.content_hash  # non-empty
    assert r.path == str(p)


def test_parse_missing_frontmatter(tmp_path):
    p = _write(tmp_path, "foo", "no frontmatter here")
    assert parse_skill_file(p) is None


def test_parse_missing_required_fields(tmp_path):
    p = _write(tmp_path, "foo", "---\nname: foo\n---\nbody")
    assert parse_skill_file(p) is None


def test_parse_malformed_yaml(tmp_path):
    p = _write(tmp_path, "foo", "---\nname: [unclosed\n---\nbody")
    assert parse_skill_file(p) is None


def test_parse_empty_file(tmp_path):
    p = _write(tmp_path, "foo", "")
    assert parse_skill_file(p) is None


def test_hash_changes_with_body(tmp_path):
    p1 = _write(tmp_path / "a", "foo", "---\nname: foo\ndescription: d\n---\nbody1")
    p2 = _write(tmp_path / "b", "foo", "---\nname: foo\ndescription: d\n---\nbody2")
    r1 = parse_skill_file(p1)
    r2 = parse_skill_file(p2)
    assert r1.content_hash != r2.content_hash
