from skill_rag.models import SearchHit, SkillRecord


def test_skill_record_embed_text_includes_body():
    # The body carries trigger phrases and examples that the one-line
    # description omits, so it must be part of the embedded text.
    r = SkillRecord(
        name="brainstorming",
        description="explore ideas",
        path="/tmp/.skills/brainstorming/SKILL.md",
        body="Use when starting a new feature before writing code.",
        content_hash="abc",
    )
    text = r.embed_text()
    assert text.startswith("brainstorming\nexplore ideas")
    assert "Use when starting a new feature" in text


def test_search_hit_fields():
    h = SearchHit(name="x", description="y", score=0.9)
    assert h.name == "x"
    assert h.description == "y"
    assert h.score == 0.9
