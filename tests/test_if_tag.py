import pytest

from micro_liquid import TemplateSyntaxError
from micro_liquid import render


def test_else() -> None:
    source = "{% if a %}a{% elif b %}b{% else %}c{% endif %}"
    data = {"a": False, "b": False}
    assert render(source, data) == "c"


def test_else_block() -> None:
    source = "{% if a %}a{% elif b %}b{% else %}{{ c }}{% endif %}"
    data = {"a": False, "b": False, "c": "d"}
    assert render(source, data) == "d"


def test_elif() -> None:
    source = "{% if a %}a{% elif b %}b{% else %}c{% endif %}"
    data = {"a": False, "b": True}
    assert render(source, data) == "b"


def test_elsif() -> None:
    source = "{% if a %}a{% elsif b %}b{% else %}c{% endif %}"
    data = {"a": False, "b": True}
    assert render(source, data) == "b"


def test_nested_if() -> None:
    source = "{% if a %}a{% if b %}b{% else %}c{% endif %}{% else %}d{% endif %}"
    data = {"a": True, "b": False}
    assert render(source, data) == "ac"


def test_too_many_else_tags() -> None:
    source = "{% if a %}a{% else %}b{% else %}c{% endif %}"
    data = {"a": False, "b": False}
    with pytest.raises(TemplateSyntaxError, match="unexpected tag 'else'"):
        render(source, data)
