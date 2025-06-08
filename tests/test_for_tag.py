import pytest

from micro_liquid import TemplateSyntaxError
from micro_liquid import render


def test_loop_over_a_list() -> None:
    source = "{% for x in y %}{{ x }}, {% endfor %}"
    data = {"y": [1, 2, 3]}
    assert render(source, data) == "1, 2, 3, "


def test_loop_over_a_string() -> None:
    source = "{% for x in y %}{{ x }}, {% endfor %}"
    data = {"y": "123"}
    assert render(source, data) == "1, 2, 3, "


def test_loop_over_a_dict() -> None:
    source = "{% for x in y %}({{ x.0 }}, {{ x.1 }}), {% endfor %}"
    data = {"y": {"a": 1, "b": 2, "c": 3}}
    assert render(source, data) == "(a, 1), (b, 2), (c, 3), "


def test_limit_is_not_allowed() -> None:
    source = "{% for x in y limit: 2 %}{% endfor %}"
    data = {"y": [1, 2, 3]}
    with pytest.raises(TemplateSyntaxError, match="unexpected ':'"):
        render(source, data)


def test_reversed_is_not_allowed() -> None:
    source = "{% for x in y reversed %}{% endfor %}"
    data = {"y": [1, 2, 3]}
    with pytest.raises(TemplateSyntaxError, match="unexpected 'reversed'"):
        render(source, data)


def test_else_empty_list() -> None:
    source = "{% for x in y %}{{ x }}, {% else %}default{% endfor %}"
    data: dict[str, object] = {"y": []}
    assert render(source, data) == "default"


def test_else_non_empty_list() -> None:
    source = "{% for x in y %}{{ x }}, {% else %}default{% endfor %}"
    data: dict[str, object] = {"y": [1, 2, 3]}
    assert render(source, data) == "1, 2, 3, "


def test_loop_var_goes_out_of_scope() -> None:
    source = "{% for x in y %}{{ x }}, {% endfor %}{{ x }}"
    data: dict[str, object] = {"y": [1, 2, 3]}
    assert render(source, data) == "1, 2, 3, "


def test_nested_loop() -> None:
    source = (
        "{% for a in b %}{% for x in y %}({{ a }}, {{ x }}), {% endfor %}{% endfor %}"
    )
    data: dict[str, object] = {"y": [1, 2, 3], "b": ["c", "d"]}
    assert render(source, data) == "(c, 1), (c, 2), (c, 3), (d, 1), (d, 2), (d, 3), "
