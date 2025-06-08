import operator
from typing import TypedDict

import pytest

from micro_liquid import Template


class Case(TypedDict):
    name: str
    template: str
    data: dict[str, object]
    result: str


TEST_CASES: list[Case] = [
    {
        "name": "logical and, last value, truthy left",
        "template": "{{ x and y }}",
        "data": {"x": True, "y": 42},
        "result": "42",
    },
    {
        "name": "logical and, last value, falsy left",
        "template": "{{ x and y }}",
        "data": {"x": False, "y": 42},
        "result": "False",
    },
    {
        "name": "logical or, last value, truthy left",
        "template": "{{ x or y }}",
        "data": {"x": 99, "y": 42},
        "result": "99",
    },
    {
        "name": "logical or, last value, falsy left",
        "template": "{{ x or y }}",
        "data": {"x": False, "y": 42},
        "result": "42",
    },
    {
        "name": "logical and, falsy left, or truthy",
        "template": "{{ x and y or z }}",
        "data": {"x": False, "y": 42, "z": 99},
        "result": "99",
    },
    {
        "name": "not binds more tightly than or",
        "template": "{{ not x or y  }}",
        "data": {"x": False, "y": True},
        "result": "True",
    },
    {
        "name": "and binds more tightly than or",
        "template": "{{ x or y and z  }}",
        "data": {"x": False, "y": True, "z": True},
        "result": "True",
    },
    {
        "name": "group terms with parentheses",
        "template": (
            "{% if (true and (false and (false or true))) %}a{% else %}b{% endif %}"
        ),
        "data": {"true": True, "false": False},
        "result": "b",
    },
    {
        "name": "more precedence",
        "template": "{% if false and false or true %}a{% else %}b{% endif %}",
        "data": {"true": True, "false": False},
        "result": "a",
    },
    {
        "name": "more grouping",
        "template": "{% if false and (false or true) %}a{% else %}b{% endif %}",
        "data": {"true": True, "false": False},
        "result": "b",
    },
    {
        "name": "loop target",
        "template": "{% for x in y or a %}{{ x }}, {% endfor %}",
        "data": {"a": [1, 2, 3]},
        "result": "1, 2, 3, ",
    },
]


@pytest.mark.parametrize("case", TEST_CASES, ids=operator.itemgetter("name"))
def test_compound_expressions(case: Case) -> None:
    assert Template(case["template"]).render(case["data"]) == case["result"]
