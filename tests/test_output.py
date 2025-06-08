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
        "name": "dotted variable",
        "template": "{{ product.title }}",
        "data": {"product": {"title": "foo"}},
        "result": "foo",
    },
    {
        "name": "bracketed variable, single quotes",
        "template": "{{ product['title'] }}",
        "data": {"product": {"title": "foo"}},
        "result": "foo",
    },
    {
        "name": "bracketed variable, double quotes",
        "template": '{{ product["title"] }}',
        "data": {"product": {"title": "foo"}},
        "result": "foo",
    },
    {
        "name": "non existent property name with a space in it",
        "template": '{{ product["no such thing"] }}',
        "data": {"product": {"foo bar": "foo"}},
        "result": "",
    },
    {
        "name": "bracketed variable, empty quotes",
        "template": '{{ product[""] }}',
        "data": {"product": {"": "foo"}},
        "result": "foo",
    },
    {
        "name": "undefined variable",
        "template": "{{ nosuchthing }}",
        "data": {},
        "result": "",
    },
    {
        "name": "undefined property",
        "template": "{{ a.nosuchthing }}",
        "data": {"a": "b"},
        "result": "",
    },
    {
        "name": "access a list item by index",
        "template": "{{ product.tags[1] }}",
        "data": {"product": {"tags": ["sports", "garden"]}},
        "result": "garden",
    },
    {
        "name": "list index out of range",
        "template": "{{ product.tags[99] }}",
        "data": {"product": {"tags": ["sports", "garden"]}},
        "result": "",
    },
    {
        "name": "shorthand index",
        "template": "{{ product.tags.1 }}",
        "data": {"product": {"tags": ["sports", "garden"]}},
        "result": "garden",
    },
    {
        "name": "attempt to access a property of a list",
        "template": "{{ product.tags.foo }}",
        "data": {"product": {"tags": ["sports", "garden"]}},
        "result": "",
    },
    {
        "name": "access an array item by negative index",
        "template": "{{ product.tags[-2] }}",
        "data": {"product": {"tags": ["sports", "garden"]}},
        "result": "sports",
    },
    {
        "name": "dump an array from context",
        "template": "{{ a }}",
        "data": {"a": ["sports", "garden"]},
        "result": '["sports", "garden"]',
    },
]


@pytest.mark.parametrize("case", TEST_CASES, ids=operator.itemgetter("name"))
def test_output(case: Case) -> None:
    assert Template(case["template"]).render(case["data"]) == case["result"]
