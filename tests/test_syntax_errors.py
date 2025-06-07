import operator
from typing import NamedTuple

import pytest

from micro_liquid import TemplateSyntaxError
from micro_liquid import Template


class Case(NamedTuple):
    description: str
    template: str
    expect_msg: str


TEST_CASES = [
    Case(
        description="no expression",
        template="{% if %}foo{% endif %}",
        expect_msg="missing expression",
    ),
    Case(
        description="end tag mismatch",
        template="{% if true %}foo{% endfor %}",
        expect_msg="unexpected tag 'endfor'",
    ),
    Case(
        description="unexpected tag",
        template="{% foo true %}foo{% endfoo %}",
        expect_msg="unknown, missing or malformed tag name",
    ),
    Case(
        description="missing tag name",
        template="{% %}foo{% endif %}",
        expect_msg="unknown, missing or malformed tag name",
    ),
    Case(
        description="missing end tag at EOF",
        template="{% if true %}foo{{ bar }}",
        expect_msg="expected tag 'endif'",
    ),
    # Case(
    #     description="orphaned break",
    #     template="{% break %}",
    #     expect_msg="unexpected 'break'",
    # ),
    # Case(
    #     description="orphaned continue",
    #     template="{% continue %}",
    #     expect_msg="unexpected 'continue'",
    # ),
    Case(
        description="orphaned else",
        template="{% else %}",
        expect_msg="unexpected tag 'else'",
    ),
    Case(
        description="missing 'in' in forloop",
        template="{% for x y %}{{ x }}{% endfor %}",
        expect_msg="missing 'in'",
    ),
    Case(
        description="missing target in forloop",
        template="{% for x in %}{{ x }}foo{% endfor %}",
        expect_msg="missing expression",
    ),
    Case(
        description="chained identifier for loop variable",
        template="{% for x.y in z %}{{ x }}{% endfor %}",
        expect_msg="expected an identifier, found a path",
    ),
    Case(
        description="invalid bracketed segment",
        template="{{ foo[1.2] }}",
        expect_msg="unexpected '.'",
    ),
    Case(
        description="unknown prefix operator",
        template="{{ +5 }}",
        expect_msg=r"unexpected '\+'",
    ),
    Case(
        description="unknown symbol",
        template="{% if 1 * 2 %}ok{% endif %}",
        expect_msg="unexpected '*'",
    ),
    Case(
        description="bad alternative condition expression",
        template="{% if false %}ok{% elsif 1*=2 %}not ok{% endif %}",
        expect_msg="unexpected '*'",
    ),
    Case(
        description="bad token in loop expression",
        template="{% for i$ in j %}{% endfor %}",
        expect_msg=r"unexpected '\$'",
    ),
    Case(
        description="missing output closing bracket",
        template=r"Hello, {{you}!",
        expect_msg="incomplete markup detected",
    ),
    Case(
        description="missing tag closing percent",
        template=r"{% if x %}foo{% endif }!",
        expect_msg="incomplete markup detected",
    ),
    Case(
        description="missing tag closing bracket",
        template=r"{% if x %}foo{% endif %!",
        expect_msg="incomplete markup detected",
    ),
    Case(
        description="path, empty brackets",
        template=r"{{ a.b[] }}",
        expect_msg="empty bracketed segment",
    ),
    Case(
        description="path, unbalanced brackets",
        template=r"{{ a.b['foo']] }}",
        expect_msg="unexpected ']'",
    ),
]


@pytest.mark.parametrize("case", TEST_CASES, ids=operator.attrgetter("description"))
def test_syntax_errors(case: Case) -> None:
    with pytest.raises(TemplateSyntaxError, match=case.expect_msg):
        Template(case.template)
