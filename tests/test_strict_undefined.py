import pytest

from micro_liquid import StrictUndefined
from micro_liquid import UndefinedVariableError
from micro_liquid import render


def test_output_strict_undefined() -> None:
    with pytest.raises(UndefinedVariableError):
        render("{{ nosuchthing }}", data={}, undefined=StrictUndefined)


def test_strict_undefined_truthiness() -> None:
    result = render(
        "{% if nosuchthing %}true{% else %}false{% endif %}",
        data={},
        undefined=StrictUndefined,
    )
    assert result == "false"


def test_loop_over_strict_undefined() -> None:
    with pytest.raises(UndefinedVariableError):
        render(
            "{% for item in nosuchthing %}..{% endfor %}",
            data={},
            undefined=StrictUndefined,
        )
