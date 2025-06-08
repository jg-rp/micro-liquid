from micro_liquid import render


def test_whitespace_control() -> None:
    source = "\n".join(
        [
            "<ul>",
            "{% for x in y ~%}",
            "    <li>{{ x }}</li>",
            "{% endfor -%}",
            "</ul>",
        ]
    )

    expect = "\n".join(
        [
            "<ul>",
            "    <li>1</li>",
            "    <li>2</li>",
            "    <li>3</li>",
            "    <li>4</li>",
            "</ul>",
        ]
    )

    data = {"y": [1, 2, 3, 4]}
    assert render(source, data) == expect
