<h1 align="center">Micro Liquid</h1>

<p align="center">
A non-evaluating alternative to f-strings and t-strings.
</p>

<p align="center">
  <a href="https://github.com/jg-rp/micro-liquid/blob/main/LICENSE.txt">
    <img src="https://img.shields.io/pypi/l/micro-liquid?style=flat-square" alt="License">
  </a>
  <a href="https://github.com/jg-rp/micro-liquid/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/jg-rp/micro-liquid/tests.yaml?branch=main&label=tests&style=flat-square" alt="Tests">
  </a>
  <br>
  <a href="https://pypi.org/project/micro-liquid">
    <img src="https://img.shields.io/pypi/v/micro-liquid.svg?style=flat-square" alt="PyPi - Version">
  </a>
  <a href="https://anaconda.org/conda-forge/micro-liquid">
    <img src="https://img.shields.io/conda/vn/conda-forge/micro-liquid?style=flat-square" alt="conda-forge">
  </a>
  <br>
  <a href="https://pypi.org/project/micro-liquid">
    <img src="https://img.shields.io/pypi/pyversions/micro-liquid.svg?style=flat-square" alt="Python versions">
  </a>
</p>

---

## Table of Contents

- [Install](#install)
- [Example](#example)
- [About](#about)
- [What's included?](#whats-included)
- [What's not included?](#whats-not-included)
- [Undefined variables](#undefined-variables)
- [Serializing objects](#serializing-objects)
- [License](#license)

## Install

```console
pip install micro-liquid
```

Or from [conda-forge](https://anaconda.org/conda-forge/micro-liquid):

```console
conda install conda-forge::micro-liquid
```

## Example

```python
from micro_liquid import Template

template = Template("Hello, {{ you }}!")
print(template.render({"you": "World"}))  # Hello, World!
```

## About

Micro Liquid implements minimal, Liquid-like templating. You can think of it as a non-evaluating alternative to [f-strings](https://peps.python.org/pep-0498/) or [t-strings](https://peps.python.org/pep-0750/), where templates are data and not necessarily string literals inside Python source files.

Full-featured Liquid ([Python Liquid](https://github.com/jg-rp/liquid) or [Shopify/liquid](https://github.com/Shopify/liquid), for example) caters for situations where end users manage their own templates. In this scenario, it's reasonable to expect some amount of application/display logic to be embedded within template text. In other scenarios we'd very much want to keep application logic out of template text.

With that in mind, Micro Liquid offers a greatly reduced feature set, implemented in a single Python file, so you can copy and paste it and hack on it if needed.

Here, developers are expected to fully prepare data passed to `Template.render()` instead of manipulating and inspecting it within template markup.

## What's included?

Micro Liquid supports variable substitution:

```liquid
Hello, {{ some_variable }}. How are you?
```

Conditional expressions:

```liquid
{% if some_variable %}
  more markup
{% elsif another_variable %}
  alternative markup
{% else %}
  default markup
{% endif %}
```

And looping:

```liquid
{% for x in y %}
  more markup with {{ x }}
{% else %}
  default markup (y was empty or not iterable)
{% endfor %}
```

Micro Liquid expressions can use logical `and`, `or` and `not`, and group terms with parentheses.

```liquid
{% if not some_variable and another_variable %}
  some markup with {{ some_variable }} and {{ another_variable }}.
{% endif %}
```

Short circuit evaluation works too, with _last value_ semantics. Here you might use a string literal.

```
Hello, {{ user.name or "guest" }}!
```

Control whitespace before and after markup delimiters with `-` and `~`. `~` will remove newlines but retain space and tab characters. `-` strips all whitespace.

```liquid
<ul>
{% for x in y ~%}
  <li>{{ x }}</li>
{% endfor -%}
</ul>
```

## What's not included?

If you need any of these features, and many more, try [Python Liquid](https://github.com/jg-rp/liquid) or [Python Liquid2](https://github.com/jg-rp/python-liquid2) instead.

- Assignment, captures and snippets (`include` and `render`). Or any tag other than `if` and `for`.
- Relational operators like `==` and `<`.
- Membership operators like `contains` and `in`.
- Filters.
- Literal Booleans, integers, floats or null/nil/None.
- `{% break %}` and `{% continue %}` tags.
- Nested variables.
- `forloop` helper variables.
- `for` tag arguments like `limit` and `reversed`.

## Other notable behavior

- We use Python truthiness, not Liquid or Ruby truthiness.
- Any `Iterable` can be looped over with the `{% for %}` tag. Non-iterable objects are silently ignored.
- Looping over dictionaries (or any Mapping) iterates key/value pairs.

## Undefined variables

When a template variable or property can't be resolved, an instance of the _undefined type_ is used instead. That is, an instance of `micro_liquid.Undefined` or a subclass of it.

The default _undefined type_ renders nothing when output, evaluates to `False` when tested for truthiness and is an empty iterable when looped over. You can pass an alternative _undefined type_ as the `undefined` keyword argument to the `Template` constructor to change this behavior.

```python
from micro_liquid import StrictUndefined
from micro_liquid import Template

t = Template("{{ foo.nosuchthing }}", undefined=StrictUndefined)

print(t.render({"foo": {}}))
# micro_liquid.UndefinedVariableError: 'foo.nosuchthing' is undefined
#   -> '{{ foo.nosuchthing }}':1:3
#   |
# 1 | {{ foo.nosuchthing }}
#   |    ^^^ 'foo.nosuchthing' is undefined
```

Or implement your own.

```python
from typing import Iterator
from micro_liquid import Template
from micro_liquid import Undefined


class MyUndefined(Undefined):
    def __str__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False

    def __iter__(self) -> Iterator[object]:
        yield from ()


t = Template("{{ foo.nosuchthing }}", undefined=MyUndefined)

print(t.render({"foo": {}}))  # <MISSING>
```

### Serializing objects

By default, when outputting an object with `{{` and `}}`, lists, dictionaries and tuples are rendered in JSON format. For all other objects we render the result of `str(obj)`.

You can change this behavior by passing a callable to the `Template` constructor or `render` function as the `serializer` keyword argument. The callable should accept an object and return its string representation suitable for output.

This example shows how one might define a serializer that can dump data classes with `json.dumps`.

```python
import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass

from micro_liquid import Template


@dataclass
class SomeData:
    foo: str
    bar: int


def json_default(obj: object) -> object:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def my_serializer(obj: object) -> str:
    return (
        json.dumps(obj, default=json_default)
        if isinstance(obj, (list, dict, tuple))
        else str(obj)
    )


template = Template("{{ some_object }}", serializer=my_serializer)
data = {"some_object": [SomeData("hello", 42)]}

print(template.render(data))  # [{"foo": "hello", "bar": 42}]
```

TODO: Walk syntax tree

## License

`micro-liquid` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
