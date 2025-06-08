# micro-liquid

---

## Table of Contents

- [Install](#install)
- [Example](#example)
- [About](#about)
- [License](#license)

## Install

```console
pip install micro-liquid
```

## Example

```python
from micro_liquid import Template

template = Template("Hello, {{ you }}!")
print(template.render({"you": "World"}))  # Hello, World!
```

## About

Micro Liquid implements minimal, Liquid-like templating. You can think of it as a non-evaluating alternative to f-strings or t-strings, where templates are data and not always string literals inside Python source files.

Full-featured Liquid ([Python Liquid](https://github.com/jg-rp/liquid) or [Shopify/liquid](https://github.com/Shopify/liquid), for example) caters for situations where end users manage their own templates. In this scenario, it's reasonable to expect some amount of application/display logic to be embedded within template text. In other scenarios we'd very much want to keep application logic out of template text.

With that in mind, Micro Liquid offers a greatly reduced feature set, implemented in a single Python file, so you can copy and paste it and hack on it if needed.

Here, developers are expected to fully prepare data passed to `Template.render()` instead of manipulating and inspecting it within template markup.

TODO:

- Just output, conditions and loops.
- **No** assignment, captures, `include`/`render` or any other tags.
- Just logical operators (`and`, `or`, `not`) with short circuit, last value semantics.
- **No** relational operators (like `==` or `<`) or membership operators (like `contains`).
- Includes whitespace control with `-` and `~`.
- **No** filters
- We use Python truthiness, not Liquid or Ruby truthiness.
- There are **no** literal strings, Booleans, integers, floats or null/nil/None.
- There are **no** `{% break %}` or `{% continue %}` tags.
- Nested variables are not allowed.
- Lists and dictionaries are output in JSON format.
- Any `Iterable` is can be looped over with the `{% for %}` tag. Non-iterable objects are silently ignored.
- Looping over dictionaries (or any Mapping) iterates key/value pairs.
- No `forloop` object

### What's included?

TODO:

### What's not included?

TODO:

### Undefined variables

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
        yield ""


t = Template("{{ foo.nosuchthing }}", undefined=MyUndefined)

print(t.render({"foo": {}}))  # <MISSING>
```

## License

`micro-liquid` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
