"""Minimal non-evaluating, Liquid-like text templating.

MIT License

Copyright (c) 2025-present James Prior <jamesgr.prior@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from contextlib import contextmanager
from enum import IntEnum
from enum import auto
from itertools import chain
from re import Pattern
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Type
from typing import TypeAlias

__all__ = (
    "render",
    "Template",
    "TemplateError",
    "TemplateSyntaxError",
    "UndefinedVariableError",
    "StrictUndefined",
    "Undefined",
)


def render(
    source: str,
    data: Mapping[str, object],
    *,
    serializer: Callable[[object], str] | None = None,
    undefined: Type[Undefined] | None = None,
) -> str:
    """Render template _source_ with variables from _data_.

    Use `template = Template(source)` then `template.render(data)` if you need
    to render the same template multiple times with different data.
    """
    return Template(source, serializer=serializer, undefined=undefined).render(data)


class Template:
    """A compiled template, ready to be rendered."""

    def __init__(
        self,
        source: str,
        *,
        serializer: Callable[[object], str] | None = None,
        undefined: Type[Undefined] | None = None,
    ):
        self.source = source
        self.serializer = serializer or serialize
        self.undefined = undefined or Undefined
        try:
            self.nodes = Parser(Scanner(source).tokens).parse()
        except TemplateError as err:
            err.source = source
            raise

    def render(self, data: Mapping[str, object]) -> str:
        """Render this template with variables from _data_."""
        scope = Scope(data, serializer=self.serializer, undefined=self.undefined)
        buffer: list[str] = []
        for node in self.nodes:
            if isinstance(node, str):
                buffer.append(node)
            else:
                try:
                    node.render(scope, buffer)
                except TemplateError as err:
                    err.source = self.source
                    raise
        return "".join(buffer)


class TemplateError(Exception):
    """Base class for all template exceptions."""

    def __init__(self, *args: object, token: Token):
        super().__init__(*args)
        self.token = token
        self.source: str | None = None

    def __str__(self) -> str:
        if not self.token or self.token.start < 0 or not self.source:
            return super().__str__()

        _kind, value, index = self.token
        context = self.error_context(self.source, index)

        if not context:
            return super().__str__()

        line, col, current = context
        position = f"{current!r}:{line}:{col}"
        pad = " " * len(str(line))
        pointer = (" " * col) + ("^" * len(value))

        return os.linesep.join(
            [
                self.args[0],
                f"{pad} -> {position}",
                f"{pad} |",
                f"{line} | {current}",
                f"{pad} | {pointer} {self.args[0]}",
            ]
        )

    def error_context(self, text: str, index: int) -> tuple[int, int, str] | None:
        """Return the line number, column number and current line of text."""
        lines = text.splitlines(keepends=True)
        cumulative_length = 0
        target_line_index = -1

        for i, line in enumerate(lines):
            cumulative_length += len(line)
            if index < cumulative_length:
                target_line_index = i
                break

        if target_line_index == -1:
            return None

        line_number = target_line_index + 1  # 1-based
        column_number = index - (cumulative_length - len(lines[target_line_index]))
        current_line = lines[target_line_index].rstrip()
        return (line_number, column_number, current_line)


class TemplateSyntaxError(TemplateError):
    """An exception raised during template parsing due to unexpected template syntax."""


class UndefinedVariableError(TemplateError):
    """An exception raised by the strict undefined type."""


class TokenKind(IntEnum):
    EOF = auto()
    WC_NONE = auto()
    WC_HYPHEN = auto()
    WC_TILDE = auto()
    OTHER = auto()
    OUT_START = auto()
    OUT_END = auto()
    TAG_START = auto()
    TAG_END = auto()
    TAG_NAME = auto()
    L_BRACKET = auto()
    R_BRACKET = auto()
    DOT = auto()
    L_PAREN = auto()
    R_PAREN = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    INT = auto()
    SINGLE_QUOTE_STRING = auto()
    DOUBLE_QUOTE_STRING = auto()
    WORD = auto()
    UNKNOWN = auto()
    ERROR = auto()


class Token(NamedTuple):
    kind: TokenKind
    value: str
    start: int


_StateFn: TypeAlias = Callable[[], Optional["_StateFn"]]

_RE_TRIVIA = re.compile(r"[ \n\r\t]+")
_RE_TAG_NAME = re.compile(r"(?:end)?(?:if|for|elif|elsif|else)")
_RE_WORD = re.compile(r"[\u0080-\uFFFFa-zA-Z_][\u0080-\uFFFFa-zA-Z0-9_-]*")
_RE_MARKUP_START = re.compile(r"\{\{|\{%")
_RE_MARKUP_END = re.compile(r"[\}%]\}?")
_RE_PUNCTUATION = re.compile(r"[\[\]\.\(\)'\"]")
_RE_INT = re.compile(r"-?\d+")

_TOKEN_MAP: dict[str, TokenKind] = {
    "[": TokenKind.L_BRACKET,
    "]": TokenKind.R_BRACKET,
    ".": TokenKind.DOT,
    "(": TokenKind.L_PAREN,
    ")": TokenKind.R_PAREN,
    "and": TokenKind.AND,
    "or": TokenKind.OR,
    "not": TokenKind.NOT,
    "in": TokenKind.IN,
}

_ESCAPES = frozenset(["b", "f", "n", "r", "t", "u", "/", "\\"])


class Scanner:
    def __init__(self, source: str):
        self.source = source
        self.tokens: list[Token] = []
        self.start = 0
        self.pos = 0

        state: _StateFn | None = self.lex_markup
        while state is not None:
            state = state()

    def emit(self, kind: TokenKind, value: str) -> None:
        self.tokens.append(Token(kind, value, self.start))
        self.start = self.pos

    def next(self) -> str:
        try:
            ch = self.source[self.pos]
            self.pos += 1
            return ch
        except IndexError:
            return ""

    def peek(self) -> str:
        try:
            return self.source[self.pos]
        except IndexError:
            return ""

    def scan(self, pattern: Pattern[str]) -> str | None:
        match = pattern.match(self.source, self.pos)
        if match:
            self.pos += match.end() - match.start()
            return match[0]
        return None

    def scan_until(self, pattern: Pattern[str]) -> str | None:
        match = pattern.search(self.source, self.pos)
        if match:
            self.pos = match.start()
            return self.source[self.start : match.start()]
        return None

    def skip(self, pattern: Pattern[str]) -> bool:
        match = pattern.match(self.source, self.pos)
        if match:
            self.pos += match.end() - match.start()
            self.start = self.pos
            return True
        return False

    def accept_whitespace_control(self) -> bool:
        ch = self.peek()
        if ch == "-":
            self.pos += 1
            self.emit(TokenKind.WC_HYPHEN, ch)
            return True
        if ch == "~":
            self.pos += 1
            self.emit(TokenKind.WC_TILDE, ch)
            return True
        return False

    def lex_markup(self) -> _StateFn | None:
        value = self.scan(_RE_MARKUP_START)

        if value == r"{{":
            self.emit(TokenKind.OUT_START, value)
            self.accept_whitespace_control()
            self.skip(_RE_TRIVIA)
            return self.lex_expression

        if value == r"{%":
            self.emit(TokenKind.TAG_START, value)
            self.accept_whitespace_control()
            self.skip(_RE_TRIVIA)
            return self.lex_tag

        return self.lex_other

    def lex_expression(self) -> _StateFn | None:
        while True:
            self.skip(_RE_TRIVIA)

            if value := self.scan(_RE_INT):
                self.emit(TokenKind.INT, value)
            elif value := self.scan(_RE_PUNCTUATION):
                if value == "'":
                    self.start = self.pos
                    self.scan_string("'", TokenKind.SINGLE_QUOTE_STRING)
                elif value == '"':
                    self.start = self.pos
                    self.scan_string('"', TokenKind.DOUBLE_QUOTE_STRING)
                else:
                    self.emit(_TOKEN_MAP.get(value, TokenKind.UNKNOWN), value)
            elif value := self.scan(_RE_WORD):
                self.emit(_TOKEN_MAP.get(value, TokenKind.WORD), value)
            else:
                break

        self.accept_whitespace_control()
        value = self.scan(_RE_MARKUP_END)

        if value == r"}}":
            self.emit(TokenKind.OUT_END, value)
        elif value == r"%}":
            self.emit(TokenKind.TAG_END, value)
        elif value in (r"%", r"}"):
            raise TemplateSyntaxError(
                "incomplete markup detected",
                token=Token(TokenKind.ERROR, value, self.start),
            )
        else:
            raise TemplateSyntaxError(
                f"unexpected {self.peek()!r}",
                token=Token(TokenKind.ERROR, self.next(), self.start),
            )

        return self.lex_markup

    def lex_tag(self) -> _StateFn | None:
        if tag_name := self.scan(_RE_TAG_NAME):
            self.emit(TokenKind.TAG_NAME, tag_name)
            self.skip(_RE_TRIVIA)
            return self.lex_expression

        raise TemplateSyntaxError(
            "unknown, missing or malformed tag name",
            token=Token(TokenKind.UNKNOWN, self.peek(), self.start),
        )

    def lex_other(self) -> _StateFn | None:
        if value := self.scan_until(_RE_MARKUP_START):
            self.emit(TokenKind.OTHER, value)
            return self.lex_markup

        self.pos = len(self.source)

        if self.pos > self.start:
            self.emit(TokenKind.OTHER, self.source[self.start :])

        return None

    def scan_string(self, quote: str, kind: TokenKind) -> None:
        if self.peek() == quote:
            # Empty string
            self.pos += 1
            self.start = self.pos
            self.emit(kind, "")
            return

        needs_unescaping = False

        while True:
            ch = self.next()

            if ch == "\\":
                peeked = self.peek()
                if peeked in _ESCAPES or peeked == quote:
                    needs_unescaping = True
                    self.pos += 1
                else:
                    raise TemplateSyntaxError(
                        "invalid escape sequence",
                        token=Token(TokenKind.ERROR, ch, self.pos),
                    )

            if ch == quote:
                if needs_unescaping:
                    try:
                        self.emit(
                            kind,
                            _unescape(self.source[self.start : self.pos - 1], quote),
                        )
                    except json.JSONDecodeError as err:
                        raise TemplateSyntaxError(
                            "invalid escape sequence",
                            token=Token(TokenKind.ERROR, ch, self.pos - 1),
                        ) from err
                else:
                    self.emit(kind, self.source[self.start : self.pos - 1])
                return

            if not ch:
                raise TemplateSyntaxError(
                    "unclosed string literal",
                    token=Token(TokenKind.ERROR, quote, self.start),
                )


def _unescape(s: str, quote: str) -> str:
    if quote == "'":
        s = s.replace('"', '\\"').replace("\\'", "'")
    unescaped: str = json.loads(f'"{s}"')
    return unescaped


class Markup(Protocol):
    """The interface for a tag or output statement."""

    def render(self, data: Scope, buffer: list[str]) -> None:
        """Render this node to _buffer_ with reference to variables in _data_."""
        ...  # pragma: no cov


class Expression(Protocol):
    """The interface for a loop or logical expression."""

    def evaluate(self, data: Scope) -> object:
        """Evaluate this expression with reference to variables in _data_."""
        ...  # pragma: no cov


Node: TypeAlias = str | Markup


def serialize(obj: object) -> str:
    return json.dumps(obj) if isinstance(obj, (list, dict, tuple)) else str(obj)


class Output:
    def __init__(self, expression: Expression):
        self.expression = expression

    def render(self, data: Scope, buffer: list[str]) -> None:
        value = self.expression.evaluate(data)
        buffer.append(data.serialize(value))


class IfTag:
    def __init__(
        self,
        blocks: list[tuple[Expression, list[Node]]],
        default: Optional[list[Node]],
    ):
        self.blocks = blocks
        self.default = default

    def render(self, data: Scope, buffer: list[str]) -> None:
        for expression, block in self.blocks:
            if expression.evaluate(data):
                for node in block:
                    if isinstance(node, str):
                        buffer.append(node)
                    else:
                        node.render(data, buffer)
                return

        if self.default:
            for node in self.default:
                if isinstance(node, str):
                    buffer.append(node)
                else:
                    node.render(data, buffer)


class ForTag:
    def __init__(
        self,
        name: str,
        target: Expression,
        block: list[Node],
        default: Optional[list[Node]],
    ):
        self.name = name
        self.target = target
        self.block = block
        self.default = default

    def render(self, data: Scope, buffer: list[str]) -> None:
        target = self.target.evaluate(data)

        if not isinstance(target, Iterable):
            if self.default:
                self.render_block(data, buffer, self.default)
            return

        if isinstance(target, Mapping):
            target = target.items()

        namespace: dict[str, object] = {}
        rendered = False

        with data.extend(namespace):
            for item in target:
                namespace[self.name] = item
                rendered = True
                self.render_block(data, buffer, self.block)

        if not rendered and self.default:
            self.render_block(data, buffer, self.default)

    def render_block(self, data: Scope, buffer: list[str], nodes: list[Node]) -> None:
        for node in nodes:
            if isinstance(node, str):
                buffer.append(node)
            else:
                node.render(data, buffer)


class Undefined:
    """The object used when a template variable can not be resolved."""

    def __init__(self, name: str, token: Token):
        self.name = name
        self.token = token

    def __str__(self) -> str:
        return ""

    def __bool__(self) -> bool:
        return False

    def __iter__(self) -> Iterable[object]:
        yield from ()


class StrictUndefined(Undefined):
    def __str__(self) -> str:
        raise UndefinedVariableError(f"{self.name!r} is undefined", token=self.token)

    def __bool__(self) -> bool:
        return False

    def __iter__(self) -> Iterable[object]:
        raise UndefinedVariableError(f"{self.name!r} is undefined", token=self.token)


_UNDEFINED = object()


class Scope(Mapping[str, object]):
    def __init__(
        self,
        *maps: Mapping[str, object],
        serializer: Callable[[object], str] = json.dumps,
        undefined: Type[Undefined] = Undefined,
    ):
        self._maps = deque(maps)
        self.serialize = serializer
        self.undefined = undefined

    def __getitem__(self, key: str) -> object:
        for mapping in self._maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return chain(*self._maps)

    def __len__(self) -> int:
        return sum(len(_map) for _map in self._maps)

    def get(self, key: str, default: object = _UNDEFINED) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def push(self, namespace: Mapping[str, object]) -> None:
        self._maps.appendleft(namespace)

    def pop(self) -> Mapping[str, object]:
        return self._maps.popleft()

    @contextmanager
    def extend(self, namespace: Mapping[str, object]) -> Iterator[Scope]:
        self.push(namespace)
        try:
            yield self
        finally:
            self.pop()


class BooleanExpression:
    def __init__(self, expression: Expression):
        self.expression = expression

    def evaluate(self, data: Scope) -> bool:
        return bool(self.expression.evaluate(data))


class LogicalNotExpression:
    def __init__(self, expression: Expression):
        self.expression = expression

    def evaluate(self, data: Scope) -> bool:
        return not bool(self.expression.evaluate(data))


class BinaryExpression:
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right


class LogicalAndExpression(BinaryExpression):
    def evaluate(self, data: Scope) -> object:
        left = self.left.evaluate(data)
        if left:
            return self.right.evaluate(data)
        return left


class LogicalOrExpression(BinaryExpression):
    def evaluate(self, data: Scope) -> object:
        left = self.left.evaluate(data)
        if left:
            return left
        return self.right.evaluate(data)


class StringLiteral:
    def __init__(self, value: str):
        self.value = value

    def evaluate(self, _data: Scope) -> object:
        return self.value


class Variable:
    def __init__(self, token: Token, path: list[str | int]):
        self.token = token
        self.path = path

    def evaluate(self, data: Scope) -> object:
        it = iter(self.path)
        root = next(it, None)

        if not isinstance(root, str):
            return data.undefined(str(root), self.token)

        obj = data.get(root, _UNDEFINED)

        if obj == _UNDEFINED:
            return data.undefined(root, self.token)

        for i, segment in enumerate(it):
            if isinstance(obj, Mapping):
                obj = obj.get(segment, _UNDEFINED)
            elif isinstance(obj, Sequence) and isinstance(segment, int):
                try:
                    obj = obj[segment]
                except IndexError:
                    obj = _UNDEFINED
            else:
                obj = _UNDEFINED

            if obj == _UNDEFINED:
                return data.undefined(self.path_to_str(self.path[: i + 2]), self.token)

        return obj

    def path_to_str(self, path: list[str | int]) -> str:
        it = iter(path)
        buf: list[str] = [str(next(it, ""))]
        for segment in it:
            if isinstance(segment, str):
                if _RE_WORD.fullmatch(segment):
                    buf.append(f".{segment}")
                else:
                    buf.append(f"[{segment!r}]")
            else:
                buf.append(f"[{segment}]")
        return "".join(buf)


_WHITESPACE_CONTROL = 1 << TokenKind.WC_HYPHEN | 1 << TokenKind.WC_TILDE
_STRING = 1 << TokenKind.SINGLE_QUOTE_STRING | 1 << TokenKind.DOUBLE_QUOTE_STRING
_PATH_START = 1 << TokenKind.WORD | 1 << TokenKind.L_BRACKET
_PATH_PUNCTUATION = 1 << TokenKind.DOT | 1 << TokenKind.L_BRACKET
_BINARY_OPERATORS = 1 << TokenKind.AND | 1 << TokenKind.OR

_WORD = (
    1 << TokenKind.WORD | 1 << TokenKind.AND | 1 << TokenKind.OR | 1 << TokenKind.NOT
)

_TERMINATE_EXPRESSION = (
    1 << TokenKind.WC_HYPHEN
    | 1 << TokenKind.WC_TILDE
    | 1 << TokenKind.OUT_END
    | 1 << TokenKind.TAG_END
    | 1 << TokenKind.OTHER
    | 1 << TokenKind.EOF
)


_END_IF_BLOCK: set[str] = {
    "else",
    "elif",
    "elsif",
    "endif",
}

_END_FOR_BLOCK: set[str] = {
    "else",
    "endfor",
}

_PRECEDENCE_LOWEST = 1
_PRECEDENCE_LOGICAL_OR = 2
_PRECEDENCE_LOGICAL_AND = 3
_PRECEDENCE_PREFIX = 4

_PRECEDENCES: dict[TokenKind, int] = {
    TokenKind.AND: _PRECEDENCE_LOGICAL_AND,
    TokenKind.OR: _PRECEDENCE_LOGICAL_OR,
    TokenKind.NOT: _PRECEDENCE_PREFIX,
}


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.eof = Token(TokenKind.EOF, "", -1)
        self.whitespace_carry: TokenKind = TokenKind.WC_NONE

    def current(self) -> Token:
        try:
            return self.tokens[self.pos]
        except IndexError:
            return self.eof

    def next(self) -> Token:
        try:
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        except IndexError:
            return self.eof

    def peek(self, offset: int = 1) -> Token:
        try:
            return self.tokens[self.pos + offset]
        except IndexError:
            return self.eof

    def eat(self, kind: TokenKind, message: str | None = None) -> Token:
        token = self.next()
        if token.kind != kind:
            raise TemplateSyntaxError(
                message or f"unexpected {token.value!r}", token=token
            )
        return token

    def eat_empty_tag(self, name: str) -> Token:
        self.eat(TokenKind.TAG_START, f"expected tag {name!r}")
        self.skip_wc()
        name_token = self.eat(TokenKind.TAG_NAME, f"expected tag {name!r}")

        if name_token.value != name:
            raise TemplateSyntaxError(
                f"unexpected tag {name_token.value!r}", token=name_token
            )

        self.carry_wc()
        self.eat(TokenKind.TAG_END, f"expected tag {name!r}")
        return name_token

    def member(self, kind: TokenKind, mask: int) -> bool:
        return (mask & (1 << kind)) != 0

    def expect_expression(self) -> None:
        if self.member(self.current().kind, _TERMINATE_EXPRESSION):
            raise TemplateSyntaxError("missing expression", token=self.current())

    def peek_wc(self) -> TokenKind:
        token = self.peek()
        if self.member(token.kind, _WHITESPACE_CONTROL):
            return token.kind
        return TokenKind.WC_NONE

    def carry_wc(self) -> None:
        if self.member(self.current().kind, _WHITESPACE_CONTROL):
            self.whitespace_carry = self.next().kind
        else:
            self.whitespace_carry = TokenKind.WC_NONE

    def skip_wc(self) -> None:
        if self.member(self.current().kind, _WHITESPACE_CONTROL):
            self.pos += 1

    def tag(self, name: str) -> bool:
        token = self.peek()
        if self.member(token.kind, _WHITESPACE_CONTROL):
            token = self.peek(2)
        return token.kind == TokenKind.TAG_NAME and token.value == name

    def peek_tag_name(self) -> str:
        token = self.current()
        if self.member(token.kind, _WHITESPACE_CONTROL):
            token = self.peek()

        if token.kind == TokenKind.TAG_NAME:
            return token.value
        raise TemplateSyntaxError("missing tag name", token=token)

    def trim(self, value: str, left_trim: TokenKind, right_trim: TokenKind) -> str:
        if left_trim == right_trim:
            if left_trim == TokenKind.WC_HYPHEN:
                return value.strip()
            if left_trim == TokenKind.WC_TILDE:
                return value.strip("\r\n")
            return value

        if left_trim == TokenKind.WC_HYPHEN:
            value = value.lstrip()
        elif left_trim == TokenKind.WC_TILDE:
            value = value.lstrip("\r\n")

        if right_trim == TokenKind.WC_HYPHEN:
            value = value.rstrip()
        elif right_trim == TokenKind.WC_TILDE:
            value = value.rstrip("\r\n")

        return value

    def parse(self, end: set[str] | None = None) -> list[Node]:
        nodes: list[Node] = []

        while True:
            token = self.next()
            kind, value, _ = token

            if kind == TokenKind.OTHER:
                nodes.append(self.trim(value, self.whitespace_carry, self.peek_wc()))
            elif kind == TokenKind.OUT_START:
                if self.member(self.current().kind, _WHITESPACE_CONTROL):
                    self.pos += 1
                nodes.append(self.parse_output())
            elif kind == TokenKind.TAG_START:
                if end and self.peek_tag_name() in end:
                    self.pos -= 1
                    return nodes

                if self.member(self.current().kind, _WHITESPACE_CONTROL):
                    self.pos += 1
                nodes.append(self.parse_tag())
            elif kind == TokenKind.EOF:
                return nodes
            else:
                raise TemplateSyntaxError(f"unexpected {kind}", token=token)

    def parse_output(self) -> Markup:
        expr = self.parse_primary()
        self.carry_wc()
        self.eat(TokenKind.OUT_END)
        return Output(expr)

    def parse_tag(self) -> Markup:
        token = self.eat(TokenKind.TAG_NAME)

        if token.value == "if":
            blocks: list[tuple[Expression, list[Node]]] = []
            self.expect_expression()
            expr = BooleanExpression(self.parse_primary())
            self.carry_wc()
            self.eat(TokenKind.TAG_END)
            block = self.parse(_END_IF_BLOCK)
            blocks.append((expr, block))

            while self.tag("elsif") or self.tag("elif"):
                self.eat(TokenKind.TAG_START)
                self.skip_wc()
                self.eat(TokenKind.TAG_NAME)
                self.expect_expression()
                expr = BooleanExpression(self.parse_primary())
                self.carry_wc()
                self.eat(TokenKind.TAG_END)
                block = self.parse(_END_IF_BLOCK)
                blocks.append((expr, block))

            if self.tag("else"):
                self.eat_empty_tag("else")
                default = self.parse(_END_IF_BLOCK)
            else:
                default = None

            self.eat_empty_tag("endif")
            return IfTag(blocks, default)

        if token.value == "for":
            self.expect_expression()
            identifier = self.parse_identifier()
            self.eat(TokenKind.IN, "missing 'in'")
            self.expect_expression()
            target = self.parse_primary()
            self.carry_wc()
            self.eat(TokenKind.TAG_END)
            block = self.parse(_END_FOR_BLOCK)

            if self.tag("else"):
                self.eat_empty_tag("else")
                default = self.parse(_END_FOR_BLOCK)
            else:
                default = None

            self.eat_empty_tag("endfor")
            return ForTag(identifier, target, block, default)

        raise TemplateSyntaxError(f"unexpected tag {token.value!r}", token=token)

    def parse_primary(self, precedence: int = _PRECEDENCE_LOWEST) -> Expression:
        token = self.current()
        left_kind = token.kind

        left: Expression

        if self.member(left_kind, _STRING):
            left = StringLiteral(self.next().value)
        elif left_kind == TokenKind.L_PAREN:
            left = self.parse_group()
        elif self.member(left_kind, _PATH_START):
            left = self.parse_path()
        elif left_kind == TokenKind.NOT:
            self.pos += 1
            left = LogicalNotExpression(self.parse_primary(_PRECEDENCE_PREFIX))
        else:
            raise TemplateSyntaxError(f"unexpected {left_kind}", token=token)

        while True:
            kind = self.current().kind

            if kind == TokenKind.UNKNOWN:
                raise TemplateSyntaxError(
                    f"unexpected {self.current().value!r}", token=self.current()
                )

            if (
                kind == TokenKind.EOF
                or _PRECEDENCES.get(kind, _PRECEDENCE_LOWEST) < precedence
                or not self.member(kind, _BINARY_OPERATORS)
            ):
                break

            left = self.parse_infix_expression(left)

        return left

    def parse_infix_expression(self, left: Expression) -> Expression:
        token = self.next()
        kind = token.kind
        precedence = _PRECEDENCES.get(kind, _PRECEDENCE_LOWEST)
        right = self.parse_primary(precedence)

        if kind == TokenKind.AND:
            return LogicalAndExpression(left, right)

        if kind == TokenKind.OR:
            return LogicalOrExpression(left, right)

        raise TemplateSyntaxError(f"unexpected operator {kind}", token=token)

    def parse_group(self) -> Expression:
        self.eat(TokenKind.L_PAREN)
        expr = self.parse_primary()
        self.eat(TokenKind.R_PAREN, "unbalanced parentheses")
        return expr

    def parse_identifier(self) -> str:
        token = self.eat(TokenKind.WORD)
        if self.member(self.current().kind, _PATH_PUNCTUATION):
            raise TemplateSyntaxError(
                "expected an identifier, found a path", token=token
            )
        return token.value

    def parse_path(self) -> Expression:
        segments: list[str | int] = []
        token = self.current()

        if self.current().kind == TokenKind.WORD:
            segments.append(self.next().value)

        while True:
            kind = self.next().kind
            if kind == TokenKind.L_BRACKET:
                segments.append(self.parse_bracketed_path_selector())
            elif kind == TokenKind.DOT:
                segments.append(self.parse_shorthand_path_selector())
            else:
                self.pos -= 1
                return Variable(token, segments)

    def parse_bracketed_path_selector(self) -> str | int:
        token = self.next()
        kind, value, _ = token
        segment: int | str

        if kind == TokenKind.INT:
            segment = int(value)
        elif self.member(kind, _STRING):
            segment = value
        elif kind == TokenKind.R_BRACKET:
            raise TemplateSyntaxError("empty bracketed segment", token=token)
        else:
            raise TemplateSyntaxError(f"unexpected {kind}", token=token)

        self.eat(TokenKind.R_BRACKET)
        return segment

    def parse_shorthand_path_selector(self) -> str | int:
        token = self.next()
        kind, value, _ = token

        if kind == TokenKind.INT:
            return int(value)

        if self.member(kind, _WORD):
            return value

        raise TemplateSyntaxError(f"unexpected {kind}", token=token)
