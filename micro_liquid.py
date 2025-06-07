"""Minimal non-evaluating, Liquid-like text templating."""

from __future__ import annotations

import os
import re
from collections import deque
from contextlib import contextmanager
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
from typing import TypeAlias


class Template:
    def __init__(self, source: str):
        try:
            self.nodes = Parser(Scanner(source).tokens).parse()
        except TemplateSyntaxError as err:
            err.source = source
            raise

    def render(self, data: Mapping[str, object]) -> str:
        scope = _Scope(data)
        buffer: list[str] = []
        for node in self.nodes:
            if isinstance(node, str):
                buffer.append(node)
            else:
                node.render(scope, buffer)
        return "".join(buffer)


class TemplateSyntaxError(Exception):
    """An exception raised during template parsing due to unexpected template syntax."""

    def __init__(self, *args: object, token: _Token):
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


class _Token(NamedTuple):
    kind: str
    value: str
    start: int


_StateFn: TypeAlias = Callable[[], Optional["_StateFn"]]

_RE_TRIVIA = re.compile(r"[ \n\r\t]+")
_RE_TAG_NAME = re.compile(r"(?:end)?(?:if|for|elif|elsif|else)")
_RE_WORD = re.compile(r"[\u0080-\uFFFFa-zA-Z_][\u0080-\uFFFFa-zA-Z0-9_-]*")
_RE_MARKUP_START = re.compile(r"\{\{|\{%")
_RE_MARKUP_END = re.compile(r"[\}%]\}?")
_RE_PUNCTUATION = re.compile(r"[\[\]\.\(\)]")
_RE_INT = re.compile(r"-?\d+")

_TOKEN_MAP: dict[str, str] = {
    "[": "TOKEN_L_BRACKET",
    "]": "TOKEN_R_BRACKET",
    ".": "TOKEN_DOT",
    "(": "TOKEN_L_PAREN",
    ")": "TOKEN_R_PAREN",
    "and": "TOKEN_AND",
    "or": "TOKEN_OR",
    "not": "TOKEN_NOT",
    "in": "TOKEN_IN",
}

_ESCAPES = frozenset(["b", "f", "n", "r", "t", "u", "/", "\\"])


class Scanner:
    def __init__(self, source: str):
        self.source = source
        self.tokens: list[_Token] = []
        self.start = 0
        self.pos = 0

        state: _StateFn | None = self.lex_markup
        while state is not None:
            state = state()

    def emit(self, kind: str, value: str) -> None:
        self.tokens.append(_Token(kind, value, self.start))
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
        if ch in ("-", "~"):
            self.pos += 1
            self.emit("TOKEN_WC", ch)
            return True
        return False

    def lex_markup(self) -> _StateFn | None:
        value = self.scan(_RE_MARKUP_START)

        if value == r"{{":
            self.emit("TOKEN_OUT_START", value)
            self.accept_whitespace_control()
            self.skip(_RE_TRIVIA)
            return self.lex_expression

        if value == r"{%":
            self.emit("TOKEN_TAG_START", value)
            self.accept_whitespace_control()
            self.skip(_RE_TRIVIA)
            return self.lex_tag

        return self.lex_other

    def lex_expression(self) -> _StateFn | None:
        while True:
            self.skip(_RE_TRIVIA)

            if value := self.scan(_RE_INT):
                self.emit("TOKEN_INT", value)
            elif value := self.scan(_RE_PUNCTUATION):
                self.emit(_TOKEN_MAP.get(value, "TOKEN_UNKNOWN"), value)
            elif value := self.scan(_RE_WORD):
                self.emit(_TOKEN_MAP.get(value, "TOKEN_WORD"), value)
            else:
                peeked = self.peek()
                if peeked == "'":
                    self.pos += 1
                    self.start = self.pos
                    self.scan_string("'", "TOKEN_SINGLE_QUOTE_STRING")
                elif peeked == '"':
                    self.pos += 1
                    self.start = self.pos
                    self.scan_string('"', "TOKEN_DOUBLE_QUOTE_STRING")
                else:
                    break

        self.accept_whitespace_control()
        value = self.scan(_RE_MARKUP_END)

        if value == r"}}":
            self.emit("TOKEN_OUT_END", value)
        elif value == r"%}":
            self.emit("TOKEN_TAG_END", value)
        elif value in (r"%", r"}"):
            raise TemplateSyntaxError(
                "incomplete markup detected",
                token=_Token("TOKEN_ERROR", value, self.start),
            )
        else:
            raise TemplateSyntaxError(
                f"unexpected {self.peek()!r}",
                token=_Token("TOKEN_ERROR", self.next(), self.start),
            )

        return self.lex_markup

    def lex_tag(self) -> _StateFn | None:
        if tag_name := self.scan(_RE_TAG_NAME):
            self.emit("TOKEN_TAG_NAME", tag_name)
            self.skip(_RE_TRIVIA)
            return self.lex_expression

        raise TemplateSyntaxError(
            "unknown, missing or malformed tag name",
            token=_Token("TOKEN_UNKNOWN", self.peek(), self.start),
        )

    def lex_other(self) -> _StateFn | None:
        if value := self.scan_until(_RE_MARKUP_START):
            self.emit("TOKEN_OTHER", value)
            return self.lex_markup

        self.pos = len(self.source)

        if self.pos > self.start:
            self.emit("TOKEN_OTHER", self.source[self.start])

        return None

    def scan_string(self, quote: str, kind: str) -> None:
        if self.peek() == quote:
            # Empty string
            self.pos += 1
            self.start = self.pos
            self.emit(kind, "")
            return

        while True:
            ch = self.next()

            if ch == "\\":
                peeked = self.peek()
                if peeked in _ESCAPES or peeked == quote:
                    self.pos += 1
                else:
                    raise TemplateSyntaxError(
                        "invalid escape sequence",
                        token=_Token("TOKEN_ERROR", peeked, self.pos + 1),
                    )

            if ch == quote:
                self.emit(kind, self.source[self.start : self.pos - 1])
                return

            if not ch:
                raise TemplateSyntaxError(
                    "unclosed string literal",
                    token=_Token("TOKEN_ERROR", quote, self.start),
                )


class Markup(Protocol):
    """The interface for a tag or output statement."""

    def render(self, data: _Scope, buffer: list[str]) -> None:
        """Render this node to _buffer_. with reference to variables in _data_."""
        ...


class Expression(Protocol):
    """The interface for a loop or logical expression."""

    def evaluate(self, data: _Scope) -> object:
        """Evaluate this expression with reference to variables in _data_."""
        ...


Node: TypeAlias = str | Markup


class Output:
    def __init__(self, expression: Expression):
        self.expression = expression

    def render(self, data: _Scope, buffer: list[str]) -> None:
        buffer.append(str(self.expression.evaluate(data)))


class IfTag:
    def __init__(
        self,
        blocks: list[tuple[Expression, list[Node]]],
        default: Optional[list[Node]],
    ):
        self.blocks = blocks
        self.default = default

    def render(self, data: _Scope, buffer: list[str]) -> None:
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

    def render(self, data: _Scope, buffer: list[str]) -> None:
        target = self.target.evaluate(data)

        if not isinstance(target, Iterable):
            return

        namespace: dict[str, object] = {}
        rendered = False

        with data.extend(namespace):
            for item in target:
                namespace[self.name] = item
                rendered = True

                for node in self.block:
                    if isinstance(node, str):
                        buffer.append(node)
                    else:
                        node.render(data, buffer)

        if not rendered and self.default:
            for node in self.default:
                if isinstance(node, str):
                    buffer.append(node)
                else:
                    node.render(data, buffer)


class _Missing:
    def __str__(self) -> str:
        return ""

    def __bool__(self) -> bool:
        return False


_MISSING = _Missing()


class _Scope(Mapping[str, object]):
    def __init__(self, *maps: Mapping[str, object]):
        self._maps = deque(maps)

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

    def get(self, key: str, default: object = _MISSING) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def push(self, namespace: Mapping[str, object]) -> None:
        self._maps.appendleft(namespace)

    def pop(self) -> Mapping[str, object]:
        return self._maps.popleft()

    @contextmanager
    def extend(self, namespace: Mapping[str, object]) -> Iterator[_Scope]:
        self.push(namespace)
        try:
            yield self
        finally:
            self.pop()


def _resolve(
    path: list[str | int],
    data: Mapping[str, object],
    *,
    default: object = _MISSING,
) -> object:
    it = iter(path)
    root = next(it, None)

    if not isinstance(root, str):
        return default

    obj: object = data.get(root, default)

    for segment in it:
        if isinstance(obj, Mapping):
            obj = obj.get(segment)
        elif isinstance(obj, Sequence) and isinstance(segment, int):
            obj = obj[segment]
        else:
            return default

    return obj


class _BooleanExpression:
    def __init__(self, expression: Expression):
        self.expression = expression

    def evaluate(self, data: _Scope) -> bool:
        return bool(self.expression.evaluate(data))


class _LogicalNotExpression:
    def __init__(self, expression: Expression):
        self.expression = expression

    def evaluate(self, data: _Scope) -> bool:
        return not bool(self.expression.evaluate(data))


class _BinaryExpression:
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right


class _LogicalAndExpression(_BinaryExpression):
    def evaluate(self, data: _Scope) -> object:
        left = self.left.evaluate(data)
        if left:
            return self.right.evaluate(data)
        return left


class _LogicalOrExpression(_BinaryExpression):
    def evaluate(self, data: _Scope) -> object:
        left = self.left.evaluate(data)
        if left:
            return left
        return self.right.evaluate(data)


class _Variable:
    def __init__(self, path: list[str | int]):
        self.path = path

    def evaluate(self, data: _Scope) -> object:
        return _resolve(self.path, data)


_TERMINATE_EXPRESSION: set[str] = {
    "TOKEN_WC",
    "TOKEN_OUT_END",
    "TOKEN_TAG_END",
    "TOKEN_OTHER",
    "TOKEN_EOF",
}

_TERMINATE_GROUPED_EXPRESSION: set[str] = {"TOKEN_EOF", "TOKEN_OTHER", "TOKEN_RPAREN"}

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

_PRECEDENCES: dict[str, int] = {
    "TOKEN_AND": _PRECEDENCE_LOGICAL_AND,
    "TOKEN_OR": _PRECEDENCE_LOGICAL_OR,
    "TOKEN_NOT": _PRECEDENCE_PREFIX,
}

_BINARY_OPERATORS: set[str] = {"TOKEN_AND", "TOKEN_OR"}


class Parser:
    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.pos = 0
        self.eof = _Token("TOKEN_EOF", "", -1)
        self.whitespace_carry: Optional[str] = None

    def current(self) -> _Token:
        try:
            return self.tokens[self.pos]
        except IndexError:
            return self.eof

    def next(self) -> _Token:
        try:
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        except IndexError:
            return self.eof

    def peek(self, offset: int = 1) -> _Token:
        try:
            return self.tokens[self.pos + offset]
        except IndexError:
            return self.eof

    def eat(self, kind: str, message: str | None = None) -> _Token:
        token = self.next()
        if token.kind != kind:
            raise TemplateSyntaxError(
                message or f"unexpected {token.value!r}", token=token
            )
        return token

    def eat_empty_tag(self, name: str) -> _Token:
        self.eat("TOKEN_TAG_START", f"expected tag {name!r}")
        self.skip_wc()
        name_token = self.eat("TOKEN_TAG_NAME", f"expected tag {name!r}")

        if name_token.value != name:
            raise TemplateSyntaxError(
                f"unexpected tag {name_token.value!r}", token=name_token
            )

        self.carry_wc()
        self.eat("TOKEN_TAG_END", f"expected tag {name!r}")
        return name_token

    def expect_expression(self) -> None:
        if self.current().kind in _TERMINATE_EXPRESSION:
            raise TemplateSyntaxError("missing expression", token=self.current())

    def peek_wc(self) -> Optional[str]:
        token = self.peek()
        if token.kind == "TOKEN_WC":
            return token.value
        return None

    def carry_wc(self) -> None:
        if self.current().kind == "TOKEN_WC":
            self.whitespace_carry = self.next().value
        else:
            self.whitespace_carry = None

    def skip_wc(self) -> None:
        if self.current().kind == "TOKEN_WC":
            self.pos += 1

    def tag(self, name: str) -> bool:
        token = self.peek()
        if token.kind == "TOKEN_WC":
            token = self.peek(2)
        return token.kind == "TOKEN_TAG_NAME" and token.value == name

    def peek_tag_name(self) -> str:
        token = self.current()
        if token.kind == "TOKEN_WC":
            token = self.peek()

        if token.kind == "TOKEN_TAG_NAME":
            return token.value
        raise TemplateSyntaxError("missing tag name", token=token)

    def trim(self, value: str, left_trim: str | None, right_trim: str | None) -> str:
        if left_trim == right_trim:
            if left_trim == "-":
                return value.strip()
            if left_trim == "~":
                return value.strip("\r\n")
            return value

        if left_trim == "-":
            value = value.lstrip()
        elif left_trim == "~":
            value = value.lstrip("\r\n")

        if right_trim == "-":
            value = value.rstrip()
        elif right_trim == "~":
            value = value.rstrip("\r\n")

        return value

    def parse(self, end: set[str] | None = None) -> list[Node]:
        nodes: list[Node] = []

        while True:
            token = self.next()
            kind, value, _ = token

            if self.current().kind == "TOKEN_WC":
                self.pos += 1

            if kind == "TOKEN_OTHER":
                nodes.append(self.trim(value, self.whitespace_carry, self.peek_wc()))
            elif kind == "TOKEN_OUT_START":
                nodes.append(self.parse_output())
            elif kind == "TOKEN_TAG_START":
                if end and self.peek_tag_name() in end:
                    self.pos -= 1
                    return nodes
                nodes.append(self.parse_tag())
            elif kind == "TOKEN_EOF":
                return nodes
            else:
                raise TemplateSyntaxError(f"unexpected {kind}", token=token)

    def parse_output(self) -> Markup:
        expr = self.parse_primary()
        self.carry_wc()
        self.eat("TOKEN_OUT_END")
        return Output(expr)

    def parse_tag(self) -> Markup:
        token = self.eat("TOKEN_TAG_NAME")

        if token.value == "if":
            blocks: list[tuple[Expression, list[Node]]] = []
            self.expect_expression()
            expr = _BooleanExpression(self.parse_primary())
            self.carry_wc()
            self.eat("TOKEN_TAG_END")
            block = self.parse(_END_IF_BLOCK)
            blocks.append((expr, block))

            while self.tag("elsif"):
                self.eat("TOKEN_TAG_START")
                self.skip_wc()
                self.eat("TOKEN_TAG_NAME")
                self.expect_expression()
                expr = _BooleanExpression(self.parse_primary())
                self.carry_wc()
                self.eat("TOKEN_TAG_END")
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
            self.eat("TOKEN_IN", "missing 'in'")
            self.expect_expression()
            target = self.parse_primary()
            self.carry_wc()
            self.eat("TOKEN_TAG_END")
            block = self.parse(_END_FOR_BLOCK)

            if self.tag("else"):
                self.eat_empty_tag("else")
                default = self.parse(_END_IF_BLOCK)
            else:
                default = None

            self.eat_empty_tag("endfor")
            return ForTag(identifier, target, block, default)

        raise TemplateSyntaxError(f"unexpected tag {token.value!r}", token=token)

    def parse_primary(self, precedence: int = _PRECEDENCE_LOWEST) -> Expression:
        token = self.current()
        left_kind = token.kind

        left: Expression

        if left_kind == "TOKEN_L_PAREN":
            left = self.parse_group()
        elif left_kind in ("TOKEN_WORD", "TOKEN_L_BRACKET"):
            left = self.parse_path()
        elif left_kind == "TOKEN_NOT":
            self.pos += 1
            left = _LogicalNotExpression(self.parse_primary(_PRECEDENCE_PREFIX))
        else:
            raise TemplateSyntaxError(f"unexpected {left_kind}", token=token)

        while True:
            kind = self.current().kind

            if kind == "TOKEN_UNKNOWN":
                raise TemplateSyntaxError(
                    f"unexpected {self.current().value!r}", token=self.current()
                )

            if (
                kind == "TOKEN_EOF"
                or _PRECEDENCES.get(kind, _PRECEDENCE_LOWEST) < precedence
                or kind not in _BINARY_OPERATORS
            ):
                break

            left = self.parse_infix_expression(left)

        return left

    def parse_infix_expression(self, left: Expression) -> Expression:
        token = self.next()
        kind = token.kind
        precedence = _PRECEDENCES.get(kind, _PRECEDENCE_LOWEST)
        right = self.parse_primary(precedence)

        if kind == "TOKEN_AND":
            return _LogicalAndExpression(left, right)

        if kind == "TOKEN_OR":
            return _LogicalOrExpression(left, right)

        raise TemplateSyntaxError(f"unexpected operator {kind}", token=token)

    def parse_group(self) -> Expression:
        self.eat("TOKEN_L_PAREN")
        expr = self.parse_primary()

        if self.current().kind not in _TERMINATE_GROUPED_EXPRESSION:
            expr = self.parse_infix_expression(expr)

        self.eat("TOKEN_R_PAREN")
        return expr

    def parse_identifier(self) -> str:
        token = self.eat("TOKEN_WORD")
        if self.current().kind in ("TOKEN_DOT", "TOKEN_L_BRACKET"):
            raise TemplateSyntaxError(
                "expected an identifier, found a path", token=token
            )
        return token.value

    def parse_path(self) -> Expression:
        segments: list[str | int] = []

        if self.current().kind == "TOKEN_WORD":
            segments.append(self.next().value)

        while True:
            kind = self.next().kind
            if kind == "TOKEN_L_BRACKET":
                segments.append(self.parse_bracketed_path_selector())
            elif kind == "TOKEN_DOT":
                segments.append(self.parse_shorthand_path_selector())
            else:
                self.pos -= 1
                return _Variable(segments)

    def parse_bracketed_path_selector(self) -> str | int:
        token = self.next()
        kind, value, _ = token
        segment: int | str

        if kind == "TOKEN_INT":
            segment = int(value)
        elif kind in ("TOKEN_DOUBLE_QUOTE_STRING", "TOKEN_SINGLE_QUOTE_STRING"):
            segment = value
        elif kind == "TOKEN_R_BRACKET":
            raise TemplateSyntaxError("empty bracketed segment", token=token)
        else:
            raise TemplateSyntaxError(f"unexpected {kind}", token=token)

        self.eat("TOKEN_R_BRACKET")
        return segment

    def parse_shorthand_path_selector(self) -> str | int:
        token = self.next()
        kind, value, _ = token

        if kind == "TOKEN_INT":
            return int(value)

        if kind in ("TOKEN_WORD", "TOKEN_AND", "TOKEN_OR", "TOKEN_NOT"):
            return value

        raise TemplateSyntaxError(f"unexpected {kind}", token=token)
