import calendar
import curses
import datetime as dt
import inspect
import json
import numpy as np
import os
import textwrap
from collections import defaultdict
from curses import panel
from enum import Enum
from functools import partial
from math import ceil, exp, factorial, log, pi, sqrt
from scipy.special import gamma as gammafun
from sqlalchemy import Boolean, Column, ForeignKey, Integer, Numeric, String, DateTime, create_engine
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship

PROGRAM = "Predictor"
VERSION = "0"

#
# Interface
#

def logfactorial(n):
    # Approximates log(n!)
    if n <= 10:
        return log(factorial(n))
    # https://math.stackexchange.com/a/152359
    return (
        n * log(n) -
        n +
        log(1/30 + n * (1 + 4*n*(1 + 2*n))) / 6 +
        log(pi) / 2
    )

class Distribution:
    def name(self):
        return self.__class__.__name__

    def cdf(self, x):
        pass

    def pdf(self, x):
        pass

    pmf = pdf

    def from_json(data):
        # TODO
        name = data["name"]
        if name == "normal":
            pars = [data["parameters"][p] for p in ["mu", "sigma"]]
            return Normal(*pars)
        if name == "infinity":
            return PointMass(float("inf"))
        if name == "exponential":
            return Exponential(data["parameters"]["lambda"])
        raise ValueError("NOT IMPLEMENTED YET", data)

def nat(n):
    return isinstance(n, int) and n >= 0

#
# Discrete distributions
#

class Discrete(Distribution):
    pass

class PointMass(Discrete):
    def __init__(self, point):
        self._point = point

    def point(self):
        return self._point

    def pmf(self, x):
        if x == self.point():
            return 1
        return 0

    def mode(self):
        return self._point

    def sd(self):
        return 0

    def xmin(self):
        return self._point

    def xmax(self):
        return self._point

def poisson_pmf(λ):
    return lambda n: 0 if not nat(n) else exp(n * log(λ) - logfactorial(n) - λ)

def poisson_cdf(λ):
    pass

def poisson_mode(λ):
    return λ # wrong, but OK?

#
# Continuous distributions
#

class Continuous(Distribution):
    pass

class Normal(Continuous):
    def __init__(self, μ, σ):
        self._μ = μ
        self._σ = σ

    def parameters(self):
        return [
            Parameter("μ", unitless=False),
            Parameter("σ", minimum=0, unitless=False)
        ]

    def pdf(self, x):
        μ = self._μ
        σ = self._σ
        return 1 / (sqrt(2 * pi) * σ) * exp(- ((x - μ) / σ) ** 2 / 2)

    def xmin(self):
        return -float("inf")

    def xmax(self):
        return float("inf")

    def mode(self):
        return self._μ

    def sd(self):
        return self._σ

class Exponential(Continuous):
    def __init__(self, λ):
        self._λ = λ

    def pdf(self, x):
        return 0 if x < 0 else self._λ * exp(-self._λ*x)

    def xmin(self):
        return 0

    def xmax(self):
        return float("inf")

    def mode(self):
        return 0

    def sd(self):
        return 1/self._λ

def ensure_not_str(c):
    if isinstance(c, str):
        return ord(c)
    return c

class Key:
    def __init__(self, c, aliases=[], name=None):
        if name is None:
            name = c
        self._name = name
        c = ensure_not_str(c)
        self._primary = c
        if isinstance(c, str):
            c = ord(c)
        self._aliases = {ensure_not_str(a) for a in aliases}
        self._aliases.add(c)

    def __str__(self):
        return chr(self._primary)

    def isalpha(self):
        return str(self).isalpha()

    def isprint(self):
        return str(self).isprintable()

    def isspace(self):
        return str(self).isspace()

    def __eq__(self, other):
        if isinstance(other, str):
            return self == Key(other)
        return not self._aliases.isdisjoint(other._aliases)

    def __repr__(self):
        return f"<Key {self._name} ({self._primary})>"

Key.ESC   = Key(27)
Key.QUIT  = Key("q")
Key.CLOSE = Key("x")
Key.SPACE = Key(" ", name="<Space>")
Key.TAB   = Key("\t", name="<Tab>")
Key.BACK_TAB = Key(curses.KEY_BTAB, name="<Back Tab>")
Key.LEFT  = Key(curses.KEY_LEFT, ["h"], "Left")
Key.RIGHT = Key(curses.KEY_RIGHT, ["l"], "Right")
Key.UP    = Key(curses.KEY_UP, ["k"], "Up")
Key.DOWN  = Key(curses.KEY_DOWN, ["j"], "Down")
Key.BACKSPACE = Key(curses.KEY_BACKSPACE, ["\b"], "Backspace")
Key.ENTER = Key("\n", name="<Enter>")

class Widget:
    def handle_key(self, k):
        return False

    def min_width(self):
        raise NotImplementedError(self.__class__)

    def min_height(self):
        raise NotImplementedError(self.__class__)

    def dim(self, maxrow, maxcol):
        raise NotImplementedError(self.__class__)

    def draw(self, writer):
        pass

    def focus(self):
        self._focused = True

    def unfocus(self):
        self._focused = False

    def find(self, cls):
        return None

    def contents(self):
        return None

class Writer:
    def write(self, row, col, text):
        pass

    def dim(self):
        pass

    def set(self, attr):
        return AttributeSetter(self, attr)

class AttributeSetter:
    def __init__(self, writer, attr):
        self._writer = writer
        self._attr = attr
        self._saved = None

    def __enter__(self):
        self._saved = self._writer.get_attributes()
        self._writer.set_attributes(self._saved | self._attr)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        self._writer.set_attributes(self._saved)
        return True

class Crop(Writer):
    def __init__(self, wrapped, amounts):
        if isinstance(amounts, list):
            if len(amounts) not in [1, 2, 4]:
                raise ValueError # TODO
            if len(amounts) == 1:
                lcrop = rcrop = tcrop = bcrop = amounts[0]
            elif len(amounts) == 2:
                tcrop = bcrop = amounts[0]
                lcrop = rcrop = amounts[1]
            else:
                [tcrop, rcrop, bcrop, lcrop] = amounts
        else: # Just one value
            lcrop = rcrop = tcrop = bcrop = amounts
        self._lcrop = lcrop
        self._rcrop = rcrop
        self._tcrop = tcrop
        self._bcrop = bcrop
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def dim(self):
        r, c = self._wrapped.dim()
        r -= self._tcrop + self._bcrop
        c -= self._lcrop + self._rcrop
        return r, c

    def write(self, r, c, text):
        self._wrapped.write(r + self._tcrop, c + self._lcrop, text)

class Empty(Widget):
    def min_width(self):
        return 0

    def min_height(self):
        return 0

    def dim(self, maxrow, maxcol):
        return 0, 0

    def draw(self, writer):
        pass

class Decorator(Widget):
    def __init__(self, decorated=Empty()):
        self._decorated = decorated

    def focus(self):
        self._decorated.focus()

    def unfocus(self):
        self._decorated.unfocus()

    def handle_key(self, k):
        return self._decorated.handle_key(k)

    def contents(self):
        return self._decorated.contents()

    def find(self, cls):
        if isinstance(self._child, cls):
            return self._child
        return self._child.find(cls)

    def __getattr__(self, name):
        return getattr(self._decorated, name)

class Box(Decorator):
    def __init__(
        self, child=Empty(),
        tl="┌", tr="┐", bl = "└", br = "┘",
        side = "│", flat = "─"
    ):
        super().__init__(child)
        self._tl = tl; self._tr = tr
        self._bl = bl; self._br = br
        self._side = side; self._flat = flat

    def min_width(self):
        return 2 + self._decorated.min_width()

    def min_height(self):
        return 2 + self._decorated.min_height()

    def dim(self, maxrow, maxcol):
        r, c = self._decorated.dim(maxrow - 2, maxcol - 2)
        return r + 2, c + 2

    def draw(self, writer):
        maxrow, maxcol = writer.dim()
        h, w = self._decorated.dim(maxrow - 2, maxcol - 2)
        h += 2; w += 2

        # Corners
        writer.write(0, 0, self._tl); writer.write(0, w - 1, self._tr)
        writer.write(h - 1, 0, self._bl); writer.write(h - 1, w - 1, self._br)

        # Top/bottom
        for c in range(1, w - 1):
            writer.write(0, c, self._flat)
            writer.write(h - 1, c, self._flat)
        # Left/right
        for r in range(1, h - 1):
            writer.write(r, 0, self._side)
            writer.write(r, w - 1, self._side)

        rcrop = maxcol - w + 1
        bcrop = maxrow - h + 1
        self._decorated.draw(Crop(writer, [1, rcrop, bcrop, 1]))

class Shadow(Decorator):
    def __init__(self, child=Empty(), shade="░"):
        super().__init__(child)
        self._shade = shade

    def min_width(self):
        return 1 + self._decorated.min_width()

    def min_height(self):
        return 1 + self._decorated.min_height()

    def dim(self, maxrow, maxcol):
        r, c = self._decorated.dim(maxrow - 1, maxcol - 1)
        return r + 1, c + 1

    def draw(self, writer):
        # Setting the color is necessary in case we've changed the background.
        with writer.set(curses.color_pair(2)):
            maxrow, maxcol = writer.dim()
            h, w = self._decorated.dim(maxrow - 1, maxcol - 1)
            h += 1; w += 1
            for r in range(1, h):
                writer.write(r, w - 1, self._shade)
            for c in range(1, w):
                writer.write(h - 1, c, self._shade)
            writer.write(h - 1, 0, " ")
            writer.write(0, w - 1, " ")
        self._decorated.draw(Crop(writer, [0, 1, 1, 0]))

class Fill(Decorator):
    def __init__(self, child=Empty(), horizontal=True, vertical=True):
        super().__init__(child)
        self._horizontal = horizontal
        self._vertical = vertical

    def min_width(self):
        return self._decorated.min_width()

    def min_height(self):
        return self._decorated.min_height()

    def dim(self, maxrow, maxcol):
        if not self._horizontal or not self._vertical:
            r, c = self._decorated.dim(maxrow, maxcol)
        if self._vertical:
            r = maxrow
        if self._horizontal:
            c = maxcol
        return r, c

    def draw(self, writer):
        self._decorated.draw(writer)

class Text(Widget):
    def __init__(self, text, align=None, bold=False):
        self._align = align
        self._bold = bold
        self._text = text

    def contents(self):
        return self._text

    def min_width(self):
        return min([len(word) for word in self._text.split()], default = 0)

    def min_height(self):
        return 1

    def dim(self, maxrow, maxcol):
        wrapped = textwrap.wrap(self._text, maxcol)
        return len(wrapped), max([len(line) for line in wrapped], default = 0)

    def draw(self, writer):
        _, maxcol = writer.dim()
        wrapped = textwrap.wrap(self._text, maxcol)
        if self._align == "center":
            if self._bold:
                with writer.set(curses.A_BOLD):
                    for i, line in enumerate(wrapped):
                        line = line.center(maxcol)
                        writer.write(i, 0, line)
            else:
                for i, line in enumerate(wrapped):
                    line = line.center(maxcol)
                    writer.write(i, 0, line)
        else:
            if self._bold:
                with writer.set(curses.A_BOLD):
                    for i, line in enumerate(wrapped):
                        line = line.center(maxcol)
                        writer.write(i, 0, line)
            for i, line in enumerate(wrapped):
                writer.write(i, 0, line)

class Titled(Decorator):
    def __init__(self, title, child=Empty()):
        self._title = Text(title, align="center", bold=True)
        super().__init__(child)

    def min_width(self):
        return max(self._decorated.min_width(), self._title.min_width())

    def min_height(self):
        return max(self._decorated.min_height(), self._title.min_height())

    def dim(self, maxrow, maxcol):
        titlerow, titlecol = self._title.dim(maxrow, maxcol)
        bodyrow, bodycol = self._decorated.dim(maxrow - titlerow, maxcol)
        return titlerow + bodyrow, max(titlecol, bodycol)

    def draw(self, writer):
        maxrow, maxcol = writer.dim()
        titlerow, titlecol = self._title.dim(maxrow, maxcol)
        self._title.draw(writer)
        self._decorated.draw(Crop(writer, [titlerow, 0, 0, 0]))

class Checkbox(Widget):
    def __init__(self, label, checked=False, focused=False):
        self._label = label
        self._checked = checked
        self._focused = focused

    def handle_key(self, k):
        if k == Key.SPACE:
            self.toggle()
            return True
        return False

    def toggle(self):
        self._checked = not self._checked

    def checked(self):
        return self._checked

    contents = checked

    def min_width(self):
        return 2 + min([len(word) for word in self._label.split()], default = 0)

    def min_height(self):
        return 1

    def dim(self, maxrow, maxcol):
        wrapped = textwrap.wrap(self._label, maxcol - 2)
        return len(wrapped), 2 + max([len(line) for line in wrapped], default = 0)

    def draw(self, writer):
        maxrow, maxcol = writer.dim()
        if self._focused:
            with writer.set(curses.A_REVERSE):
                writer.write(0, 0, "☑" if self._checked else "☐")
        else:
            writer.write(0, 0, "☑" if self._checked else "☐")
        wrapped = textwrap.wrap(self._label, width=maxcol-2)
        for i, line in enumerate(wrapped):
            writer.write(i, 2, line)

class Clock(Widget):
    WIDTH  = 8
    HEIGHT = 3

    def __init__(self, time=None):
        if time is None:
            time = dt.datetime.now()
        self._time = time
        self._cursor = 0 # 0 = hour, 1 = minute, 2 = AM/PM
        self._focused = False

    def min_height(self):
        return Clock.HEIGHT

    def min_width(self):
        return Clock.WIDTH

    def dim(self, maxrow, maxcol):
        return Clock.HEIGHT, Clock.WIDTH

    def handle_key(self, k):
        if k == Key.RIGHT:
            self.right()
        elif k == Key.LEFT:
            self.left()
        elif k == Key.UP:
            self.up()
        elif k == Key.DOWN:
            self.down()
        else:
            return False
        return True

    def right(self):
        if self._cursor != 2:
            self._cursor += 1

    def left(self):
        if self._cursor != 0:
            self._cursor -= 1

    def up(self):
        if self._cursor == 0:
            h = self._time.hour
            self._time = self._time.replace(hour=(h+1)%24)
        elif self._cursor == 1:
            m = self._time.minute
            self._time = self._time.replace(minute=(m+1)%60)
            if m == 59:
                self._time = self._time.replace(hour=(self._time.hour+1)%24)
        else:
            h = self._time.hour
            self._time = self._time.replace(hour=(h+12)%24)

    def down(self):
        if self._cursor == 0:
            h = self._time.hour
            self._time = self._time.replace(hour=(h-1)%24)
        elif self._cursor == 1:
            m = self._time.minute
            self._time = self._time.replace(minute=(m-1)%60)
            if m == 0:
                self._time = self._time.replace(hour=(self._time.hour-1)%24)
        else:
            h = self._time.hour
            self._time = self._time.replace(hour=(h-12)%24)

    def draw(self, writer):
        time = self._time
        _, cols = writer.dim()
        hour = time.hour % 12
        if hour == 0:
            hour = 12
        hour = f"{hour:2}"
        minute = f"{time.minute:02}"
        am_pm = f"{time.strftime('%p')}"
        if self._focused and self._cursor == 0:
            writer.write(0, 1, "▲")
            with writer.set(curses.A_REVERSE):
                writer.write(1, 0, hour)
            writer.write(2, 1, "▼")
        else:
            writer.write(1, 0, hour)
        writer.write(1, 2, ":")
        if self._focused and self._cursor == 1:
            writer.write(0, 4, "▲")
            with writer.set(curses.A_REVERSE):
                writer.write(1, 3, minute)
            writer.write(2, 4, "▼")
        else:
            writer.write(1, 3, minute)
        if self._focused and self._cursor == 2:
            writer.write(0, 6, "▲")
            with writer.set(curses.A_REVERSE):
                writer.write(1, 6, am_pm)
            writer.write(2, 6, "▼")
        else:
            writer.write(1, 6, am_pm)

    def contents(self):
        return self._time

class CategoricalInput(Widget):
    def __init__(self, labels):
        self._probs = [(label, 0) for label in labels]
        self._graph = BarGraph({label: p for label, p in self._probs})
        self._current_label = 0

    def min_width(self):
        return max(len(label) for label, _ in self._probs) + 4 # ?

    def min_height(self):
        return len(self._probs) + 2 # ? probably not ?

    def dim(self, maxrow, maxcol):
        return maxrow, maxcol # TODO

    def handle_key(self, k):
        if k == Key.UP:
            self.up()
        elif k == Key.DOWN:
            self.down()
        elif k == Key.RIGHT:
            self.right()
        elif k == Key.LEFT:
            self.left()
        else:
            return False
        return True

    def up(self):
        if self._current_label == 0:
            return
        self._current_label -= 1

    def down(self):
        if self._current_label == len(self._probs) - 1:
            return
        self._current_label += 1

    # TODO: adjust other probabilities automatically
    def left(self):
        i = self._current_label
        l, p = self._probs[i]
        if p > 0:
            self._probs[i] = l, p - 1

    def right(self):
        i = self._current_label
        l, p = self._probs[i]
        if p < 100:
            self._probs[i] = l, p + 1

    def draw(self, writer):
        self._graph.update({label: prob for label, prob in self._probs})
        self._graph.draw(writer)

    def contents(self):
        return {label: prob for label, prob in self._probs}

class DistributionInput(Widget):
    def __init__(
        self, discrete=False, is_date=False, has_infinity=False,
        range_lo=-float("inf"), range_hi=float("inf"), labels=None
    ):
        self._components = []
        self._discrete = discrete
        self._is_date = is_date
        self._has_infinity = has_infinity
        self._range_lo = range_lo
        self._range_hi = range_hi
        self._labels = labels

        # This is just for demonstration
        self._mu = 0
        self._sigma = 1

    def min_width(self):
        return 10 # idk

    def min_height(self):
        return 10 # idk

    def dim(self, maxrow, maxcol):
        return maxrow, maxcol # idk

    def handle_key(self, k):
        if k == Key.UP:
            self._mu += 1
            return True
        if k == Key.DOWN:
            self._mu -= 1
            return True
        if k == Key.RIGHT:
            if self._sigma < 1:
                self._sigma /= 0.9
            else:
                self._sigma += 1
            return True
        if k == Key.LEFT:
            if self._sigma <= 1:
                self._sigma *= 0.9
            else:
                self._sigma -= 1
            return True
        return False

    def draw(self, writer):
        # Again, demonstration
        d = Normal(self._mu, self._sigma)
        g = LineGraph([(1, d)], self._has_infinity, self._range_lo, self._range_hi)
        g.draw(writer)

class Calendar(Widget):
    WIDTH  = 20
    HEIGHT =  8

    def __init__(self, date=None):
        if date is None:
            date = dt.date.today()
        self._date = date
        self._cursor_on_month = False
        self._focused = False

    def handle_key(self, k):
        if k == Key.UP:
            self.up()
        elif k == Key.DOWN:
            self.down()
        elif k == Key.LEFT:
            self.left()
        elif k == Key.RIGHT:
            self.right()
        else:
            return False
        return True

    def up(self):
        # Move to the last week if it's still the same month.
        # Otherwise put the cursor on the month.
        # NOTE: It might make sense to move to the last week in any case, but
        # then how would we change the month?
        if self._cursor_on_month:
            return
        date = self._date
        if date.day - 7 < 1:
            self._cursor_on_month = True
        else:
            self._date = dt.date(date.year, date.month, date.day - 7)

    def down(self):
        # Move to the next week if it's still the same month.
        # NOTE: See note for method `up`.
        date = self._date
        if self._cursor_on_month:
            self._date = dt.date(date.year, date.month, 1)
            self._cursor_on_month = False
        else:
            _, maxday = calendar.monthrange(date.year, date.month)
            self._date = dt.date(date.year, date.month, min(date.day + 7, maxday))

    def right(self):
        # Move to the next date or month, as appropriate.
        date = self._date
        if self._cursor_on_month:
            if date.month == 12:
                self._date = dt.date(date.year + 1, 1, 1)
            else:
                self._date = dt.date(date.year, date.month + 1, 1)
        else:
            _, maxday = calendar.monthrange(date.year, date.month)
            if date.day == maxday:
                if date.month == 12:
                    self._date = dt.date(date.year + 1, 1, 1)
                else:
                    self._date = dt.date(date.year, date.month + 1, 1)
            else:
                self._date = dt.date(date.year, date.month, date.day + 1)

    def left(self):
        # Move to the previous date or month, as appropriate.
        date = self._date
        if self._cursor_on_month:
            if date.month == 1:
                self._date = dt.date(date.year - 1, 12, 1)
            else:
                self._date = dt.date(date.year, date.month - 1, 1)
        else:
            if date.day == 1:
                if date.month == 1:
                    self._date = dt.date(date.year - 1, 12, 31)
                else:
                    _, maxday = calendar.monthrange(date.year, date.month - 1)
                    self._date = dt.date(date.year, date.month - 1, maxday)
            else:
                self._date = dt.date(date.year, date.month, date.day - 1)

    def get_date(self):
        return self._date

    contents = get_date

    def min_width(self):
        return Calendar.WIDTH

    def min_height(self):
        return Calendar.HEIGHT

    def dim(self, maxrow, maxcol):
        return Calendar.HEIGHT, Calendar.WIDTH

    def draw(self, writer):
        date = self._date
        if date.year != dt.date.today().year:
            topline = date.strftime("%B %Y")
        else:
            topline = date.strftime("%B")

        lpad = (Calendar.WIDTH - len(topline)) // 2
        if self._focused and self._cursor_on_month:
            with writer.set(curses.A_REVERSE):
                writer.write(0, lpad, topline)
        else:
            writer.write(0, lpad, topline)
        if self._focused and self._cursor_on_month:
            writer.write(0, 0, "◀")
            writer.write(0, Calendar.WIDTH - 1, "▶")
        writer.write(1, 0, "S  M  T  W  T  F  S ")

        first, ndays = calendar.monthrange(date.year, date.month)
        # Make Sunday the first day of the week
        first = (first + 1) % 7 # Sunday = 0

        for day in range(1, ndays + 1):
            row = (first + day - 1) // 7
            col = ((first + day - 1) % 7)
            col = 3 * col
            if day == date.day and not self._cursor_on_month:
                with writer.set(curses.A_REVERSE):
                    writer.write(row + 2, col, f"{day:2}")
            else:
                writer.write(row + 2, col, f"{day:2}")

class Button(Widget):
    def __init__(self, label, pressed=False):
        self._label = label
        self._pressed = pressed

    def toggle(self):
        self._pressed = not self._pressed

    def pressed(self):
        return self._pressed

    contents = pressed

    def min_width(self):
        return len(self._label) + 4

    def min_height(self):
        return 1

    def dim(self, maxrow, maxcol):
        return 1, len(self._label) + 4

    def draw(self, writer):
        label = self._label
        if self._focused:
            writer.write(0, 0, "[")
            writer.write(0, len(label) + 3, "]")
            with writer.set(curses.A_REVERSE):
                writer.write(0, 2, label)
        else:
            writer.write(0, 0, f"[ {label} ]")

class TextLineInput(Widget):
    def __init__(self, placeholder=""):
        self._placeholder = placeholder
        self._text = []
        self._focused = False
        # TODO: text editing
        # TODO: escape should unfocus

    def contents(self):
        return "".join(self._text)

    def clear(self):
        self._text = []

    def handle_key(self, k):
        if self._focused:
            if k == Key.BACKSPACE:
                if self._text:
                    self._text.pop()
                return True
            if k == Key.BACK_TAB: # apparently is printable
                return False
            if k.isprint():
                self._text.append(str(k))
                return True
        return False

    def min_height(self):
        return 1

    def min_width(self):
        return max(2, len(self._placeholder))

    def dim(self, maxrow, maxcol):
        return 1, maxcol

    def draw(self, writer):
        _, maxcol = writer.dim()
        if self._focused:
            text = self._text
        else:
            text = self._text or self._placeholder
        textlen = len(text)
        if textlen > maxcol - 1:
            text = text[-(maxcol-1):]
        text = "".join(text)
        if self._focused:
            writer.write(0, 0, text)
            writer.write(0, len(text), "█")
        else:
            with writer.set(curses.A_DIM):
                writer.write(0, 0, text)

class Tags(Widget):
    def __init__(self, placeholder=""):
        self._input = TextLineInput(placeholder)
        self._tags = []
        self._focused = False

    def focus(self):
        self._input.focus()
        self._focused = True

    def unfocus(self):
        self._input.unfocus()
        self._focused = False

    def draw(self, writer):
        rows, cols = writer.dim()
        self._input.draw(writer)
        spent, _ = self._input.dim(rows, cols)
        tags = self._tags
        for tag in reversed(tags):
            if rows - spent == 0:
                break
            writer.write(spent, 0, f"({tag})")
            spent += 1

    def contents(self):
        return self._tags

    def handle_key(self, k):
        if self._focused:
            if self._input.handle_key(k):
                return True
            if k == Key.ENTER:
                tag = self._input.contents()
                if tag:
                    self._input.clear()
                    self._tags.append(tag)
                    return True
        return False

    def min_height(self):
        return 2

    def min_width(self):
        return max(2, len(self._input._placeholder))

    def dim(self, maxrow, maxcol):
        return min(maxrow, len(self._tags) + 3), maxcol # TODO

class Info(Widget):
    def __init__(self):
        self._question = None
        self._predictions = []

    def populate(self, question, predictions):
        self._question = question
        self._predictions = predictions

    def min_height(self):
        return min(3, len(self._items))

    def min_width(self):
        return 1 # TODO: OK?

    def dim(self, maxrow, maxcol):
        return 2, maxcol # TODO: OK?

    def draw(self, writer):
        rows, cols = writer.dim()
        q = self._question
        cd = q.close_date
        closed = cd <= dt.datetime.today()

        # Print title at top
        title = textwrap.wrap(q.title, width=cols)
        for i, line in enumerate(title):
            with writer.set(curses.A_BOLD):
                writer.write(i, 0, line)
        pad = len(title)

        # Print close date
        if cd is None:
            with writer.set(curses.A_ITALIC):
                writer.write(pad, 0, "close date not set")
        else:
            verb = "closed" if closed else "closes"
            d = f"{cd.strftime('%b')} {cd.day}, {cd.year}"
            h = 12 if cd.hour % 12 == 0 else cd.hour % 12
            t = f"{h}:{cd.strftime('%M %p')}"
            with writer.set(curses.A_ITALIC):
                writer.write(pad, 0, f"{verb} on {d} at {t}")
        pad += 2

        # Print latest prediction
        predictions = self._predictions
        if not predictions:
            msg = "No predictions"
            if not closed:
                msg += " yet"
            msg = msg.center(cols)
            with writer.set(curses.A_DIM):
                writer.write(pad, 0, msg)
            return
        latest = predictions[0]
        distinfo = latest.json

        # Categorical distributions
        labels = q.labels
        if labels:
            # In this case, the prediction should be a list of
            # probability-label objects.
            pdict = dict()
            for lab in labels:
                pdict[lab] = 0
            for obj in latest.json:
                pdict[obj["label"]] = obj["p"]
            g = BarGraph(pdict)
        else:
            if isinstance(distinfo, list):
                # It's a combination of distributions.
                # In this case the format is [{"p": p, "distribution": dist}, ...],
                # where p is a probability and dist is a distribution object.
                dists = [(obj["p"], Distribution.from_json(obj["distribution"])) for obj in distinfo]
            else:
                # It's a single distribution.
                dists = [(1, Distribution.from_json(latest.json))]
            g = LineGraph(dists, q.has_infinity, q.range_lo, q.range_hi)
        g.draw(Crop(writer, [pad, 0, 0, 0]))

class ScrollList(Widget):
    class Item:
        def __init__(self, label, obj, hover, trigger):
            self._label = label
            self._object = obj
            self._hover = hover
            self._trigger = trigger

        def hover(self):
            if self._hover:
                self._hover()
                return True
            return False

        def trigger(self):
            if self._trigger:
                self._trigger()
                return True
            return False

    def __init__(self, selected=curses.A_BOLD):
        self._selected = selected
        self._items = []
        self._cursor_item = -1 # index into list of items
        self._cursor_row  = -1 # row of window where cursor is

    def sort(self, key, reverse=False):
        self._items.sort(key=lambda it: key(it._object), reverse=reverse)

    def get(self):
        if self._items:
            return self._items[self._cursor_item]._object

    def pop(self):
        if self._items:
            i = self._cursor_item
            if i == len(self._items) - 1:
                self._cursor_item -= 1
            it = self._items.pop(i)._object
            if self._cursor_item > -1:
                self._items[self._cursor_item].hover()
            return it
        return None

    def current_label(self):
        if self._items:
            return self._items[self._cursor_item]._label

    def add_item(self, label, obj, hover=None, trigger=None, prepend=False):
        if not self._items:
            self._cursor_item = 0
            self._cursor_row = 0
        it = ScrollList.Item(label, obj, hover, trigger)
        if prepend:
            self._items = [it] + self._items
        else:
            self._items.append(it)

    def min_height(self):
        return min(3, len(self._items))

    def min_width(self):
        return 1 # TODO: OK?

    def dim(self, maxrow, maxcol):
        rows = min(len(self._items), maxrow)
        cols = max(len(it._label) for it in self._items)
        return rows, cols

    def handle_key(self, k):
        if k == Key.DOWN:
            self.next()
        elif k == Key.UP:
            self.prev()
        elif k == Key("g"):
            self.first()
        elif k == Key("G"):
            self.last()
        elif k == Key.ENTER:
            return self._items[self._cursor_item].trigger()
        else:
            return False
        return True

    def next(self):
        nitems = len(self._items)
        if self._cursor_item == nitems - 1: # last item
            return
        self._cursor_item += 1
        self._cursor_row += 1 # may have to be reduced when drawn
        self._items[self._cursor_item].hover()

    def prev(self):
        if self._cursor_item <= 0: # first item or empty list
            return
        self._cursor_item -= 1
        if self._cursor_row > 0:
            self._cursor_row -= 1
        self._items[self._cursor_item].hover()

    def first(self):
        if self._items:
            self._cursor_item = 0
            self._cursor_row  = 0
            self._items[0].hover()

    def last(self):
        if self._items:
            self._cursor_item = len(self._items) - 1
            self._cursor_row  = self._cursor_item
            self._items[len(self._items) - 1].hover()

    def draw(self, writer):
        nitems = len(self._items)
        if nitems == 0:
            return

        rows, cols  = writer.dim()
        cursor_item = self._cursor_item
        cursor_row  = min(rows - 1, self._cursor_row)
        begin_row   = 0 # Row where the list starts, not counting arrows

        if nitems <= rows: # Everything fits on screen
            items = self._items
            begin_index = 0
        else:
            # If the list is too big for the screen,
            # we may need to draw arrows to indicate
            # there are more items above/below the screen
            # that aren't shown.
            narrows = "↑ more ↑".center(cols)
            sarrows = "↓ more ↓".center(cols)
            space_above = cursor_row
            space_below = rows - cursor_row - 1
            if cursor_item == 1 and cursor_row == 0:
                # Special case: We get pushed down, just as if we were going to
                # draw arrows, but there's only one list item above us, so we
                # will draw that instead.
                space_above = cursor_row = 1
                space_below -= 1
            elif cursor_item > cursor_row:
                # Need to draw arrows on top.
                # But we may have to check again if we draw arrows on bottom.
                if space_above > 0:
                    space_above -= 1
                else:
                    # We get pushed down.
                    cursor_row  += 1
                    space_below -= 1
                writer.write(0, 0, narrows)
                begin_row = 1

            if nitems - cursor_item - 1 == 1 and space_below == 0:
                space_above -= 1
                cursor_row  -= 1
                space_below  = 1
                if cursor_item == cursor_row + 1:
                    writer.write(0, 0, narrows)
                    space_above -= 1
                    begin_row = 1
            elif nitems - cursor_item - 1 > space_below:
                # Need arrows on bottom
                if space_below == 0: # We get pushed up.
                    space_above -= 1
                    cursor_row -= 1
                    # Now we have to check for top arrows again.
                    # This only changes things if we didn't have them already.
                    if cursor_item == cursor_row + 1:
                        writer.write(0, 0, narrows)
                        space_above -= 1
                        begin_row = 1
                else:
                    space_below -= 1
                writer.write(rows - 1, 0, sarrows)
            begin_index = cursor_item - space_above
            end_index = cursor_item + space_below + 1
            items = self._items[begin_index:end_index]
            self._cursor_row = cursor_row

        for i, item in enumerate(items):
            label = textwrap.shorten(item._label, cols, placeholder=" …")
            label = label.ljust(cols) # clears long titles while scrolling
            if i + begin_index == cursor_item:
                with writer.set(self._selected):
                    writer.write(i + begin_row, 0, label)
            else:
                writer.write(i + begin_row, 0, label)
        #writer.write(rows - 1, 0, f"index: {self._cursor_item}, below: {space_below}, currow: {self._cursor_row}")

# TODO:
# [ ] arrange things nicely
# [ ] allow fields to have an "empty" label that doesn't show
# [ ] don't mark things with a default value (e.g. checkboxes) as required
# [ ] make submission and cancellation possible
class Form(Widget):
    class Field:
        def __init__(self, widget, action, display, required):
            self._widget = widget
            self._action = action
            self._display = display
            self._required = required

        def read(self):
            return self._action()

        def __getattr__(self, name):
            return getattr(self._widget, name)

    def __init__(self, title=None):
        self._fields = {}
        self._focused = False
        self._focused_field = None
        self._title = title

    def focus(self):
        ff = self._focused_field
        if ff is not None:
            self._fields[ff].focus()

    def unfocus(self):
        ff = self._focused_field
        if ff is not None:
            self._fields[ff].unfocus()

    def focus_off(self):
        ff = self._focused_field
        if ff is not None:
            self._fields[ff].unfocus()
        self._focused_field = None
        return ff

    def focus_on(self, label):
        if label is None:
            return self.focus_off()
        ff = self._focused_field
        if ff is not None:
            self._fields[ff].unfocus()
        self._fields[label].focus()
        self._focused_field = label
        return ff

    def focus_first(self):
        return self.focus_on(next(iter(self._fields)))

    def focus_next(self):
        labels = list(self._fields.keys())
        old_ff = self.focus_off()
        if old_ff is None:
            return self.focus_on(labels[0])
        old_index = labels.index(old_ff)
        if old_index == len(labels) - 1:
            new_index = 0
        else:
            new_index = old_index + 1
        self.focus_on(labels[new_index])
        return old_ff

    def focus_prev(self):
        labels = list(self._fields.keys())
        old_ff = self.focus_off()
        if old_ff is None:
            return self.focus_on(labels[len(labels)-1])
        old_index = labels.index(old_ff)
        if old_index == 0:
            new_index = len(labels) - 1
        else:
            new_index = old_index - 1
        self.focus_on(labels[new_index])
        return old_ff

    def handle_key(self, k):
        ff = self._focused_field
        if ff is not None:
            handled = self._fields[ff].handle_key(k)
            if handled:
                return True
        if k == Key.TAB:
            self.focus_next()
            return True
        elif k == Key.BACK_TAB:
            self.focus_prev()
            return True
        elif k == Key.ESC and self._focused_field:
            self.focus_off()
            return True
        return False

    def add_field(self, label, widget, action, display=None, required=True):
        if display is None:
            display = label
        self._fields[label] = Form.Field(widget, action, display, required)

    def read(self):
        info = {}
        for label, field in self._fields.items():
            info[label] = field.read()
        return info

    def find(self, cls):
        for field in self._fields.values():
            if isinstance(field._widget, cls):
                return field._widget
            result = field.find(cls)
            if result is not None:
                return result
        return None

    def min_width(self): # TODO: revisit
        return max([f.min_width() for f in self._fields.values()], default=0)

    def min_height(self): # TODO: revisit
        return len(self._fields) + sum([f.min_height() for f in self._fields.values()])

    def dim(self, maxrow, maxcol): # TODO: revisit
        rows_left = maxrow
        width = 0
        for f in self._fields.values():
            rows_left -= 1 # for the label
            h, w = f.dim(rows_left, maxcol)
            width = max(w, width)
            rows_left -= h
        return maxrow - rows_left, width

    def draw(self, writer):
        maxrow, maxcol = writer.dim()
        currow = nextrow = 0
        curcol = 0
        ff = self._focused_field
        if self._title:
            with writer.set(curses.A_BOLD):
                writer.write(0, 0, self._title.center(maxcol))
            currow += 1
        for label, field in self._fields.items():
            display = "*" + field._display if field._required else field._display
            width_needed = max(len(display), field.min_width())
            if width_needed > maxcol - curcol:
                currow = nextrow
                curcol = 0
                height, width = field.dim(maxrow - currow - 1, maxcol) # -1 for label
            else:
                height, width = field.dim(maxrow - currow - 1, maxcol - curcol)
            if label == ff:
                writer.write(currow, curcol, display)
            else:
                with writer.set(curses.A_DIM):
                    writer.write(currow, curcol, display)
            if label == ff:
                field.draw(Crop(writer, [currow + 1, 0, 0, curcol]))
            else:
                with writer.set(curses.A_DIM):
                    field.draw(Crop(writer, [currow + 1, 0, 0, curcol]))
            curcol = curcol + width + 1
            nextrow = max(currow + height + 1, nextrow)

class Canvas(Decorator, Writer):
    def __init__(self, child=None, rows=None, cols=None, begin_row=0, begin_col=0):
        super().__init__(child)
        if not rows:
            rows = child.min_height() if child else curses.LINES
        if not cols:
            cols = child.min_width() if child else curses.COLS
        self._fast_quit_enabled = True
        self._window = curses.newwin(rows, cols, begin_row, begin_col)
        self._window.keypad(True) # Makes arrow keys work
        self._panel = panel.new_panel(self._window)
        self._attrs = curses.A_NORMAL

    def set_background(self, color):
        self._window.bkgd(" ", curses.color_pair(color))

    def get_key(self):
        return Key(self._window.getch())

    def hide(self):
        self._panel.hide()
        panel.update_panels()

    def show(self):
        self._panel.show()
        panel.update_panels()

    def top(self):
        self._panel.top()
        panel.update_panels()

    def draw(self):
        self._window.erase()
        self._decorated.draw(self)
        self._window.noutrefresh()
        curses.doupdate()

    def get_and_handle_key(self):
        self.draw()
        k = self.get_key()
        handled = self._decorated.handle_key(k)
        return k, handled

    def set_attributes(self, attrs):
        self._attrs = attrs

    def get_attributes(self):
        return self._attrs

    def dim(self):
        return self._window.getmaxyx()

    def write(self, row, col, text):
        attrs = self._attrs
        try:
            if len(text) == 1:
                self._window.addch(row, col, text, attrs)
            else:
                self._window.addstr(row, col, text, attrs)
        except curses.error:
            # This is necessary for writing on the right edge of the window.
            pass

class PrintWriter(Writer):
    def __init__(self, maxrow, maxcol):
        super().__init__()
        self._maxrow = maxrow
        self._maxcol = maxcol
        self._lines = [list(" " * maxcol) for _ in range(maxrow)]

    def write(self, row, col, text):
        if not 0 <= row < self._maxrow:
            raise ValueError # TODO
        if not 0 <= col < self._maxcol:
            raise ValueError # TODO
        lines = text.splitlines()
        for i, line in enumerate(lines):
            self.write_line(row, col + i, line)

    def dim(self):
        return self._maxrow, self._maxcol

    def write_line(self, row, col, text):
        chars = list(text)
        start = col
        end = col + len(chars)
        self._lines[row][start:end] = chars

    def __repr__(self):
        return "\n".join(["".join(line) for line in self._lines])

class Graph:
    def draw(self, writer):
        pass

class LineGraph(Graph):
    def __init__(self, dists, has_infinity=False, range_lo=None, range_hi=None):
        self._dists = dists
        self._has_infinity = has_infinity
        self._range_lo = range_lo
        self._range_hi = range_hi

    def draw(self, writer):
        rows, cols = writer.dim()
        dists = self._dists
        isolated_masses = [d.point() for _, d in dists if isinstance(d, PointMass)]
        has_infinity = any([x == float("inf") for x in isolated_masses])
        all_discrete = all([isinstance(d, Discrete) for _, d in dists])
        has_infinity = self._has_infinity
        mass_at_infinity = 0
        if has_infinity:
            for p, d in dists:
                if isinstance(d, PointMass) and d.point() == float("inf"):
                    mass_at_infinity += p

        # Here we figure out the graph dimensions.
        # ┐gggg
        # │gggg     The "g" area is height x width
        # ┘gggg
        #  ─┬──
        #   5
        # We need two rows for the lower axis.
        height = rows - 3 # TODO: why 3 not 2?
        width = (
            cols -
            1 - # axis
            1 - # pad
            4   # space for labels, e.g. "~0.5" or "0.25"
        )
        if has_infinity:
            width -= 2

        # First we need to find the interval we will graph.
        # We /never/ go outside the question's interval.
        hard_xmin = self._range_lo or -float("inf")
        hard_xmax = self._range_hi or  float("inf")
        # And we don't bother going beyond the prediction's interval.
        soft_xmin = min([d.xmin() for _, d in dists])
        soft_xmax = max([d.xmax() for _, d in dists])
        # To narrow things down further, we want to include
        # every distribution's mode and at least 3 SDs on either side.
        sd_xmin =  float("inf")
        sd_xmax = -float("inf")
        for _, d in dists:
            if isinstance(d, PointMass):
                continue
            m  = d.mode()
            sd = d.sd()
            sd_xmin = min(sd_xmin, m - 3 * sd)
            sd_xmax = max(sd_xmax, m + 3 * sd)
        soft_xmin = max(soft_xmin, sd_xmin)
        soft_xmax = min(soft_xmax, sd_xmax)
        # And if they conflict, the question wins.
        xmin = max(hard_xmin, soft_xmin)
        xmax = min(hard_xmax, soft_xmax)
        xs = np.linspace(xmin, xmax, width)
        if all_discrete:
            xs = [round(x) for x in xs]

        # Now we find out the range.
        if all_discrete:
            ys = [sum([p * d.pmf(x) for p, d in dists]) for x in xs]
            ymax = 1
        else:
            # We'll handle the continuous/discrete parts separately.
            def density(x):
                continuous = [(p, d) for p, d in dists if isinstance(d, Continuous)]
                return sum([p * d.pdf(x) for p, d in continuous])
            def mass(x):
                discrete = [(p, d) for p, d in dists if isinstance(d, Discrete)]
                return sum([p * d.pmf(x) for p, d in discrete])
            ys_continuous = [density(x) for x in xs]
            ys_discrete   = [mass(round(x)) for x in xs]
            ys_discrete.append(mass_at_infinity)
            ymax = ceil(10 * max(ys_continuous)) / 10 # round up
            ymax = max(ymax, *ys_discrete)
            if any([y >= 0.5 for y in ys_discrete]):
                ymax = max(1, ymax)
        ystep = ymax / height

        # Draw the y-axis
        writer.write(0, 5, "┐")
        for row in range(1, height):
            writer.write(row, 5, "│")
        writer.write(height, 5, "┘")

        # Draw the top y label
        # TODO: handle very large ymax
        ydigits = 2 if ymax < 10 else 1
        yerr = abs(round(ymax, ydigits) - ymax)
        if yerr < ystep / 10:
            maxlab = str(round(ymax, ydigits))
        else:
            maxlab = "~" + str(round(ymax, ydigits - 1))
        writer.write(0, 0, maxlab.rjust(4))

        # Draw the bottom y label
        # TODO: add digits after decimal if appropriate
        minlab = "0".ljust(ydigits + 1)
        writer.write(height, 0, minlab.rjust(4))

        # TODO: other y labels?  (need to decide before drawing axis)

        # Draw the x-axis
        # We need to decide what the labels will be first.
        for col in range(width):
            writer.write(height + 1, col + 6, "─")
        # TODO: fix
        if has_infinity:
            level = round(mass_at_infinity / ystep)
            writer.write(height + 1, width + 6, "╌")
            writer.write(height + 1, width + 6 + 1, "┬")
            writer.write(height + 2, width + 6 + 1, "∞")
            for i in range(level):
                writer.write(height - i, width + 6 + 1, "▓")
        # TODO: draw x labels

        # Draw the function
        if not all_discrete:
            # TODO: discrete parts
            ys = ys_continuous
        init_level = round(ys[0] / ystep)
        next_row = height - init_level
        for i, y in enumerate(ys, start=1):
            level = round(y / ystep)
            row_approx = height - round(level)
            if next_row < row_approx:
                writer.write(next_row, i + 5, "╮")
                next_row += 1
                while next_row < row_approx:
                    writer.write(next_row, i + 5, "│")
                    next_row += 1
                writer.write(next_row, i + 5, "╰")
            elif next_row > row_approx:
                writer.write(next_row, i + 5, "╯")
                next_row -= 1
                while next_row > row_approx:
                    writer.write(next_row, i + 5, "│")
                    next_row -= 1
                writer.write(next_row, i + 5, "╭")
            else:
                writer.write(next_row, i + 5, "─")

class BarGraph(Graph):
    def __init__(self, categories):
        self._cats = categories

    def update(self, categories):
        self._cats = categories

    def draw(self, writer):
        rows, cols = writer.dim()
        labels = list(self._cats.keys())
        credences = list(self._cats.values())

        # How wide will the labels be?
        max_label_width = cols // 3
        # Wrap lines
        line_groups = [textwrap.wrap(label, width=max_label_width) for label in labels]
        # Figure out how wide the widest label line will be
        label_width = max([len(line) for group in line_groups for line in group])
        # Right-align all the lines
        line_groups = [[line.rjust(label_width) for line in group] for group in line_groups]

        height = rows - 2 # take off 2 for the probability axis
        width = (
            cols -
            1 - # axis
            1 - # pad
            label_width # labels
        )

        # Print the probability axis
        writer.write(0, label_width + 1, "0%")
        writer.write(0, cols - 4, "100%")
        writer.write(1, label_width + 1, "├")
        writer.write(1, cols - 1, "┘")
        for i in range(label_width + 2, cols - 1):
            writer.write(1, i, "─")

        # Do we have enough rows to show all categories?
        # If we can't show them all, we need 1 more row for a warning.
        if len(line_groups[0]) + 1 > height:
            # TODO: can't show even the first label, so now what?
            pass
        else:
            height_used = len(line_groups[0])
            can_show = 1
            for group in line_groups[1:]:
                # Once again, 1 row may be taken up by a warning.
                if height_used + len(group) + 2 > height:
                    break
                can_show += 1
                height_used += len(group) + 1 # pad above
            current_row = 2
            for i in range(can_show):
                if i > 0:
                    writer.write(current_row, label_width + 1, "│")
                    current_row += 1 # pad
                writer.write(current_row, 0, line_groups[i][0])
                writer.write(current_row, label_width + 1, "┤")
                bar = round(credences[i] / 100 * width)
                for j in range(bar):
                    with writer.set(curses.A_REVERSE):
                        writer.write(current_row, label_width + 2 + j, " ")
                numstr = f"{credences[i]}%"
                if bar + len(numstr) <= width:
                    writer.write(current_row, label_width + 2 + bar, numstr)
                else:
                    with writer.set(curses.A_REVERSE):
                        writer.write(current_row, label_width + 2 + bar - len(numstr), numstr)
                current_row += 1
                for line in line_groups[i][1:]:
                    writer.write(current_row, 0, line)
                    writer.write(current_row, label_width + 1, "│")
                    current_row += 1

            if can_show < len(line_groups):
                # We're missing some
                missing = len(line_groups) - can_show
                writer.write(height_used + 2, label_width + 1, f"┆ ({missing} more not shown)")

class Menu:
    def __init__(self):
        self._main = Canvas(rows=1, cols=curses.COLS)
        self._main.set_background(1)
        self._submenus = dict()
        self._active_submenu = None
        # TODO: sort (or "arrange"), search, view, ...

    def focus_off(self):
        self._active_submenu = None

    def add_submenu(self, name, items):
        col = sum([len(n) for n in self._submenus]) + len(self._submenus)
        width = max([len(label) for label, _ in items])
        submenu = Canvas(
            Shadow(ScrollList(selected=curses.A_REVERSE)),
            rows=len(items)+1, cols=width+1, begin_row=1, begin_col=col
        )
        submenu.set_background(1)
        for label, trigger in items:
            submenu.add_item(label, None, trigger=trigger)
        self._submenus[name] = submenu

    def draw(self):
        col = 0
        for name in self._submenus:
            initial = name[0]
            rest = name[1:]
            if name == self._active_submenu:
                with self._main.set(curses.A_REVERSE):
                    with self._main.set(curses.A_UNDERLINE):
                        self._main.write(0, col, initial)
                    self._main.write(0, col + 1, rest)
            else:
                with self._main.set(curses.A_UNDERLINE):
                    self._main.write(0, col, initial)
                self._main.write(0, col + 1, rest)
            col += len(name) + 1
        progname = PROGRAM + " v" + VERSION
        with self._main.set(curses.A_ITALIC):
            self._main.write(0, curses.COLS - len(progname), progname)
        active = self._active_submenu
        if active:
            self._submenus[active].draw()
            self._submenus[active]._window.noutrefresh()
        self._main._window.noutrefresh()
        curses.doupdate()

    def get_and_handle_key(self):
        self.draw()
        k = self._main.get_key()
        handled = self.handle_key(k)
        return k, handled

    def handle_key(self, key):
        if self._active_submenu is None:
            # Is the user activating the menu?
            for name, submenu in self._submenus.items():
                if name[0].lower() == key:
                    self._active_submenu = name
                    return True
        elif self._active_submenu:
            active = self._active_submenu
            return self._submenus[active].handle_key(key)
        return False

#
# Database
#

Base = declarative_base()

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)
    kind = Column(String(), nullable=False)
    title = Column(String(), nullable=False)
    description = Column(String())
    open_date = Column(DateTime, nullable=False)
    close_date = Column(DateTime)
    expiry_date = Column(DateTime)

    predictions = relationship(
        "Prediction", backref="question",
        cascade="all, delete",
        passive_deletes=True
    )

    __mapper__args = {
        "polymorphic_identity": "question",
        "polymorphic_on": kind
    }

    def __init__(self, **kwargs):
        kwargs["open_date"] = dt.datetime.now()
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<Question {self.id}: {self.title}>"

class CategoricalQuestion(Question):
    __tablename__ = "categorical_questions"

    id = Column(Integer, ForeignKey(Question.id), primary_key=True)
    extendable = Column(Boolean(), nullable=False)
    categories = relationship(
        "Category", backref="question",
        cascade="all, delete",
        passive_deletes=True
    )

    __mapper_args__ = {
        "polymorphic_identity": "categorical"
    }

class BinaryQuestion(Question):
    __tablename__ = "binary_questions"

    id = Column(Integer, ForeignKey(Question.id), primary_key=True)

    __mapper_args__ = {
        "polymorphic_identity": "binary"
    }

    def __init__(self, **kwargs):
        kwargs["kind"] = BinaryQuestion.__mapper_args__["polymorphic_identity"]
        super().__init__(**kwargs)

class NumericQuestion(Question):
    __tablename__ = "numeric_questions"

    id = Column(Integer, ForeignKey(Question.id), primary_key=True)
    discrete = Column(Boolean(), nullable=False)
    right_mass = Column(Boolean(), nullable=False)
    left_mass = Column(Boolean(), nullable=False)
    range_lo = Column(Numeric()) # -infinity if null
    range_hi = Column(Numeric()) # +infinity if null

    __mapper_args__ = {
        "polymorphic_identity": "numeric"
    }

#class DateQuestion(Question):
#    __tablename__ = "date_questions"
#
#    id = Column(Integer, ForeignKey(Question.id), primary_key=True)
#    right_mass = Column(Boolean(), nullable=False)
#    left_mass = Column(Boolean(), nullable=False)
#    range_lo = Column(Numeric()) # -infinity if null
#    range_hi = Column(Numeric()) # +infinity if null
#
#    __mapper_args__ = {
#        "polymorphic_identity": "numeric"
#    }

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey(CategoricalQuestion.id, ondelete="CASCADE"))
    name = Column(String(), nullable=False)
    hidden = Column(Boolean(), nullable=False)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    kind = Column(String(), nullable=False)
    question_id = Column(Integer, ForeignKey(Question.id, ondelete="CASCADE"))
    datetime = Column(DateTime, nullable=False)
    # TODO: add attribution (e.g. to pundits, celebrities, ...)

    __mapper__args = {
        "polymorphic_identity": "prediction",
        "polymorphic_on": kind
    }

    def __init__(self, **kwargs):
        kwargs["datetime"] = dt.datetime.now()
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<Prediction {self.id} at {self.datetime}>"

class CategoricalPrediction(Prediction):
    __tablename__ = "categorical_predictions"

    id = Column(Integer, ForeignKey(Prediction.id), primary_key=True)
    probs = ARRAY(Column("prob", Integer, nullable=False), dimensions=1)

    __mapper_args__ = {
        "polymorphic_identity": "categorical"
    }

class BinaryPrediction(Prediction):
    __tablename__ = "binary_predictions"

    id = Column(Integer, ForeignKey(Prediction.id), primary_key=True)
    prob = Column(Integer, nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "binary"
    }

    def __init__(self, **kwargs):
        kwargs["kind"] = BinaryPrediction.__mapper_args__["polymorphic_identity"]
        super().__init__(**kwargs)

def date_time(date, time):
    return dt.datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=time.hour,
        minute=time.minute
    )

class App:
    def __init__(self, session):
        self._quitting = False
        self._session = session
        self._in_menu = False

        menu = Menu()
        menu.add_submenu("New", [("Question",   self.new_question),
                                 ("Prediction", self.new_prediction)])
        menu.add_submenu("Edit", [("Delete", self.delete_question)])
        menu.add_submenu("Arrange", [("Newest first", self.newest_first),
                                     ("Oldest first", self.oldest_first)])
        self._menu = menu

        picker = Shadow(Box(
            Titled("What kind of question?", ScrollList(selected=curses.A_REVERSE)),
            tl="╔", tr="╗", bl="╚", br="╝", side="║", flat="═"
        ))
        picker.add_item("Binary", None)
        picker.add_item("Categorical", None)
        picker.add_item("Numeric", None)
        picker_rows, picker_cols = picker.dim(curses.LINES, curses.COLS)
        picker_hmargin = (curses.COLS - picker_cols) // 2
        picker_vmargin = (curses.LINES - picker_rows) // 2
        self._picker = Canvas(
            picker,
            rows=picker_rows,
            cols=picker_cols,
            begin_col=picker_hmargin,
            begin_row=picker_vmargin
        )
        self._picker.hide()

        binary_form = Shadow(Box(
            Fill(Form("New Binary Question")),
            tl="╔", tr="╗", bl="╚", br="╝", side="║", flat="═"
        ))
        title = Box(TextLineInput(placeholder="Write your question title here"))
        close_calendar = Box(Calendar())
        close_clock = Box(Clock())
        expiry_calendar = Box(Calendar())
        expiry_clock = Box(Clock())
        infinity_check = Checkbox("∞ is an option")
        discrete_check = Checkbox("this is a discrete question")
        label_set = Box(Tags())
        binary_form.add_field("Title", title, title.contents)
        binary_form.add_field("Close Date", close_calendar, close_calendar.contents)
        binary_form.add_field("Close Time", close_clock, close_clock.contents, required=False)
        binary_form.add_field("Expiry Date", expiry_calendar, expiry_calendar.contents)
        binary_form.add_field("Expiry Time", expiry_clock, expiry_clock.contents, required=False)

        form_width = (3 * curses.COLS) // 4
        form_vmargin = 2
        form_height = curses.LINES - 2 * form_vmargin
        form_hmargin = (curses.COLS - form_width) // 2
        self._binary_form = Canvas(
            binary_form,
            rows=form_height,
            cols=form_width,
            begin_col=form_hmargin,
            begin_row=form_vmargin
        )
        self._binary_form.hide()

        self._binary_prediction = Canvas(
            Shadow(Box(Fill(Form("New Prediction")), tl="╔", tr="╗", bl="╚", br="╝", side="║", flat="═")),
            rows=form_height,
            cols=form_width,
            begin_col=form_hmargin,
            begin_row=form_vmargin
        )
        prob = CategoricalInput(["Yes", "No"])
        self._binary_prediction.add_field("Prediction", prob, prob.contents)

        rows_under_bar = curses.LINES - 1
        info_width = (2 * curses.COLS) // 3
        scroll_width = curses.COLS - info_width

        info = Fill(Info(), horizontal=False)
        self._info = Canvas(info, cols=info_width, rows=rows_under_bar, begin_row=1, begin_col=scroll_width)

        scroll = Fill(ScrollList(), horizontal=False)
        questions = session.query(Question).order_by(Question.close_date.desc()).all()
        for q in questions:
            scroll.add_item(q.title, q, hover=partial(self.display_question, q))
        self._scroll = Canvas(scroll, cols=scroll_width, rows=rows_under_bar, begin_row=1)
        scroll.first() # Draws the first question
        menu.draw()

    def warn(self, msg):
        vmargin = (curses.LINES - 1) // 2
        hmargin = (curses.COLS - len(msg)) // 2
        warning = Canvas(Text(msg), rows=1, cols=20, begin_row = vmargin, begin_col = hmargin)
        warning.set_background(3)
        key, _ = warning.get_and_handle_key()
        if key == Key.QUIT:
            self._quitting = True
        warning.hide()

    def enter_menu(self):
        self._in_menu = True

    def exit_menu(self):
        self._in_menu = False
        self._menu.focus_off()
        self._menu.draw()

    def in_menu(self):
        return self._in_menu

    def display_question(self, q):
        p = self._session.query(Prediction).filter_by(question_id=q.id).all()
        self._info.populate(q, p)
        self._info.draw()

    def new_question(self):
        self.exit_menu()
        self._picker.show()
        active_widget = self._picker
        while True:
            key, handled = active_widget.get_and_handle_key()
            if handled:
                continue
            if key == Key.QUIT:
                self._quitting = True
                return
            if key == Key.ENTER:
                if active_widget is self._picker:
                    picked = self._picker.current_label()
                    if picked == "Binary":
                        self._binary_form.show()
                        active_widget = self._binary_form
                    else:
                        self.warn("Not implemented yet!")
                        continue
                    self._picker.hide()
                elif active_widget is self._binary_form:
                    self._binary_form.hide()
                    info = self._binary_form.read()
                    close  = date_time(info["Close Date"],  info["Close Time"])
                    expiry = date_time(info["Expiry Date"], info["Expiry Time"])
                    q = BinaryQuestion(
                        title=info["Title"],
                        open_date=dt.datetime.now(),
                        close_date=close,
                        expiry_date=expiry
                    )
                    self._session.add(q)
                    self._session.commit()
                    self._scroll.add_item(q.title, q, hover=partial(self.display_question, q), prepend=True)
                    self._scroll.first()
                    self._scroll.draw()
                    return

    def new_prediction(self):
        q = self._scroll.get()
        latest = self._session.query(Prediction).filter_by(question_id=q.id).first() # TODO: sort?
        if q.kind == "binary":
            #self._binary_prediction.populate
            self._binary_prediction.show()
            self._binary_prediction.focus_first()
            while True:
                key, handled = self._binary_prediction.get_and_handle_key()
                if handled:
                    continue
                if key == Key.ESC:
                    self._binary_prediction.hide()
                    return
                if key == Key.QUIT:
                    self._quitting = True
                    return
                if key == Key.ENTER:
                    info = self._binary_prediction.read()
                    probs = info["Prediction"]
                    p = BinaryPrediction(prob=probs["Yes"])
                    self._session.add(p)
                    self._session.commit()
                    return
        else:
            self.warn("Not implemented yet")
            return

    def delete_question(self):
        q = self._scroll.pop()
        self._session.delete(q)
        self._session.commit()
        self.exit_menu()
        self._scroll.draw()

    def newest_first(self):
        self._scroll.sort(lambda q: q.open_date, reverse=True)
        self.exit_menu()
        self._scroll.draw()

    def oldest_first(self):
        self._scroll.sort(lambda q: q.open_date)
        self.exit_menu()
        self._scroll.draw()

    def run(self):
        while not self._quitting:
            if self.in_menu():
                key, handled = self._menu.get_and_handle_key()
                if key == Key.ESC:
                    self.exit_menu()
                    continue
            else:
                key, handled = self._scroll.get_and_handle_key()
                if not handled:
                    self.enter_menu()
                    handled = self._menu.handle_key(key)
                    if not handled:
                        self.exit_menu()
            if handled:
                continue
            if key == Key.QUIT:
                break

def main(stdscr):
    curses.set_escdelay(25) # makes ESC key work
    curses.curs_set(0) # invisible cursor
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_RED)
    engine = create_engine("postgresql:///mydb")
    with Session(engine) as session:
        app = App(session)
        app.run()

if __name__ == "__main__":
    curses.wrapper(main)
