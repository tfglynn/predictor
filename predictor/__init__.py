import calendar
import curses
import datetime as dt
import inspect
import json
import numpy as np
import os
import textwrap
from collections import defaultdict, OrderedDict
from curses import panel
from enum import Enum
from math import ceil, exp, factorial, log, pi, sqrt
from scipy.special import gamma as gammafun
from sqlalchemy import Boolean, Column, ForeignKey, Integer, Numeric, String, DateTime, create_engine
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from time import sleep

PROGRAM = "Predictor"
VERSION = "0"

#
# Interface
#

class QuitCommand(Exception):
    pass

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

# TODO: we need a better system for this
class Key:
    QUIT = ord("q")
    CLOSE = ord("x")

# This can be subclassed to make testing graphs etc possible.
class Writer:
    def write(self, row, col, text):
        pass

class Widget(Writer):
    def __init__(self, nrow, ncol, begin_row, begin_col):
        self._window  = curses.newwin(nrow, ncol, begin_row, begin_col)
        self._panel   = panel.new_panel(self._window)
        self._keymap  = {}
        self._window.keypad(True) # Makes arrow keys work
        self.addkey("q", self.quit)
        self._panel.top()
        panel.update_panels()

    def write(self, row, col, text, *attr):
        try:
            if len(text) == 1:
                self._window.addch(row, col, text, *attr)
            else:
                self._window.addstr(row, col, text, *attr)
        except curses.error:
            # This is necessary for writing on the right edge of the window.
            pass

    def center(self):
        height, width = self.maxrowcol()
        screen_height = curses.LINES
        screen_width  = curses.COLS
        vpad = (screen_height - height) // 2
        hpad = (screen_width - width) // 2
        self._panel.move(vpad, hpad)
        panel.update_panels()

    def hide(self):
        self._panel.hide()
        panel.update_panels()

    def show(self):
        self._panel.show()
        panel.update_panels()

    def top(self):
        self._panel.top()
        panel.update_panels()

    def noutrefresh(self):
        self.show()
        self._window.noutrefresh()

    def quit(self):
        raise QuitCommand

    def maxrowcol(self):
        return self._window.getmaxyx()

    def addkey(self, key, callback, aliases=[]):
        keys = [key] + aliases
        for key in keys:
            if isinstance(key, str):
                key = ord(key)
            self._keymap[key] = callback

    def delkey(self, key, callback):
        # TODO: handle aliases?
        del(self._keymap[key])

    def react(self):
        self._panel.show()
        self._panel.top()
        panel.update_panels()
        curses.doupdate()
        c = self._window.getch()
        try:
            callback = self._keymap[c]
            callback()
        except KeyError:
            pass
        return c

class Shadow(Widget):
    def draw(self):
        #shade = "░▒▓▞"
        shade = "░"
        tl = "┌"; tr = "┐"
        bl = "└"; br = "┘"
        side = "│"; flat = "─"

        maxrow, maxcol = super().maxrowcol()

        # Draw box
        super().write(0, 0, tl)
        super().write(0, maxcol - 2, tr)
        super().write(maxrow - 2, 0, bl)
        super().write(maxrow - 2, maxcol - 2, br)
        for col in [0, maxcol - 2]:
            for row in range(1, maxrow - 2):
                super().write(row, col, side)
        for row in [0, maxrow - 2]:
            for col in range(1, maxcol - 2):
                super().write(row, col, flat)

        # Draw shadow
        for col in range(1, maxcol - 1):
            super().write(maxrow - 1, col, shade)
        for row in range(1, maxrow):
            try:
                super().write(row, maxcol - 1, shade)
            except curses.error:
                pass

        self.noutrefresh()

    def maxrowcol(self):
        row, col = super().maxrowcol()
        return row - 3, col - 3 # side + side + shadow = 3

    def write(self, row, col, *attr):
        super().write(row + 1, col + 1, *attr)

class QuestionForm(Shadow):
    def __init__(self):
        super().__init__(curses.LINES - 3, curses.COLS - 2, 2, 1)
        self._title = ""
        self._close_date = None
        self._is_discrete = False
        self._is_date = False
        self._has_infinity = False
        self._active = "title"

    def get_input(self):
        self.top()
        try:
            curses.curs_set(2) # visible cursor
            curses.echo()
            self._window.move(1, 8)
            while True:
                if self._active == "title":
                    currow, curcol = self._window.getyx()
                    self.draw() # may move cursor, so we have to reset it
                    self._window.move(currow, curcol)
                else:
                    self.draw()
                curses.doupdate()
                c = self._window.getch()
                if c == ord("q") and not self._active == "title":
                    raise QuitCommand
                if self._active == "title":
                    if c == curses.KEY_BACKSPACE:
                        row, col = self._window.getyx()
                        self.write(row - 1, col - 1, " ")
                        self._window.move(row, col)
                    elif c == ord("\n"):
                        break
                    elif c == ord("\t"):
                        self._active = "discrete"
                elif self._active == "discrete":
                    if c == ord("\n"):
                        self._is_discrete = not self._is_discrete
                    if c == ord("\t"):
                        self._active = "date"
                elif self._active == "date":
                    if c == ord("\n"):
                        self._is_date = not self._is_date
                    if c == ord("\t"):
                        self._active = "infinity"
                elif self._active == "infinity":
                    if c == ord("\n"):
                        self._has_infinity = not self._has_infinity
            curses.curs_set(0)
            curses.noecho()
        finally:
            curses.curs_set(0)
            curses.noecho()
        self.hide()

    def draw(self):
        self.write(0, 0, "Title: ")
        self.write(1, 0, "Close date: ")

        if self._active == "discrete":
            self.write(2, 0, "☑" if self._is_discrete else "☐", curses.A_REVERSE)
            self.write(2, 2, "discrete values")
        else:
            self.write(2, 0, "☑" if self._is_discrete else "☐")
            self.write(2, 2, "discrete values")

        if self._active == "date":
            self.write(3, 0, "☑" if self._is_date else "☐", curses.A_REVERSE)
            self.write(3, 2, "is a date")
        else:
            self.write(3, 0, "☑" if self._is_date else "☐")
            self.write(3, 2, "is a date")

        if self._active == "infinity":
            self.write(4, 0, "☑" if self._has_infinity else "☐", curses.A_REVERSE)
            self.write(4, 2, "∞ is an option")
        else:
            self.write(4, 0, "☑" if self._has_infinity else "☐")
            self.write(4, 2, "∞ is an option")

        super().draw()

class DateForm(Shadow):
    def __init__(self, begin_row, begin_col, start=None, **kwargs):
        super().__init__(11, 23, begin_row, begin_col, **kwargs)
        if start is None:
            start = dt.date.today()
        self._date = start
        self._cursor_on_month = False
        self.addkey(curses.KEY_RIGHT, self.right, aliases=["l"])
        self.addkey(curses.KEY_LEFT,  self.left, aliases=["h"])
        self.addkey(curses.KEY_UP,    self.up, aliases=["k"])
        self.addkey(curses.KEY_DOWN,  self.down, aliases=["j"])

    def get_date(self):
        self._date = dt.date.today()
        self.show()
        self.draw()
        while True:
            key = self.react()
            if key == Key.CLOSE:
                self.hide()
                return None
            if key == ord("\n") and not self._cursor_on_month:
                self.hide()
                return self._date

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
        self.draw()

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
        self.draw()

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
        self.draw()

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
        self.draw()

    def draw(self):
        self._window.clear()
        date = self._date
        _, cols = self.maxrowcol()
        if date.year != dt.date.today().year:
            topline = date.strftime("%B %Y")
        else:
            topline = date.strftime("%B")

        lpad = (cols - len(topline)) // 2
        if self._cursor_on_month:
            self.write(0, lpad, topline, curses.A_STANDOUT)
        else:
            self.write(0, lpad, topline)
        if self._cursor_on_month:
            self.write(0, 0, "◀")
            self.write(0, cols - 1, "▶")
        self.write(1, 0, "S  M  T  W  T  F  S ")

        first, ndays = calendar.monthrange(date.year, date.month)
        # Make Sunday is the first day of the week
        first = (first + 1) % 7 # Sunday = 0

        for day in range(1, ndays + 1):
            row = (first + day - 1) // 7
            col = ((first + day - 1) % 7)
            col = 3 * col
            if day == date.day and not self._cursor_on_month:
                self.write(row + 2, col, f"{day:2}", curses.A_STANDOUT)
            else:
                self.write(row + 2, col, f"{day:2}")

        super().draw()

class TimeForm(Shadow):
    def __init__(self, begin_row, begin_col, start=None, **kwargs):
        super().__init__(6, 11, begin_row, begin_col, **kwargs)
        if start is None:
            start = dt.datetime.now()
        self._time=start
        self.addkey(curses.KEY_RIGHT, self.right, aliases=["l"])
        self.addkey(curses.KEY_LEFT,  self.left, aliases=["h"])
        self.addkey(curses.KEY_UP,    self.up, aliases=["k"])
        self.addkey(curses.KEY_DOWN,  self.down, aliases=["j"])
        self._cursor = 0 # 0 = hour, 1 = minute, 2 = AM/PM

    def right(self):
        if self._cursor != 2:
            self._cursor += 1
        self.draw()

    def left(self):
        if self._cursor != 0:
            self._cursor -= 1
        self.draw()

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
        self.draw()

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
        self.draw()

    def draw(self):
        self._window.clear()
        time = self._time
        _, cols = self.maxrowcol()
        hour = time.hour % 12
        if hour == 0:
            hour = 12
        hour = f"{hour:2}"
        minute = f"{time.minute:02}"
        am_pm = f"{time.strftime('%p')}"
        if self._cursor == 0:
            self.write(0, 1, "▲")
            self.write(1, 0, hour, curses.A_STANDOUT)
            self.write(2, 1, "▼")
        else:
            self.write(1, 0, hour)
        self.write(1, 2, ":")
        if self._cursor == 1:
            self.write(0, 4, "▲")
            self.write(1, 3, minute, curses.A_STANDOUT)
            self.write(2, 4, "▼")
        else:
            self.write(1, 3, minute)
        if self._cursor == 2:
            self.write(0, 6, "▲")
            self.write(1, 6, am_pm, curses.A_STANDOUT)
            self.write(2, 6, "▼")
        else:
            self.write(1, 6, am_pm)
        super().draw()

    def get_time(self):
        self._time = dt.datetime.now()
        self.show()
        self.draw()
        while True:
            key = self.react()
            if key == Key.CLOSE:
                self.hide()
                return None
            if key == ord("\n"):
                self.hide()
                return self._time

class Info(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._question = None
        self._predictions = []
        self.addkey("m", self.maximize)

    def draw(self):
        self._window.clear()
        rows, cols = self.maxrowcol()
        q = self._question
        cd = q.close_date
        closed = cd <= dt.datetime.today()

        # Print title at top
        title = textwrap.wrap(q.title, width=cols)
        for i, line in enumerate(title):
            self.write(i, 0, line, curses.A_BOLD)
        pad = len(title)

        # Print close date
        if cd is None:
            self.write(pad, 0, "close date not set", curses.A_ITALIC)
        else:
            verb = "closed" if closed else "closes"
            d = f"{cd.strftime('%b')} {cd.day}, {cd.year}"
            h = 12 if cd.hour % 12 == 0 else cd.hour % 12
            t = f"{h}:{cd.strftime('%M %p')}"
            self.write(pad, 0, f"{verb} on {d} at {t}", curses.A_ITALIC)
        pad += 2

        # Print latest prediction
        predictions = self._predictions
        if not predictions:
            msg = "No predictions"
            if not closed:
                msg += " yet"
            msg = msg.center(cols)
            self.write(pad, 0, msg, curses.A_DIM)
            self.noutrefresh()
            return

        latest = predictions[0]
        distinfo = latest.json

        # Special case: categorical distributions
        labels = q.labels or [] # ensure it's not None
        if labels:
            # In this case, the prediction should be a list of
            # probability-label objects.
            pdict = defaultdict(int)
            for obj in latest.json:
                pdict[obj["label"]] = obj["p"]
            credences = []
            for label in labels:
                credences.append(pdict[label])

            # How wide will the labels be?
            max_label_width = cols // 3
            # Wrap lines
            line_groups = [textwrap.wrap(label, width=max_label_width) for label in labels]
            # Figure out how wide the widest label line will be
            label_width = max([len(line) for group in line_groups for line in group])
            # Right-align all the lines
            line_groups = [[line.rjust(label_width) for line in group] for group in line_groups]

            height = rows - pad - 2 # take off 2 for the probability axis
            width = (
                cols -
                1 - # axis
                1 - # pad
                label_width # labels
            )

            # Print the probability axis
            self.write(pad, label_width + 1, "0%")
            self.write(pad, cols - 4, "100%")
            self.write(pad + 1, label_width + 1, "├")
            self.write(pad + 1, cols - 1, "┘")
            for i in range(label_width + 2, cols - 1):
                self.write(pad + 1, i, "─")

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
                    if height_used + len(group) + 1 > height:
                        break
                    can_show += 1
                    height_used += len(group) + 1 # pad above
                current_row = pad + 2
                for i in range(can_show):
                    if i > 0:
                        self.write(current_row, label_width + 1, "│")
                        current_row += 1 # pad
                    self.write(current_row, 0, line_groups[i][0])
                    self.write(current_row, label_width + 1, "┤")
                    bar = round(credences[i] * width)
                    for j in range(bar):
                        self.write(current_row, label_width + 2 + j, " ", curses.A_REVERSE)
                    numstr = f"{round(100 * credences[i])}%"
                    if bar + len(numstr) <= width:
                        self.write(current_row, label_width + 2 + bar, numstr)
                    else:
                        self.write(current_row, label_width + 2 + bar - len(numstr), numstr, curses.A_REVERSE)
                    current_row += 1
                    for line in line_groups[i][1:]:
                        self.write(current_row, 0, line)
                        self.write(current_row, label_width + 1, "│")
                        current_row += 1

                if can_show < len(line_groups):
                    # We're missing some
                    missing = len(line_groups) - can_show
                    self.write(height + pad + 1, label_width + 1, f"┆ ({missing} more not shown)")
            self.noutrefresh()
            return

        if isinstance(distinfo, list):
            dists = [(obj["p"], Distribution.from_json(obj["distribution"])) for obj in distinfo]
        else:
            dists = [(1, Distribution.from_json(latest.json))]
        isolated_masses = [d.point() for _, d in dists if isinstance(d, PointMass)]
        has_infinity = any([x == float("inf") for x in isolated_masses])
        all_discrete = all([isinstance(d, Discrete) for _, d in dists])
        has_infinity = q.has_infinity
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
        height = rows - pad - 3 # TODO: why 3 not 2?
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
        hard_xmin = q.range_lo or -float("inf")
        hard_xmax = q.range_hi or  float("inf")
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
        with open("log.txt", "w") as outfile:
            print(xs, file=outfile)

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
        self.write(pad, 5, "┐")
        for row in range(1, height):
            self.write(pad + row, 5, "│")
        self.write(pad + height, 5, "┘")

        # Draw the top y label
        # TODO: handle very large ymax
        ydigits = 2 if ymax < 10 else 1
        yerr = abs(round(ymax, ydigits) - ymax)
        if yerr < ystep / 10:
            maxlab = str(round(ymax, ydigits))
        else:
            maxlab = "~" + str(round(ymax, ydigits - 1))
        self.write(pad, 0, maxlab.rjust(4))

        # Draw the bottom y label
        # TODO: add digits after decimal if appropriate
        minlab = "0".ljust(ydigits + 1)
        self.write(pad + height, 0, minlab.rjust(4))

        # TODO: other y labels?  (need to decide before drawing axis)

        # Draw the x-axis
        # We need to decide what the labels will be first.
        for col in range(width):
            self.write(pad + height + 1, col + 6, "─")
        # TODO: fix
        if has_infinity:
            level = round(mass_at_infinity / ystep)
            self.write(pad + height + 1, width + 6, "╌")
            self.write(pad + height + 1, width + 6 + 1, "┬")
            self.write(pad + height + 2, width + 6 + 1, "∞")
            for i in range(level):
                self.write(pad + height - i, width + 6 + 1, "▓")
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
                self.write(pad + next_row, i + 5, "╮")
                next_row += 1
                while next_row < row_approx:
                    self.write(pad + next_row, i + 5, "│")
                    next_row += 1
                self.write(pad + next_row, i + 5, "╰")
            elif next_row > row_approx:
                self.write(pad + next_row, i + 5, "╯")
                next_row -= 1
                while next_row > row_approx:
                    self.write(pad + next_row, i + 5, "│")
                    next_row -= 1
                self.write(pad + next_row, i + 5, "╭")
            else:
                self.write(pad + next_row, i + 5, "─")

        self.noutrefresh()

    def populate(self, question, predictions):
        self._question = question
        self._predictions = predictions

    def maximize(self):
        self._window.resize(curses.LINES, curses.COLS)
        self._window.mvwin(0, 0)
        self._panel.top()
        panel.update_panels()

# TODO: make refresh more efficient so we don't see glitches

class Listing(Widget):
    def __init__(self, *args, wait=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.addkey(curses.KEY_DOWN, self.next, aliases=["j"])
        self.addkey(curses.KEY_UP,   self.prev, aliases=["k"])
        self.addkey("g", self.first)
        self.addkey("G", self.last)
        if wait:
            self.addkey("\n", self.act)
        self._wait = wait
        self._items = []
        self._cursor_item = 0 # index into list of items
        self._cursor_row  = 0 # row of window where cursor is

    def loop(self):
        while True:
            c = self.react()
            # TODO: I'm adding this for the topbar menus
            if c == ord("x"):
                break

    def draw(self):
        nitems = len(self._items)
        cursor_item = self._cursor_item
        cursor_row  = self._cursor_row
        rows, cols  = self.maxrowcol()
        begin_row   = 0 # Row where the menu starts, not counting arrows

        if nitems <= rows: # The whole menu fits on screen
            menu = self._items
            begin_index = 0
        else:
            # If the menu is too big for the screen,
            # we may need to draw arrows to indicate
            # there are more items above/below the screen
            # that aren't shown.
            narrows = "↑ more ↑".center(cols)
            sarrows = "↓ more ↓".center(cols)
            space_above = cursor_row
            if cursor_item > space_above:
                assert(space_above > 0)
                # Need to draw arrows on top
                self.write(0, 0, narrows, curses.A_LEFT)
                space_above -= 1
                begin_row = 1
            space_below = rows - cursor_row - 1
            if nitems - cursor_item - 1 > space_below:
                assert(space_below > 0)
                # Need arrows on bottom
                self.write(rows - 1, 0, sarrows)
                space_below -= 1
            begin_index = cursor_item - space_above
            end_index = cursor_item + space_below + 1
            menu = self._items[begin_index:end_index]

        for i, (label, _) in enumerate(menu):
            label = textwrap.shorten(label, cols, placeholder=" …")
            label = label.ljust(cols) # clears long titles while scrolling
            if i + begin_index == cursor_item:
                self.write(i + begin_row, 0, label, curses.A_STANDOUT)
            else:
                self.write(i + begin_row, 0, label)

        self.noutrefresh()

    def populate(self, items):
        # TODO: do we need to account for empty lists?
        self._items = items
        self._cursor_item = 0 # index into list of items
        self._cursor_row  = 0 # row of window where cursor is
        if self._items and not self._wait:
            self._items[0][1](self)
        self.draw()

    def next(self):
        rows, _ = self.maxrowcol()
        nitems  = len(self._items)
        if not self._cursor_item < nitems - 1: # last item
            return
        self._cursor_item += 1
        if not self._wait:
            self.act()
        # Don't go forward if we were at an arrow
        if (nitems <= rows or
            self._cursor_row < rows - 2 or
            nitems - self._cursor_item < 2):
            self._cursor_row += 1
        self.draw()

    def prev(self):
        rows, _ = self.maxrowcol()
        nitems  = len(self._items)
        if self._cursor_item == 0: # first item
            return
        self._cursor_item -= 1
        if not self._wait:
            self.act()
        # Don't go back if we were at an arrow
        if (nitems <= rows or
            self._cursor_row > 1 or
            self._cursor_item < 1):
            self._cursor_row -= 1
        self.draw()

    def act(self):
        _, action = self._items[self._cursor_item]
        if action is not None:
            action(self)

    def first(self):
        self._cursor_item = 0
        self._cursor_row  = 0
        if not self._wait:
            self.act()
        self.draw()

    def last(self):
        self._cursor_item = len(self._items) - 1
        self._cursor_row  = self.maxrowcol()[0] - 1
        if not self._wait:
            self.act()
        self.draw()

class ListDict(OrderedDict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)

class Topbar(Widget):
    def __init__(self):
        cols = curses.COLS
        super().__init__(1, cols, 0, 0)
        self._window.bkgd(" ", curses.color_pair(1))
        self._submenus = ListDict()
        self._active_submenu = None
        # TODO: sort (or "arrange"), search, view, ...

    def add_submenu(self, name, items): # TODO: aliases
        col = sum([len(n) for n in self._submenus]) + len(self._submenus)
        width = max([len(label) for label, _ in items])
        lst = Listing(len(items), width, 1, col, wait=True)
        lst._window.bkgd(" ", curses.color_pair(1))
        lst.populate(items)
        self._submenus[name] = lst
        initial = name[0].lower()
        self.addkey(initial, self.submenu(name))

    def submenu(self, name):
        def f():
            self._active_submenu = name
            self.draw()
            self._submenus[name].loop()
            self._active_submenu = None
            self.draw()
        return f

    def draw(self):
        col = 0
        for name in self._submenus:
            initial = name[0]
            rest = name[1:]
            if name == self._active_submenu:
                self.write(0, col, initial, curses.A_UNDERLINE | curses.A_REVERSE)
                self.write(0, col + 1, rest, curses.A_REVERSE)
            else:
                self.write(0, col, initial, curses.A_UNDERLINE)
                self.write(0, col + 1, rest)
            col += len(name) + 1
        progname = PROGRAM + " v" + VERSION
        self.write(0, curses.COLS - len(progname), progname, curses.A_ITALIC)
        if self._active_submenu:
            self._submenus[self._active_submenu].draw()
        self.noutrefresh()

#
# Database
#

Base = declarative_base()

class Prediction(Base):
    """A prediction for a specific question at a specific time."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("question.id"))
    datetime = Column(DateTime, nullable=False)
    json = Column(JSON, nullable=False)
    # TODO: add attribution (e.g. to pundits, celebrities, ...)

    def __repr__(self):
        return f"<Prediction {self.id} at {self.datetime}>"

class Question(Base):
    """A question for which predictions can be made"""

    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)
    title = Column(String(), nullable=False)
    discrete = Column(Boolean(), nullable=False)
    is_date = Column(Boolean(), nullable=False)
    has_infinity = Column(Boolean(), nullable=False)
    range_lo = Column(Numeric()) # -infinity if null
    range_hi = Column(Numeric()) # +infinity if null
    labels = Column(ARRAY(String))
    description = Column(String())
    open_date = Column(DateTime, nullable=False)
    close_date = Column(DateTime)

    def __init__(self, **kwargs):
        if not kwargs["title"].endswith("?"):
            raise ValueError("title should end in '?'")
        Base.__init__(self, **kwargs)

    def __repr__(self):
        return f"<Question {self.id}: {self.title}>"

def main(stdscr):
    curses.curs_set(0) # invisible cursor
    curses.start_color # enables colors
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)

    maxrow = curses.LINES
    maxcol = curses.COLS
    qwidth = int(maxcol / 3)

    listing = Listing(maxrow - 1, qwidth, 1, 0)
    info = Info(maxrow - 1, maxcol - qwidth - 1, 1, qwidth + 1)
    date = DateForm(10, 0)
    time = TimeForm(0, 0)
    date.center(); date.hide()
    time.center(); time.hide()
    qform = QuestionForm()
    pform = QuestionForm() # TODO
    topbar = Topbar()
    topbar.add_submenu("New", [("Question", lambda _: qform.get_input()),
                               ("Prediction", lambda _: pform.get_input())])
    topbar.draw()
    topbar.noutrefresh()

    # Connect to the database
    engine = create_engine("postgresql:///mydb")
    with Session(engine) as session:
        questions = session.query(Question).order_by(Question.close_date.desc()).all()
        def display(q):
            def f(listing):
                p = session.query(Prediction).filter_by(question_id=q.id).all()
                info.populate(q, p)
                info.draw()
            return f
        listing.populate([(q.title, display(q)) for q in questions])
        listing.addkey("c", date.get_date)
        listing.addkey("t", time.get_time)
        listing.addkey("n", topbar.submenu("New"))
        listing.addkey("e", topbar.submenu("Edit"))
        try:
            listing.loop()
        except QuitCommand:
            pass

if __name__ == "__main__":
    curses.wrapper(main)
