# predictor

This is a command-line tools for tracking predictions about real-world events.
It uses a curses GUI and a PostgreSQL database for storage.

Inspired by Metaculus, the goal is to be more flexible by, for instance,
allowing discrete questions that have more than two possible outcomes.
For example, who will replace Breyer on the Supreme Court?
Metaculus handles this question by having multiple questions, e.g.
[here][Kruger] and [here][KBJ].
You'll sometimes see people try to hack a numeric range question if the number
of possibilities is large.

This program solves this problem by supporting questions of this type natively.

![Three bars for predicting one of three SCOTUS candidates](/img/cat.png)

The whole thing is still in development, but it's in a mostly-usable state right now.

[Kruger]: https://www.metaculus.com/questions/9585/l-kruger-confirmed-to-scotus-before-2023/
[KBJ]: https://www.metaculus.com/questions/9584/kbj-confirmed-to-scotus-before-2023/
