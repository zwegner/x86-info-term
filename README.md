x86-info-term
===
x86-info-term is a curses-based viewer for x86 instruction info built with
Python 3. It combines the following data sources:
* Intrinsics from the [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/).
* Performance info from [uops.info](https://uops.info).

![screenshot](screenshot.png)

Features
===

* **Offline use:** after the data sources are downloaded on the first run, requires no
network access to run
* **More advanced filtering:** multiple filters are allowed, and so is filtering by
regular expression (using [Python's regex syntax](
https://docs.python.org/3/library/re.html#regular-expression-syntax)).
* **Keyboard-only navigation:** using a combination of vim- and emacs-style key bindings,
x86-info-term is sure to delight and/or annoy everyone.
* **100% Pretty Cool**

Key bindings
===

**Scrolling**
* `Ctrl-Y`: one line up
* `Ctrl-E`: one line down
* `Ctrl-U`: one half-page up
* `Ctrl-D`: one half-page down
* `Page Up`: one page up
* `Page Down`: one page down (who would've thought?)
* `h`, `Left`: scroll tables left (right now, just the performance data from uops.info)
* `l`, `Right`: scroll tables right

**Cursor movement**

*This is for the intrinsic-selection cursor, not for text editing: see "Filtering" below for that*
* `j`, `Down`: one row down
* `k`, `Up`: one row up
* `J`: ten rows down
* `K`: ten rows up
* `u`: one half-page up
* `d`: one half-page down
* `g`, `Home`: first row
* `G`, `End`: last row

**Detail view**

*Each intrinsic can be expanded/collapsed to show/hide complete data*
* `Space`, `Enter`: toggle open/close of selected row
* `o`: open selected row
* `c`: close selected row
* `O`: open all rows
* `C`: close all rows

**Filtering**
* `f`, `/`, `Tab`: from the normal browsing mode, start entering a filter
* most normal keys, arrow keys, delete/backspace, etc.: edit the filter with usual key meanings
* `Enter`, `Tab`: keep filter, return to browsing mode
* `Esc`: remove current filter and return to browsing mode
* `Ctrl-F`: move cursor forward one character
* `Ctrl-B`: move cursor back one character
* `Ctrl-W`: delete word before cursor
* `Ctrl-U`: delete line before cursor
* `Ctrl-K`: delete line after cursor

Limitations
===

* For now, performance information is joined to the intrinsics in an imperfect
way: the performance info for all instruction variants that match one of the
instructions listed in the Intrinsics guide are shown. This means that sometimes
spurious instructions are shown, such as showing both loads and stores under
a load intrinsic, or showing instruction variants with the wrong vector sizes.
Normalizing the instruction operand formats used by the Intrinsics Guide and uops.info
to fix this is rather tricky, and for now I have opted for false positives over
false negatives (showing more possibly-useless data rather than less).
