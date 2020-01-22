#!/usr/bin/env python3
import curses
import re
import sys
import xml.etree.ElementTree as ET
from enum import Enum, auto

class Mode(Enum):
    BROWSE = auto()
    FILTER = auto()

# Just a basic container class for keeping some attrs together
class DummyObj:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

# "child classes" that are just aliases because we don't care
Context = DummyObj
Table = DummyObj

def clip(value, lo, hi):
    return max(lo, min(value, hi - 1))

################################################################################
## Key bindings ################################################################
################################################################################

def CTRL(k):
    return chr(ord(k) & 0x1f)
ESC = CTRL('[')
DEL = '\x7f'

# Key map. Screen-size dependent values are lambdas
SCROLL_KEYS = {
    CTRL('E'):  +1,
    CTRL('Y'):  -1,
    CTRL('D'):  lambda r, c: +(r // 2),
    CTRL('U'):  lambda r, c: -(r // 2),
}
CURS_KEYS = {
    'd':        lambda r, c: +(r // 2),
    'u':        lambda r, c: -(r // 2),
    'g':        -1e10,
    'G':        +1e10,
    'j':        +1,
    'J':        +10,
    CTRL('J'):  +10,
    'k':        -1,
    'K':        -10,
    CTRL('K'):  -10,
}

# Mode-specific keys
CMD_KEYS = {
    Mode.BROWSE: {
        'q': 'quit',
        ESC: 'quit',

        'f': 'start-filter',
        '/': 'start-filter',

        ' ': 'toggle-fold',
        'o': 'open-fold',
        'O': 'open-all-folds',
        'c': 'close-fold',
        'C': 'close-all-folds',
    },
    Mode.FILTER: {
        ESC: 'clear-filter',
        '\n': 'start-browse',
        DEL: 'backspace',
    },
}

# Calculate an offset from scroll/cursor movement based on the screen size
def get_offset(table, key):
    offset = table[key]
    if callable(offset):
        offset = offset(curses.LINES, curses.COLS)
    return offset

################################################################################
## Text rendering helpers ######################################################
################################################################################

def pad(s, width, right=False):
    # Pad and truncate
    if right:
        return (' ' * width + s)[-width:]
    else:
        return (s + ' ' * width)[:width]

def wrap_lines(cell, width):
    cell = cell.strip()
    cell = cell.replace('\t', ' ' * 4)
    for line in cell.splitlines():
        while len(line) > width:
            split = line.rfind(' ', 0, width)
            if split == -1:
                split == width
            [chunk, line] = line[:split], line[split:]
            yield chunk
        yield line

def get_col_width(table, col):
    width = 0
    for row in table:
        cells = row['cells'] if isinstance(row, dict) else row
        cell = cells[col]
        if not isinstance(cell, str):
            [cell, info] = cell
        width = max(width, len(cell))
    return width

################################################################################
## Intrinsic info ##############################################################
################################################################################

INTR_COLORS = {
    'MMX':          11,
    'SSE':          46,
    'SSE2':         154,
    'SSE3':         34,
    'SSSE3':        30,
    'SSE4.1':       24,
    'SSE4.2':       12,
    'AVX':          54,
    'AVX2':         127,
    'FMA':          162,
    'AVX-512':      196,
    'KNC':          208,
    'AVX-512/KNC':  9,
    'SVML':         39,
    'SVML/KNC':     39,
    'Other':        252,
}

def parse_intrinsics_guide(path):
    root = ET.parse(path)

    table = []
    for i, intrinsic in enumerate(root.findall('intrinsic')):
        tech = intrinsic.attrib['tech']
        name = intrinsic.attrib['name']
        desc = [d.text for d in intrinsic.findall('description')][0]
        insts = [(inst.attrib['name'], inst.attrib.get('form', ''))
                for inst in intrinsic.findall('instruction')]
        key = '%s %s %s %s' % (tech, name, desc, ' '.join(n for n, f in insts))
        table.append({
            'id': i,
            'tech': tech,
            'name': name,
            'params': [(p.attrib['varname'], p.attrib['type'])
                for p in intrinsic.findall('parameter')],
            'return_type': intrinsic.attrib['rettype'],
            'desc': desc,
            'operations': [op.text for op in intrinsic.findall('operation')],
            'insts': ['%s %s' % (n, f) for n, f in insts],
            'search-key': key.lower(),
        })
    return table

def get_intr_info_table(intr):
    blank_row = ['', '']
    inst_rows = [['Instruction' if i == 0 else '', inst]
            for i, inst in enumerate(intr['insts'])]
    op_rows = [['Operation', [op, {'attr': 'code', 'wrap': True}]] for op in intr['operations']]
    rows = [
        blank_row,
        ['Description', [intr['desc'], {'wrap': True}]],
        blank_row,
        *inst_rows,
        blank_row,
        *op_rows,
        blank_row,
    ]
    # Clean up title column
    for row in rows:
        if row[0]:
            row[0] = ['%s:    ' % row[0], {'attr': 'bold'}]
    return Table(rows=rows, widths=[20, 0], alignment=[1, 0])

def get_intr_table(intrinsics, start, stop, folds={}):
    # Gather table data
    rows = []
    for i, intr in enumerate(intrinsics[start:stop]):
        expand = (intr['id'] in folds)

        params = ', '.join('%s %s' % (type, param) for param, type in intr['params'])
        tech = intr['tech']
        rows.append({
            'id': i + start,
            'cells': [
                [tech, {'attr': tech}],
                # HACK: pad on both sides
                ' %s ' % intr['return_type'],
                ['%s(%s)' % (intr['name'], params), {'wrap': expand}],
            ],
            'subtables': [get_intr_info_table(intr)] if expand else [],
        })

    if not rows:
        rows = [{'id': 0, 'cells': ['No results.']}]
        return Table(rows=rows, widths=[curses.COLS], alignment=[0])
    widths = [12, get_col_width(rows, 1), 0]
    return Table(rows=rows, widths=widths, alignment=[0, 1, 0])

################################################################################
## Curses stuff ################################################################
################################################################################

def draw_table(ctx, table, start_row, stop_row, curs_row=None):
    # Right column is shrunk or padded out to screen width
    if len(table.widths) > 1:
        table.widths[-1] = curses.COLS - sum(table.widths[:-1])

    # Draw the table
    line = start_row
    for row in table.rows:
        if line >= stop_row:
            break

        if not isinstance(row, dict):
            row = {'cells': row}
        row_id = row.get('id', -1)
        cells = row['cells']
        subtables = row.get('subtables', [])
        hl = curses.A_REVERSE if row_id == curs_row else 0

        # Line-wrap and pad all cells. Do this in a separate loop to get the
        # maximum number of lines in a cell
        wrapped_cells = []
        for width, alignment, cell in zip(table.widths, table.alignment, cells):
            info = {}
            if not isinstance(cell, str):
                [cell, info] = cell
            attr = ctx.attrs[info.get('attr', 'default')]
            if info.get('wrap', False):
                cell_lines = wrap_lines(cell, width)
            else:
                cell_lines = [cell]

            cell_lines = [pad(cell, width, right=alignment) for cell in cell_lines]
            wrapped_cells.append((cell_lines, attr, width))

        n_lines = max(len(c) for c, a, w in wrapped_cells)
        # Render lines
        col = 0
        for cell, attr, width in wrapped_cells:
            # Pad vertically
            cell += [' ' * width] * (n_lines - len(cell))
            wrap_line = line
            for cell_line in cell:
                if wrap_line >= stop_row:
                    break
                ctx.window.insstr(wrap_line, col, cell_line, attr | hl)
                wrap_line += 1
            col += width
        line += n_lines

        # Render subtables if necessary
        for subtable in subtables:
            line = draw_table(ctx, subtable, line, stop_row, curs_row=None)

    return line

def update_filter(ctx):
    if ctx.filter is not None:
        filter_list = ctx.filter.lower().split()
        new_fd = []
        # Try filtering with input regexes. If the parse fails, keep the old
        # filtered data and annoy the user by flashing an error
        try:
            for intr in ctx.intr_data:
                for f in filter_list:
                    if re.search(f, intr['search-key']) is None:
                        break
                else:
                    new_fd.append(intr)
            ctx.filtered_data = new_fd
        except re.error as e:
            ctx.flash_error = str(e)
        ctx.curs_row = 0
        ctx.start_row = 0
    else:
        ctx.filtered_data = ctx.intr_data

DARK_MODE = True

def main(stdscr, intr_data):
    colors = {k: (curses.COLOR_BLACK, v) for k, v in INTR_COLORS.items()}
    fg, bg = (15, 0) if DARK_MODE else (0, 15)
    colors.update({
        'default':  (fg, bg),
        'sep':      (bg, 12),
        'error':    (fg, 196),
        'bold':     (fg, bg, curses.A_BOLD),
        'code':     (fg, 237 if DARK_MODE else 251),
    })
    # Create attributes
    attrs = {}
    for tech, (fg, bg, *extra) in colors.items():
        n = len(attrs) + 1
        curses.init_pair(n, fg, bg)
        attrs[tech] = curses.color_pair(n) + sum(extra, 0)

    curses.raw()
    # Make cursor invisible, get an alias for stdscr
    curses.curs_set(0)
    # Create a big dummy object for passing around a bunch of random state
    ctx = Context(window=stdscr, mode=Mode.BROWSE, intr_data=intr_data,
            filter=None, filtered_data=[], flash_error=None,
            curs_row=0, start_row=0, attrs=attrs, folds=set())

    update_filter(ctx)

    while True:
        # Clear screen
        ctx.window.clear()

        # Get a layout table of all filtered intrinsics. Narrow the range down
        # by the rows on screen so we can set dynamic column widths
        table = get_intr_table(ctx.filtered_data, ctx.start_row,
                ctx.start_row + curses.LINES, folds=ctx.folds)

        # Draw status lines
        filter_line = None
        status_lines = []
        if ctx.flash_error is not None:
            status_lines.append((ctx.flash_error, 'error'))
            ctx.flash_error = None
        if ctx.filter is not None:
            hl = 'bold' if ctx.mode == Mode.FILTER else 'default'
            filter_line = 'Filter: %s' % ctx.filter
            status_lines.append((filter_line, hl))

        if status_lines:
            status_lines = [('', 'sep')] + status_lines
            for i, (line, attr) in enumerate(status_lines):
                row = curses.LINES - (len(status_lines) - i)
                ctx.window.insstr(row, 0, pad(line, curses.COLS), ctx.attrs[attr])

        draw_table(ctx, table, 0, curses.LINES - len(status_lines),
                curs_row=ctx.curs_row)

        # Show the cursor for filter mode: always at the end of the row.
        # Ugh I hope this is good enough, I really don't want to reimplement
        # readline.
        if ctx.mode == Mode.FILTER and filter_line is not None:
            curs_col = clip(len(filter_line), 0, curses.COLS - 1)
            ctx.window.move(curses.LINES - 1, curs_col)
            curses.curs_set(1)
        else:
            curses.curs_set(0)

        # Draw the window
        ctx.window.refresh()

        # Get input
        try:
            key = ctx.window.getkey()
        except curses.error:
            continue
        # Mode-specific commands
        if key in CMD_KEYS[ctx.mode]:
            cmd = CMD_KEYS[ctx.mode][key]
            # Mode switches
            if cmd == 'start-filter':
                ctx.mode = Mode.FILTER
                ctx.filter = ctx.filter or ''
            elif cmd == 'start-browse':
                ctx.mode = Mode.BROWSE
            elif cmd == 'clear-filter':
                ctx.filter = None
                update_filter(ctx)
                ctx.mode = Mode.BROWSE

            # Folds
            elif 'fold' in cmd:
                selection = ctx.filtered_data[ctx.curs_row]['id']
                if cmd == 'open-fold':
                    ctx.folds |= {selection}
                elif cmd == 'open-all-folds':
                    ctx.folds = set(range(len(ctx.intr_data)))
                elif cmd == 'close-fold':
                    ctx.folds -= {selection}
                elif cmd == 'close-all-folds':
                    ctx.folds = set()
                elif cmd == 'toggle-fold':
                    ctx.folds ^= {selection}
                else:
                    assert False, cmd

            # Input editing
            elif cmd == 'backspace':
                ctx.filter = ctx.filter[:-1]
                update_filter(ctx)

            elif cmd == 'quit':
                return

            else:
                assert False, cmd
        elif key == 'KEY_RESIZE':
            curses.update_lines_cols()
        # Filter text input
        elif ctx.mode == Mode.FILTER and key.isprintable():
            ctx.filter += key
            update_filter(ctx)
        # Scroll
        elif key in SCROLL_KEYS:
            ctx.start_row += get_offset(SCROLL_KEYS, key)
            ctx.start_row = clip(ctx.start_row, 0, len(ctx.filtered_data))
            ctx.curs_row = clip(ctx.curs_row, ctx.start_row, ctx.start_row + curses.LINES)
        # Move cursor
        elif key in CURS_KEYS:
            ctx.curs_row += get_offset(CURS_KEYS, key)
            ctx.curs_row = clip(ctx.curs_row, 0, len(ctx.filtered_data))
            ctx.start_row = clip(ctx.start_row, ctx.curs_row - curses.LINES, ctx.curs_row + 1)
        else:
            ctx.flash_error = 'Unknown key: %r' % key

if __name__ == '__main__':
    # Command line "parsing"
    path = 'data-latest.xml'
    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        path = sys.argv[2]
    if '--light' in sys.argv:
        DARK_MODE = False

    try:
        intr_data = parse_intrinsics_guide(path)
    except FileNotFoundError:
        print('Usage: %s [-i INTRINSIC_XML_PATH]\n'
                'To use this, you must download the intrinsic XML data from:\n'
                '  https://software.intel.com/sites/landingpage/IntrinsicsGuide'
                '/files/data-latest.xml\n' 'By default, this tool will look in '
                'the current directory for data-latest.xml.' % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    curses.wrapper(main, intr_data)
