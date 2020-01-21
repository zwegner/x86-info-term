import curses
import re
import sys
import xml.etree.ElementTree as ET
from enum import Enum, auto

def parse_intrinsics_guide(path):
    try:
        root = ET.parse(path)
    except FileNotFoundError:
        print('To use this, you must download the intrinsic XML data from:\n'
                'https://software.intel.com/sites/landingpage/IntrinsicsGuide'
                '/files/data-3.4.6.xml', file=sys.stderr)
        sys.exit(1)

    table = []
    for intrinsic in root.findall('intrinsic'):
        tech = intrinsic.attrib['tech']
        name = intrinsic.attrib['name']
        desc = [d.text for d in intrinsic.findall('description')][0]
        insts = [(inst.attrib['name'], inst.attrib.get('form', ''))
                for inst in intrinsic.findall('instruction')]
        key = '%s %s %s %s' % (tech, name, desc, ' '.join(n for n, f in insts))
        table.append({
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

def clip(value, lo, hi):
    return max(lo, min(value, hi - 1))

# Key map helpers
def CTRL(k):
    return chr(ord(k) & 0x1f)
ESC = CTRL('[')
DEL = '\x7f'

class Mode(Enum):
    BROWSE = auto()
    FILTER = auto()

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

def get_intr_info_table(intr):
    blank_row = ['', '']
    inst_rows = [['Instruction' if i == 0 else '', inst]
            for i, inst in enumerate(intr['insts'])]
    op_rows = [['Operation', [op, {'attr': 'code', 'wrap': True}]] for op in intr['operations']]
    table = [
        blank_row,
        ['Description', [intr['desc'], {'wrap': True}]],
        blank_row,
        *inst_rows,
        blank_row,
        *op_rows,
        blank_row,
    ]
    # Clean up title column
    for row in table:
        if row[0]:
            row[0] = ['%s:    ' % row[0], {'attr': 'bold'}]
    col_widths = [20, 0]
    col_align_r = [1, 0]
    return [table, col_widths, col_align_r]

def get_intr_table(intrinsics, start, stop):
    # Gather table data
    table = []
    for i, intr in enumerate(intrinsics[start:stop]):
        params = ', '.join('%s %s' % (type, param) for param, type in intr['params'])
        tech = intr['tech']
        table.append({
            'id': i + start,
            'cells': [
                [tech, {'attr': tech}],
                # HACK: pad on both sides
                ' %s ' % intr['return_type'],
                '%s(%s)' % (intr['name'], params),
            ],
            'subtables': [get_intr_info_table(intr)],
        })

    if table:
        col_widths = [12, get_col_width(table, 1), 0]
        col_align_r = [0, 1, 0]
    else:
        table = [{'id': 0, 'cells': ['No results.']}]
        col_widths = [curses.COLS]
        col_align_r = [0]
    return [table, col_widths, col_align_r]

def draw_table(win, table, col_widths, col_align_r, start_row, stop_row,
        curs_row=None, attrs=None):

    # Right column is shrunk or padded out to screen width
    if len(col_widths) > 1:
        col_widths[-1] = curses.COLS - sum(col_widths[:-1])

    # Draw the table
    line = start_row
    for row in table:
        if line >= stop_row:
            break

        if not isinstance(row, dict):
            row = {'cells': row}
        row_id = row.get('id', -1)
        cells = row['cells']
        subtables = row.get('subtables', [])

        hl = curses.A_REVERSE if curs_row == row_id else 0
        col = 0
        # Keep track of longest cell in lines (from line wrapping)
        # Yeah, we support wrapping in multiple columns.
        next_line = line
        for width, align_r, cell in zip(col_widths, col_align_r, cells):
            info = {}
            if not isinstance(cell, str):
                [cell, info] = cell
            attr = attrs[info.get('attr', 'default')]
            if info.get('wrap', False):
                cell_lines = wrap_lines(cell, width)
            else:
                cell_lines = [cell]

            wrap_line = line
            for cell in cell_lines:
                if wrap_line >= stop_row:
                    break
                cell = pad(cell, width, right=align_r)
                win.insstr(wrap_line, col, cell, attr | hl)
                wrap_line += 1
            next_line = max(next_line, wrap_line)
            col += width
        line = next_line

        # Render subtables if necessary
        for [st, cw, ca] in subtables:
            line = draw_table(win, st, cw, ca, line, stop_row, attrs=attrs)

    return line

def main(stdscr):
    intr_data = parse_intrinsics_guide('../data.xml')

    intr_colors = {
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
    colors = {k: (curses.COLOR_BLACK, v) for k, v in intr_colors.items()}
    colors.update({
        'default':  (15, curses.COLOR_BLACK),
        'sep':      (curses.COLOR_BLACK, 12),
        'error':    (15, 196),
        'bold':     (15, curses.COLOR_BLACK, curses.A_BOLD),
        'code':     (15, 237),
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
    win = stdscr

    def update_filter():
        nonlocal filter, filtered_data, flash_error, curs_row, start_row
        if filter is not None:
            filter_list = filter.lower().split()
            new_fd = []
            # Try filtering with input regexes. If the parse fails, keep the old
            # filtered data and annoy the user by flashing an error
            try:
                for intr in intr_data:
                    for f in filter_list:
                        if re.search(f, intr['search-key']) is None:
                            break
                    else:
                        new_fd.append(intr)
                filtered_data = new_fd
            except re.error as e:
                flash_error = str(e)
            curs_row = 0
            start_row = 0
        else:
            filtered_data = intr_data

    mode = Mode.BROWSE

    flash_error = None

    filter = None
    filtered_data = []
    update_filter()

    start_row = 0
    curs_row = 0

    while True:
        # Clear screen
        win.clear()

        # Get a layout table of all filtered intrinsics. Narrow the range down
        # by the rows on screen so we can set dynamic column widths
        [table, col_widths, col_align_r] = get_intr_table(filtered_data,
                start_row, start_row + curses.LINES)

        # Draw status lines
        status_lines = []
        if flash_error is not None:
            status_lines.append((flash_error, 'error'))
            flash_error = None
        if filter is not None:
            status_lines.append(('Filter: %s' % filter, 'default'))

        if status_lines:
            status_lines = status_lines[::-1] + [('', 'sep')]
            for i, (line, attr) in enumerate(status_lines):
                win.insstr(curses.LINES - i - 1, 0, pad(line, curses.COLS), attrs[attr])

        draw_table(win, table, col_widths, col_align_r,
                0, curses.LINES - len(status_lines),
                curs_row=curs_row, attrs=attrs)

        # Draw the window
        win.refresh()

        # Get input
        try:
            key = win.getkey()
        except curses.error:
            flash_error = 'Got no key...?'
            continue
        # Scroll
        if mode == Mode.BROWSE and key in SCROLL_KEYS:
            start_row += get_offset(SCROLL_KEYS, key)
            start_row = clip(start_row, 0, len(filtered_data))
            curs_row = clip(curs_row, start_row, start_row + curses.LINES)
        # Move cursor
        elif mode == Mode.BROWSE and key in CURS_KEYS:
            curs_row += get_offset(CURS_KEYS, key)
            curs_row = clip(curs_row, 0, len(filtered_data))
            start_row = clip(start_row, curs_row - curses.LINES, curs_row + 1)
        # Mode-specific commands
        elif key in CMD_KEYS[mode]:
            cmd = CMD_KEYS[mode][key]
            if cmd == 'start-filter':
                mode = Mode.FILTER
                filter = filter or ''
            elif cmd == 'start-browse':
                mode = Mode.BROWSE
            elif cmd == 'clear-filter':
                filter = None
                update_filter()
                mode = Mode.BROWSE
            elif cmd == 'backspace':
                filter = filter[:-1]
                update_filter()
            elif cmd == 'quit':
                return
            else:
                assert False, cmd
        elif key == 'KEY_RESIZE':
            curses.update_lines_cols()
        # Filter text input
        elif mode == Mode.FILTER and key.isprintable():
            filter += key
            update_filter()
        else:
            flash_error = 'Unknown key: %r' % key

if __name__ == '__main__':
    curses.wrapper(main)
