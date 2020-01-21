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
    return max(lo, min(value, hi))

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
    CTRL('E'): +1,
    CTRL('Y'): -1,
    CTRL('D'): lambda r, c: +(r // 2),
    CTRL('U'): lambda r, c: -(r // 2),
}
CURS_KEYS = {
    'd': lambda r, c: +(r // 2),
    'u': lambda r, c: -(r // 2),
    'j': +1,
    'J': +10,
    CTRL('J'): +10,
    'k': -1,
    'K': -10,
    CTRL('K'): -10,
}

# Mode-specific keys
BROWSE_CMD_KEYS = {
    'q': 'quit',
    'f': 'start-filter',
    '/': 'start-filter',
}
FILTER_CMD_KEYS = {
    ESC: 'clear-filter',
    DEL: 'backspace',
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
    })
    # Create attributes
    attrs = {}
    for tech, (fg, bg) in colors.items():
        n = len(attrs) + 1
        curses.init_pair(n, fg, bg)
        attrs[tech] = curses.color_pair(n)

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

    COLS = curses.COLS
    ROWS = curses.LINES - 1

    start_row = 0
    curs_row = 0

    while True:
        # Clear screen
        win.clear()

        row = 0

        # Gather table data
        table = []
        for i, intr in enumerate(filtered_data):
            if i < start_row:
                continue
            if i > start_row + ROWS:
                break
            params = ', '.join('%s %s' % (type, param) for param, type in intr['params'])
            tech = intr['tech']
            table.append((i, [
                [tech, attrs[tech]],
                # HACK: pad on both sides
                [' %s ' % intr['return_type']],
                ['%s(%s)' % (intr['name'], params)],
            ]))

        if table:
            col_widths = [12, max(len(row[1][0]) for row_id, row in table)]
            col_widths += [COLS - sum(col_widths)]
            col_align_r = [0, 1, 0]
        else:
            table = [[0, [['No results.']]]]
            col_widths = [COLS]
            col_align_r = [0]

        # Draw the table
        line = 0
        for i, row in table:
            hl = curses.A_REVERSE if curs_row == i else 0
            col = 0
            for width, align_r, [cell, *attr] in zip(col_widths, col_align_r, row):
                attr = attr[0] if attr else attrs['default']
                cell = pad(cell, width, right=align_r)
                win.insstr(line, col, cell, attr | hl)
                col += width
            line += 1
            if line > ROWS:
                break

        # Draw status lines
        status_lines = []
        if flash_error is not None:
            status_lines.append((flash_error, 'error'))
        if filter is not None:
            status_lines.append(('Filter: %s' % filter, 'default'))

        if status_lines:
            status_lines = status_lines[::-1] + [('', 'sep')]
            for i, (line, attr) in enumerate(status_lines):
                win.insstr(ROWS - i, 0, pad(line, COLS), attrs[attr])

        win.refresh()

        key = win.getkey()
        flash_error = None
        # Scroll
        if mode == Mode.BROWSE and key in SCROLL_KEYS:
            start_row += get_offset(SCROLL_KEYS, key)
            start_row = clip(start_row, 0, len(filtered_data) - 1)
            curs_row = clip(curs_row, start_row, start_row + ROWS)
        # Move cursor
        elif mode == Mode.BROWSE and key in CURS_KEYS:
            curs_row += get_offset(CURS_KEYS, key)
            curs_row = clip(curs_row, 0, len(filtered_data) - 1)
            start_row = clip(start_row, curs_row - ROWS, curs_row)
        # Browse commands
        elif mode == Mode.BROWSE and key in BROWSE_CMD_KEYS:
            cmd = BROWSE_CMD_KEYS[key]
            if cmd == 'start-filter':
                mode = Mode.FILTER
                filter = ''
            elif cmd == 'quit':
                return
            else:
                assert False, cmd
        # Filter commands
        elif mode == Mode.FILTER and key in FILTER_CMD_KEYS:
            cmd = FILTER_CMD_KEYS[key]
            if cmd == 'clear-filter':
                filter = None
                update_filter()
                mode = Mode.BROWSE
            elif cmd == 'backspace':
                filter = filter[:-1]
                update_filter()
        # Filter text input
        elif mode == Mode.FILTER and key.isprintable():
            filter += key
            update_filter()
        else:
            flash_error = 'Unknown key: %r' % key

if __name__ == '__main__':
    curses.wrapper(main)
