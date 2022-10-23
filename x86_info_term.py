#!/usr/bin/env python3
#
# x86-info-term
# Copyright (c) 2020 Zach Wegner, Travis Downs
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import curses
import gzip
import json
import os
import re
import sys
import urllib.request
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
Dataset = DummyObj

def clip(value, lo, hi):
    return max(lo, min(value, hi - 1))

################################################################################
## Key bindings ################################################################
################################################################################

def CTRL(k):
    return chr(ord(k) & 0x1f)
ESC = CTRL('[')
# Key map. Screen-size dependent values are lambdas
SCROLL_KEYS = {
    CTRL('E'):  +1,
    CTRL('Y'):  -1,
    CTRL('D'):  lambda r, c: +(r // 2),
    CTRL('U'):  lambda r, c: -(r // 2),
    'KEY_NPAGE':lambda r, c: +r,
    'KEY_PPAGE':lambda r, c: -r,
}
CURS_KEYS = {
    'd':        lambda r, c: +(r // 2),
    'u':        lambda r, c: -(r // 2),
    'g':        -1e10,
    'G':        +1e10,
    'KEY_HOME': -1e10,
    'KEY_END':  +1e10,
    'j':        +1,
    'KEY_DOWN': +1,
    'J':        +10,
    'k':        -1,
    'KEY_UP':   -1,
    'K':        -10,
}

# Mode-specific keys
CMD_KEYS = {
    Mode.BROWSE: {
        'q': 'quit',
        CTRL('C'): 'quit',

        'f': 'start-filter',
        '/': 'start-filter',
        '\t': 'start-filter',

        ' ': 'toggle-fold',
        '\n': 'toggle-fold',
        'o': 'open-fold',
        'O': 'open-all-folds',
        'c': 'close-fold',
        'C': 'close-all-folds',

        's': 'switch-data-source',

        'h':        'scroll-left',
        'KEY_LEFT': 'scroll-left',
        'l':        'scroll-right',
        'KEY_RIGHT':'scroll-right',
    },
    Mode.FILTER: {
        'KEY_LEFT':  'cursor-left',
        'KEY_RIGHT': 'cursor-right',
        CTRL('B'):   'cursor-left',
        CTRL('F'):   'cursor-right',
        'KEY_UP':    'cursor-home',
        'KEY_DOWN':  'cursor-end',
        'KEY_HOME':  'cursor-home',
        'KEY_END':   'cursor-end',

        # curses' window.getkey() doesn't have a portable return value for
        # backspace, apparently (see https://stackoverflow.com/a/54303430)
        'KEY_BACKSPACE': 'backspace',
        '\x7f':          'backspace',
        '\b':            'backspace',

        'KEY_DC':  'delete',
        CTRL('W'): 'kill-word-back',
        CTRL('K'): 'kill-line-fwd',
        CTRL('U'): 'kill-line-back',

        ESC: 'clear-filter',

        '\n': 'start-browse',
        '\t': 'start-browse',
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

# AStr is a wrapper for strings keeping attributes on ranges of characters
# This is probably overengineering but its fun+pretty, so whatever
class AStr:
    def __init__(self, value, attrs=None):
        self.value = value
        if attrs is None:
            attrs = [(0, 'default')]
        elif isinstance(attrs, str):
            attrs = [(0, attrs)]
        self.attrs = attrs
    def offset_attrs(self, delta):
        attrs = [(offset + delta, attr) for offset, attr in self.attrs]
        # Chop off negative entries, unless they cover the start
        start = 0
        for i, [offset, attr] in enumerate(self.attrs):
            if offset > 0:
                break
            start = i
        return attrs[start:]

    def __add__(self, other):
        if not isinstance(other, AStr):
            other = AStr(other)
        attrs = self.attrs + other.offset_attrs(len(self.value))
        return AStr(self.value + other.value, attrs)
    def __radd__(self, other):
        return AStr(other) + self
    def __getitem__(self, s):
        assert isinstance(s, slice)
        assert s.step == 1 or s.step is None
        attrs = self.attrs
        if s.start:
            # Convert negative indices to positive so offset_attrs() works
            if s.start < 0:
                s = slice(max(0, len(self.value)+s.start), s.stop, s.step)
            attrs = self.offset_attrs(-s.start)
        if s.stop:
            attrs = [(offset, attr) for offset, attr in attrs if offset < s.stop]
        return AStr(self.value[s], attrs=attrs)
    def __len__(self):
        return len(self.value)

    # Hacky reimplementations of str methods
    def splitlines(self):
        while '\n' in self.value:
            index = self.value.find('\n')
            line, self = self[:index], self[index+1:]
            yield line
        yield self
    def rfind(self, *args):
        return self.value.rfind(*args)
    def strip(self):
        # This is dumb+inefficient
        sub = self.value.lstrip()
        start = len(self) - len(sub)
        return self[start:start + len(sub.rstrip())]
    def lstrip(self):
        sub = self.value.lstrip()
        attrs = self.offset_attrs(len(sub) - len(self))
        return AStr(sub, attrs=attrs)
    def replace(self, pat, sub):
        result = AStr('')
        while pat in self.value:
            index = self.value.find(pat)
            result = self.value[:index] + sub
            self = self[index + 1:]
        return result + self

# Like str.join but for AStr
def a_join(sep, args):
    if not args: return ''
    result = args[0]
    for arg in args[1:]:
        result = result + sep + arg
    return result

def pad(s, width, right=False):
    # Pad and truncate
    if right:
        return (' ' * width + s)[-width:]
    else:
        return (s + ' ' * width)[:width]

def wrap_lines(cell, width, indent=0):
    cell = cell.replace('\t', ' ' * 4)
    indent = ' ' * indent
    prefix = ''
    for line in cell.splitlines():
        while len(line) > width:
            split = line.rfind(' ', 0, width)
            if split == -1:
                split = width
            [chunk, line] = line[:split], line[split:].lstrip()
            yield prefix + chunk
            prefix = indent
        yield prefix + line

def get_col_width(table, col):
    width = 0
    for row in table:
        if isinstance(row, dict):
            if row.get('span', False):
                continue
            cells = row['cells'] 
        else:
            cells = row
        cell = cells[col]
        if not isinstance(cell, (str, AStr)):
            [cell, info] = cell
        width = max(width, len(cell))
    return width + 1

################################################################################
## Intrinsic info ##############################################################
################################################################################

def parse_intrinsics_guide(path):
    root = ET.parse(path)

    version = root.getroot().attrib['version']
    version = tuple(int(x) for x in version.split('.'))

    table = []
    for i, intrinsic in enumerate(root.findall('intrinsic')):
        try:
            tech = intrinsic.attrib['tech']
            name = intrinsic.attrib['name']
            desc = [d.text for d in intrinsic.findall('description')][0]
            insts = [(inst.attrib['name'].lower(), inst.attrib.get('form', ''))
                    for inst in intrinsic.findall('instruction')]
            # Return type spec changed in XML as of 3.5.0
            return_type = (intrinsic.attrib['rettype'] if version < (3, 5, 0) else
                    [r.attrib['type'] for r in intrinsic.findall('return')][0])
            key = '%s %s %s %s' % (tech, name, desc, insts)
            table.append({
                'id': i,
                'tech': tech,
                'name': name,
                'params': [(p.attrib.get('varname', ''), p.attrib['type'])
                    for p in intrinsic.findall('parameter')],
                'return_type': return_type,
                'desc': desc,
                'operations': [op.text for op in intrinsic.findall('operation')],
                'insts': insts,
                'search-key': key.lower(),
            })
        except:
            print('Error while parsing %s:' % name)
            print(ET.tostring(intrinsic, encoding='unicode'))
            raise

    return [version, table]

def get_intr_subtable(intr):
    blank_row = ['', '']
    inst_rows = [['Instruction' if i == 0 else '', AStr(n, 'inst') + ' %s' % (f)]
            for [i, [n, f]] in enumerate(intr['insts'])]
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

def get_intr_table(ctx, start, stop, folds={}):
    rows = []
    prev_tech = ''
    for i, intr in enumerate(ctx.filtered_data[start:stop]):
        expand = (intr['id'] in folds)
        tech = intr['tech']

        # Avoid repeatedly printing the same ISA set name, but always mention it
        # for expanded entries
        show_tech = expand or tech != prev_tech
        prev_tech = tech
        shown_tech = tech if show_tech else ''

        # Look up the row in the cache. We hackily replace the ID
        # in the cache because that changes all the time
        cache_key = (intr['id'], expand, show_tech)
        if cache_key in ctx.intr_table_cache:
            row = ctx.intr_table_cache[cache_key]
            row['id'] = i + start
            rows.append(row)
            continue

        params = a_join(', ', [AStr(type, 'type') + (' ' + param)
                for param, type in intr['params']])
        decl = AStr(intr['name'], 'bold') + '(' + params.strip() + ')'

        # If this intrinsic is unfolded, show intrinsic detail (description,
        # pseudocode, etc.) as well as uops.info performance data
        if expand:
            subtables = [get_intr_subtable(intr)]

            for [mnem, form] in intr['insts']:
                if mnem in ctx.uops_info:
                    uop_forms = get_intr_uop_matches(ctx, mnem, form)
                    subtables.append(get_uop_subtable(ctx, ctx.uops_info[mnem],
                            uop_forms=uop_forms))
        else:
            subtables = []

        row = {
            'id': i + start,
            'cells': [
                ['', {'attr': tech}],
                [shown_tech, {}],
                # Add padding on right because the return type column is right-aligned
                AStr(intr['return_type'] + ' ', 'type'),
                [decl, {'wrap': True, 'indent': 4}],
            ],
            'subtables': subtables,
        }
        ctx.intr_table_cache[cache_key] = row
        rows.append(row)

    widths = [2, 12, -1, 0]
    return Table(rows=rows, widths=widths, alignment=[0, 0, 1, 0])

################################################################################
## Info from uops.info #########################################################
################################################################################

# All architectures measured by uops.info. This is just here for consistent
# ordering
ALL_ARCHES = ['CON', 'WOL', 'NHM', 'WSM', 'SNB', 'IVB', 'HSW', 'BDW', 'SKL',
        'SKX', 'KBL', 'CFL', 'CNL', 'ICL', 'ZEN+', 'ZEN2']

# Sentinel value for unknown latency
MAX_LATENCY = 1e100

def parse_uops_info(path):
    root = ET.parse(path)

    version = root.getroot().attrib['date']

    uops_info = {}
    for ext in root.findall('extension'):
        extension = ext.attrib['name']
        for inst in ext.findall('instruction'):
            mnem = inst.attrib['asm'].lower()
            form = inst.attrib['string'].lower()
            arch_info = {}
            for arch in inst.findall('architecture'):
                arch_name = arch.attrib['name']
                for meas in arch.findall('measurement'):
                    ports = meas.attrib.get('ports', '')
                    ports = re.sub(r'\b1\*', '', ports)
                    if 'TP' in meas.attrib:
                        tp = meas.attrib['TP']
                    else:
                        tp = meas.attrib['TP_unrolled']

                    # Look through every operand->result latency measurement,
                    # and get the min/max. Each of min/max can be an upper
                    # bound, meaning the measurement method can't guarantee the
                    # "true" minimum latency, and it might actually be lower.
                    # We store these as (latency, is_exact) tuples, which sort
                    # in the right way to get the overall min/max.
                    lat_min = (MAX_LATENCY, True)
                    lat_max = (0, False)
                    for lat in meas.findall('latency'):
                        for attr, value in lat.attrib.items():
                            if 'upper_bound' in attr:
                                assert value == '1'
                            elif 'cycles' in attr:
                                is_exact = (attr + '_is_upper_bound') not in lat.attrib
                                latency = (int(value), is_exact)
                                lat_min = min(lat_min, latency)
                                lat_max = max(lat_max, latency)
                            else:
                                assert attr in ('start_op', 'target_op')
                    arch_info[arch_name] = (ports, tp, (lat_min, lat_max))
            if not arch_info:
                continue

            # Strip out extra uops specifiers, like "lock", "{store}", etc.
            if ' ' in mnem:
                [prefix, mnem] = mnem.rsplit(None, 1)
                form = prefix + ' ' + form

            # Add a dict to hold all forms of this mnemonic
            # XXX We store the first extension here to use for sorting, though
            # not all forms have the same extension
            if mnem not in uops_info:
                uops_info[mnem] = {
                    'id': len(uops_info),
                    'mnem': mnem,
                    'extension': extension,
                    'forms': []
                }

            uops_info[mnem]['forms'].append({
                'form': form,
                'extension': extension,
                'search-key': (form + ' ' + extension).lower(),
                'arch': arch_info
            })

    # Update the search key for each instruction with all the forms
    for [mnem, uop] in uops_info.items():
        uop['search-key'] = ' '.join(f['search-key'] for f in uop['forms'])

    return [version, uops_info]

def get_uop_table(ctx, start, stop, folds={}):
    rows = []
    prev_ext = ''
    for [i, uop] in enumerate(ctx.filtered_data[start:stop]):
        expand = (uop['id'] in folds)
        subtables = [get_uop_subtable(ctx, uop)] if expand else []

        ext = uop['extension']
        # Hacky: get a color for this instruction by matching the longest
        # prefix of the extension that's also an intrinsic extension
        color = 'Other'
        for prefix in range(len(ext), 0, -1):
            if ext[:prefix] in INTR_COLORS:
                color = ext[:prefix]
                break

        # Make a clean-ish description line with all the instruction forms
        forms = ';  '.join(re.sub(r'.*\((.*)\)', r'\1', form['form'])
                for form in uop['forms'])

        row = {
            'id': i + start,
            'cells': [
                ['',  {'attr': color}],
                [ext if expand or prev_ext != ext else '', {}],
                # Pad mnemonic on left
                [' ' + uop['mnem'], {'attr': 'bold'}],
                forms,
            ],
            'subtables': subtables,
        }
        prev_ext = ext
        rows.append(row)

    widths = [2, 12, -1, 0]
    return Table(rows=rows, widths=widths, alignment=[0, 0, 0, 0])

def get_uop_subtable(ctx, uop, uop_forms=None):
    # Get the union of all arches in each form for consistent columns. We sort
    # by the entries in ALL_ARCHES, but add any extra arches at the end for
    # future proofing
    seen_arches = {arch for form in uop['forms'] for arch in form['arch']}
    arches = [a for a in ALL_ARCHES if a in seen_arches] + list(seen_arches - set(ALL_ARCHES))
    if ctx.arches:
        arches = [a for a in arches if a.lower() in ctx.arches]

    if not arches:
        return None

    columns = len(arches) + 1
    blank_row = [''] * columns
    pad = ' ' * 4
    header = [AStr(arch, 'bold') for arch in arches]

    if uop_forms is None:
        uop_forms = uop['forms']

    rows = []
    for form in uop_forms:
        latencies = []
        throughputs = []
        port_usages = []
        # Create separate rows for latency/throughput/port usage
        for arch in arches:
            if arch in form['arch']:
                [ports, tp, lat_bounds] = form['arch'][arch]

                if lat_bounds[0][0] != MAX_LATENCY:
                    lat_bounds = ['%s%s' % (('≤' if not is_exact else ''), value)
                            for [value, is_exact] in lat_bounds]
                    [lat_min, lat_max] = lat_bounds
                    lat = lat_min if lat_min == lat_max else '%s;%s' % (lat_min, lat_max)
                    latencies.append(lat)
                else:
                    latencies.append('-')
                throughputs.append(str(tp))
                port_usages.append(str(ports))
            else:
                latencies.append('-')
                throughputs.append('-')
                port_usages.append('-')

        rows.extend([
            {'cells': [AStr(pad + form['form'], 'inst')], 'span': True},
            ['', *header],
            [AStr(pad + 'Ports:', 'bold'), *port_usages],
            [AStr(pad + 'Latency:', 'bold'), *latencies],
            [AStr(pad + 'Throughput:', 'bold'), *throughputs],
            blank_row,
        ])

    widths = [-1] * columns
    alignment = [False] * columns
    scroll = [True] * columns
    scroll[0] = False
    return Table(rows=rows, widths=widths, alignment=alignment, scroll=scroll)

################################################################################
## Intrinsic/uops.info unification #############################################
################################################################################

INTR_ARG_REMAP = {
    'vm32x': 'vsib_xmm',
    'vm32y': 'vsib_ymm',
    'vm32z': 'vsib_zmm',
    'vm64x': 'vsib_xmm',
    'vm64y': 'vsib_ymm',
    'vm64z': 'vsib_zmm',
    'mib':   'm192',
    'imm8':  'i8',
}

# Extra argument options: intrinsic register args can match memory args of the
# same size, but not vice versa--if an intrinsic has a memory arg, it's generally
# a load/store etc. and requires memory
INTR_ARG_EXTRA = {
    'r16': {'m16', 'i16'},
    'r32': {'m32', 'i32'},
    'r64': {'m64', 'i64'},
    'r8':  {'m8', 'i8'},
    'i8':  {'r8'}, # Special case for ror/rol reg, cl
    'xmm': {'m128'},
    'ymm': {'m256'},
    'zmm': {'m512'},
}

UOP_ARG_REMAP = {
    'al':        'r8',
    'ax':        'r16',
    'cl':        'r8',
    'dx':        'r16',
    'eax':       'r32',
    'rax':       'r64',
    'm32_1to2':  'm64',
    'm32_1to4':  'm128',
    'm32_1to8':  'm256',
    'm32_1to16': 'm512',
    'm64_1to2':  'm128',  
    'm64_1to4':  'm256',  
    'm64_1to8':  'm512', 
}

# Get a list of matching uop instruction forms for this instruction
def get_intr_uop_matches(ctx, mnem, target_form):
    matching_forms = []

    # Filter out some stuff and normalize
    target_form = (target_form.replace(' {z}', ', z').replace(' {k}', ', k')
            .replace(' {er}', '').replace(' {sae}', '').replace(' ', ''))

    # Create a set of matching arguments for each instruction argument
    intr_args = []
    for arg in target_form.split(','):
        arg = INTR_ARG_REMAP.get(arg, arg)
        # Add an extra option for register/memory matching
        arg_opts = {arg} | INTR_ARG_EXTRA.get(arg, set())
        intr_args.append(arg_opts)

    # Loop through all uop forms with the same mnemonic
    for form in ctx.uops_info[mnem]['forms']:
        inst, _, inst_form = form['form'].rstrip(')').partition(' (')
        # Mask zeroing variants in AVX-512 are indicated by a _z suffix. In
        # that case, replace any mask arguments 'k' with 'z', which we used
        # in the intrinsic replacements above to indicate the zeroing variant.
        if '_z' in inst:
            assert inst.endswith('_z')
            inst_form = inst_form.replace('k,', 'z,')

        uops_args = [UOP_ARG_REMAP.get(arg, arg) for arg in inst_form.split(', ')]

        # See if any form of the intrinsic matches this form. Note that zip()
        # only iterates as far as the shortest of the arguments, so it will
        # allow mismatched lengths as long as the prefix matches. This actually
        # works in our favor, for intrinsics like _mm256_cmpge_epi8_mask that
        # don't show an immediate in the instruction, since the immediate is
        # implied by the intrinsic (i.e. _MM_CMPINT_NLT). OK maybe this isn't
        # always beneficial but it will only lead to false positives...
        if all(arg in opts for [arg, opts] in zip(uops_args, intr_args)):
            matching_forms.append(form)

    return matching_forms

################################################################################
## Curses stuff ################################################################
################################################################################

# ANSI colors for different instruction sets. Approximately match colors in the
# Intel intrinsics guide, but avoid very saturated, bright, or dark colors.
INTR_COLORS = {
    'MMX':          185, #cccc33
    'SSE':          150, #99cc66
    'SSE2':         107, #669933
    'SSE3':         72,  #339966
    'SSSE3':        153, #99ccff
    'SSE4.1':       117, #66ccff
    'SSE4.2':       74,  #3399cc
    'AVX':          183, #cc99ff
    'AVX2':         134, #9933cc
    'FMA':          175, #cc6699
    'AVX_VNNI':     168, #cc3366
    'AVX-512':      173, #cc6633
    'KNC':          172, #cc6600
    'AMX':          172, #cc6600
    'SVML':         221, #ffcc33
    'Other':        244,
    # Plus some just for uops.info extensions
    'AVX512':       173,
    'SSE4':         117,
}

FILTER_PREFIX = 'Filter: '
CURS_OFFSET = len(FILTER_PREFIX)

# Draw a table on screen. This is fairly complicated for a number of reasons:
# * Line wrapping on individual cells, padding/truncating to width
# * Horizontal scrolling
# * Highlighting on substrings within cells
# * Highlighting the cursor row
# * Drawing subtables (this is a recursive function, but is just two deep now)
# * The first row can be partially scrolled off screen
# * Sometimes we don't actually render but just calculate layout (draw=False)
# * We limit drawing only to visible rows
# ...so it's a big hack, but it works
def draw_table(ctx, table, start_row, stop_row, curs_row_id=None, draw=True):
    # Fill in widths for auto-sizing columns
    table.widths = [get_col_width(table.rows, i) if width == -1 else width
            for [i, width] in enumerate(table.widths)]
    # Right column is shrunk or padded out to screen width
    if len(table.widths) > 1:
        table.widths[-1] = curses.COLS - sum(table.widths[:-1])
    if not hasattr(table, 'scroll'):
        table.scroll = [False] * len(table.widths)

    # Keep track of how many lines each row takes up
    screen_lines = {}

    # Draw the table
    current_row = start_row
    for row in table.rows:
        if current_row >= stop_row:
            break

        if not isinstance(row, dict):
            row = {'cells': row}
        row_id = row.get('id', -1)
        cells = row['cells']
        subtables = row.get('subtables', [])
        highlight = curses.A_REVERSE if row_id == curs_row_id else 0

        # Line-wrap and pad all cells. Do this in a separate loop to get the
        # maximum number of lines in a cell
        if not row.get('span'):
            wrapped_cells = []
            for width, alignment, scroll, cell in zip(table.widths, table.alignment,
                    table.scroll, cells):
                info = {}
                if not isinstance(cell, (str, AStr)):
                    [cell, info] = cell
                attr = ctx.attrs[info.get('attr', 'default')]
                if info.get('wrap', False):
                    cell_lines = wrap_lines(cell, width, indent=info.get('indent', 0))
                else:
                    cell_lines = [cell]

                cell_lines = [pad(cell, width, right=alignment) for cell in cell_lines]
                wrapped_cells.append((cell_lines, attr, width, scroll))
        # "Span" rows: one cell takes up all columns and ignores widths etc
        else:
            [cell] = cells
            width = curses.COLS
            cell_lines = list(wrap_lines(cell, width))
            wrapped_cells = [(cell_lines, 'default', width, False)]

        n_lines = max(len(c) for c, a, w, s in wrapped_cells)

        # Check for skipping rows (if the top row is partially scrolled off screen)
        if ctx.current_skip_rows:
            offset = min(ctx.current_skip_rows, n_lines)
            for cell_lines, _, _, _ in wrapped_cells:
                del cell_lines[:offset]
            ctx.current_skip_rows -= offset
            n_lines -= offset

        # Render lines
        col = 0
        skip_cols = ctx.skip_cols
        for cell_lines, attr, width, scroll in wrapped_cells:
            # Handle horizontal scrolling
            if skip_cols and scroll:
                if width > skip_cols:
                    cell_lines = [cell_line[skip_cols:] for cell_line in cell_lines]
                    width -= skip_cols
                    skip_cols = 0
                else:
                    skip_cols -= width
                    continue

            # Chop off the right side if the terminal isn't wide enough
            width = min(width, curses.COLS - col - 1)
            if width == 0:
                continue

            # Pad vertically
            cell_lines += [' ' * width] * (n_lines - len(cell_lines))
            wrap_line = current_row

            # Skip drawing the table on screen if requested (sometimes this is
            # called just for scrolling calculations)
            if not draw:
                continue

            for cell_line in cell_lines:
                if wrap_line >= stop_row:
                    break

                # Chop up ranges by attribute if this is an AStr
                if isinstance(cell_line, AStr):
                    sub_col = col
                    next_attrs = cell_line.attrs[1:] + [[None, None]]
                    for [[offset, attr], [next_offset, _]] in zip(cell_line.attrs, next_attrs):
                        # HACK: don't just reverse video for types, that looks awful
                        if highlight and attr == 'type':
                            attr = 'default'
                        attr = ctx.attrs[attr] | highlight

                        offset = max(0, offset)
                        next_offset = next_offset and max(0, next_offset)
                        part = cell_line.value[offset:next_offset]

                        ctx.window.insstr(wrap_line, sub_col, part, attr)
                        sub_col += len(part)
                        if sub_col - col >= width:
                            break
                else:
                    ctx.window.insstr(wrap_line, col, cell_line, attr | highlight)
                wrap_line += 1
            col += width

        next_row = current_row + n_lines

        # Render subtables if necessary
        for subtable in subtables:
            if subtable:
                next_row, _ = draw_table(ctx, subtable, next_row, stop_row,
                        curs_row_id=None, draw=draw)

        screen_lines[row_id] = next_row - current_row

        current_row = next_row

    return current_row, screen_lines

# If we don't know how big a row is on screen, we need to do a throwaway render
# to see how many lines it takes up. This really sucks for modularity and
# efficiency. Please forgive me
def get_n_screen_lines(ctx, row_id):
    one_row_table = get_data_table(ctx, row_id, row_id + 1, folds=ctx.folds)
    n_lines, _ = draw_table(ctx, one_row_table, 0, 1e100, draw=False)
    return n_lines

def scroll(ctx, offset, screen_lines, move_cursor=False):
    old_start_id = ctx.start_row_id
    # Scroll up
    if offset < 0:
        while offset < 0 and (ctx.start_row_id > 0 or ctx.skip_rows > 0):
            # If a row is partially displayed, scroll up within it
            if ctx.skip_rows > 0:
                off = min(ctx.skip_rows, -offset)
                ctx.skip_rows -= off
                offset += off
            # Otherwise, move up to the next row, and get its size
            else:
                assert ctx.start_row_id > 0
                ctx.start_row_id -= 1
                ctx.skip_rows = get_n_screen_lines(ctx, ctx.start_row_id) - 1
                offset += 1

        # Cursor moving here is complicated, handle it with a re-render

    # Scroll down
    elif offset > 0:
        while offset > 0:
            if ctx.start_row_id >= len(ctx.filtered_data):
                break

            # Get size in screen lines of next row
            if ctx.start_row_id in screen_lines:
                n_lines = screen_lines[ctx.start_row_id]
            else:
                n_lines = get_n_screen_lines(ctx, ctx.start_row_id)

            if n_lines > 0:
                off = min(n_lines - 1, offset)
                ctx.skip_rows += off
                offset -= off
            if offset > 0:
                if ctx.start_row_id >= len(ctx.filtered_data) - 1:
                    break
                ctx.start_row_id += 1
                ctx.skip_rows = 0
                offset -= 1

        # Move the cursor if necessary
        if move_cursor:
            ctx.curs_row_id = max(ctx.start_row_id, ctx.curs_row_id)

# For scrolling down to show a particular line, we use this hacky method in order
# to keep the UI snappy--scrolling down quickly to put the line on the top of the
# screen, seeing how many rows it is, then scrolling up as much as possible while
# keeping the row on-screen. This isn't perfect, but much quicker than the
# iterative scrolling down, which was very slow, especially with longer jumps
# and lots of open folds
def approx_scroll_to(ctx, row_id, screen_lines):
    n_lines = get_n_screen_lines(ctx, row_id)
    ctx.start_row_id = row_id
    ctx.skip_rows = 0
    # Check if there's enough room on the screen to scroll back up and keep the
    # whole selected entry visible, and scroll if so. We add a hacky offset
    # here to get around a corner case that can cause a hang during rendering:
    # because we dynamically reflow the widths of columns based on the widths
    # of visible cells, scrolling can change the number of screen lines a
    # particular row takes up. If the scrolling up changes the column widths
    # such that the selected row falls back off the bottom, we will endlessly
    # loop scrolling up and down. approx_scroll_offset counts the number of
    # unsuccessful scroll ups (resetting on a successful render), so we scroll
    # upwards less each time. This might mean the selected row isn't the
    # bottommost row, I'm not totally sure (this code is a bit complicated).
    # I *think* this always terminates, but now feels like a good time to remind
    # you of that whole "WITHOUT ANY WARRANTY" thing at the top of the file.
    scroll_lines = ctx.n_visible_lines - n_lines - ctx.approx_scroll_offset
    if scroll_lines > 0:
        scroll(ctx, -scroll_lines, screen_lines, move_cursor=False)
        ctx.approx_scroll_offset += 1

def update_filter(ctx):
    if ctx.filter:
        filter_list = ctx.filter.lower().split()
        # Helper to determine if an entry matches the filter
        def is_match(key):
            for f in filter_list:
                # Handle filter inversion with the '!' character
                invert = f.startswith('!')
                if invert:
                    f = f[1:]
                if not invert ^ bool(re.search(f, key)):
                    return False
            return True
        new_fd = []
        # Try filtering with input regexes. If the parse fails, keep the old
        # filtered data and annoy the user by flashing an error
        try:
            for item in ctx.data_source:
                if is_match(item['search-key']):
                    # For uops data, also filter the displayed forms
                    if ctx.show_uops:
                        item = item.copy()
                        item['forms'] = [form for form in item['forms']
                                if is_match(form['search-key'])]
                    new_fd.append(item)
            ctx.filtered_data = new_fd
        except re.error as e:
            ctx.flash_error = str(e)
        ctx.curs_row_id = 0
        ctx.start_row_id = 0
        ctx.skip_rows = 0
    else:
        ctx.filtered_data = ctx.data_source

def get_data_table(ctx, start, stop, folds={}):
    if not ctx.filtered_data:
        rows = [{'id': 0, 'cells': ['No results.']}]
        return Table(rows=rows, widths=[curses.COLS], alignment=[0])

    if ctx.show_uops:
        return get_uop_table(ctx, start, stop, folds=folds)
    else:
        return get_intr_table(ctx, start, stop, folds=folds)

def run_ui(stdscr, args, intr_data, uops_info):
    # Set up the color table, using the per-ISA table, and adding in other
    # syntax colors depending on the color scheme (light/dark)
    colors = {k: (curses.COLOR_BLACK, v) for k, v in INTR_COLORS.items()}
    # With use_default_colors, -1 denotes the default foreground/background
    curses.use_default_colors()
    fg, bg = (-1, -1)
    colors.update({
        'default':  (fg, bg),
        'type':     ( 2, bg),
        'inst':     ( 2, bg),
        'sep':      (15, 12, curses.A_BOLD),
        'error':    (fg,  9),
        'bold':     (fg, bg, curses.A_BOLD),
        'code':     (231, 237) if args.dark_mode else (0, 251),
    })
    # Create attributes
    attrs = {}
    for tech, (fg, bg, *extra) in colors.items():
        n = len(attrs) + 1
        curses.init_pair(n, fg, bg)
        attrs[tech] = curses.color_pair(n) + sum(extra, 0)

    curses.raw()
    curses.mousemask((1 << 32) - 1)

    # Make cursor invisible
    curses.curs_set(0)

    if args.show_uops:
        data_source = sorted(uops_info.values(), key=lambda u: (u['extension'], u['mnem'])) 
    else:
        data_source = intr_data

    # Create a big dummy object for passing around a bunch of random state
    ctx = Context(window=stdscr, mode=Mode.BROWSE, data_source=data_source,
            intr_data=intr_data, uops_info=uops_info, show_uops=args.show_uops,
            filter=args.filter, count='', filtered_data=[], flash_error=None,
            curs_row_id=0, start_row_id=0, skip_rows=0, skip_cols=0,
            attrs=attrs, folds=set(), move_flag=False,
            curs_col=len(args.filter) if args.filter else 0,
            arches=args.arch, intr_table_cache={}, n_visible_lines=0,
            approx_scroll_offset=0)

    update_filter(ctx)

    while True:
        # Clear screen
        ctx.window.clear()

        # Get a layout table of all filtered intrinsics. Narrow the range down
        # by the rows on screen so we can set dynamic column widths
        table = get_data_table(ctx, ctx.start_row_id,
                ctx.start_row_id + curses.LINES, folds=ctx.folds)

        # Draw status lines
        filter_line = None
        status_lines = []
        if ctx.flash_error is not None:
            status_lines.append((ctx.flash_error, 'error'))
            ctx.flash_error = None
        if ctx.count:
            count_line = 'Count: ' + ctx.count
            status_lines.append((count_line, 'default'))
        elif ctx.filter is not None:
            hl = 'bold' if ctx.mode == Mode.FILTER else 'default'
            filter_line = FILTER_PREFIX + ctx.filter
            status_lines.append((filter_line, hl))

        status = '%s/%s    ' % (ctx.curs_row_id + 1, len(ctx.filtered_data))
        status_lines = [(pad(status, curses.COLS, right=True), 'sep')] + status_lines

        ctx.n_visible_lines = curses.LINES - len(status_lines)

        for i, [line, attr] in enumerate(status_lines):
            row = ctx.n_visible_lines + i
            ctx.window.insstr(row, 0, pad(line, curses.COLS), ctx.attrs[attr])

        # Set a counter with the number of rows to skip in rendering (used for
        # scrolling rows partially off screen). We will subtract from this as
        # we render line by line
        ctx.current_skip_rows = ctx.skip_rows

        # Render the current rows, and calculate how many rows are showing on screen
        n_screen_lines, screen_lines = draw_table(ctx, table, 0,
                ctx.n_visible_lines, curs_row_id=ctx.curs_row_id)
        n_lines = len(screen_lines)

        # Check that the cursor is on screen. If not, we decide whether to move
        # the cursor or scroll based on the last action performed. Dealing with
        # this efficiently is complicated and annoying, so basically we just
        # allow rendering to fail, correct the cursor/scroll position, then
        # continue and re-render.
        if ctx.curs_row_id not in screen_lines:
            max_row = max(screen_lines)
            if ctx.curs_row_id > max_row:
                # If we moved last, scroll
                if ctx.move_flag:
                    approx_scroll_to(ctx, ctx.curs_row_id, screen_lines)
                # Otherwise, move the cursor
                else:
                    ctx.curs_row_id = max_row
                continue

            elif ctx.curs_row_id < ctx.start_row_id:
                # If we moved last, scroll. We can do this cheaply by starting
                # to render on the cursor row
                if ctx.move_flag:
                    ctx.start_row_id = ctx.curs_row_id
                    ctx.skip_rows = 0
                # Otherwise, move the cursor
                else:
                    ctx.curs_row_id = ctx.start_row_id
                continue

        # Alternatively, if we moved up to the top row on the screen, but it's not
        # entirely visible, scroll to the beginning of it
        elif (ctx.curs_row_id == ctx.start_row_id and
                ctx.move_flag and ctx.skip_rows > 0):
            ctx.skip_rows = 0
            continue

        ctx.move_flag = False
        ctx.approx_scroll_offset = 0

        # Show the cursor for filter mode
        if ctx.mode == Mode.FILTER and filter_line is not None:
            # Clip the cursor to the screen, and add an offset for the "Filter: " prefix
            col_max = min(len(ctx.filter) + 1, curses.COLS - CURS_OFFSET - 1)
            ctx.curs_col = clip(ctx.curs_col, 0, col_max)
            ctx.window.move(curses.LINES - 1, ctx.curs_col + CURS_OFFSET)
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

            # Reset count here always
            ctx.count = ''

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
                selection = ctx.filtered_data[ctx.curs_row_id]['id']
                if cmd == 'open-fold':
                    ctx.folds |= {selection}
                elif cmd == 'open-all-folds':
                    ctx.folds = set(range(len(ctx.data_source)))
                elif cmd == 'close-fold':
                    ctx.folds -= {selection}
                elif cmd == 'close-all-folds':
                    ctx.folds = set()
                elif cmd == 'toggle-fold':
                    ctx.folds ^= {selection}
                else:
                    assert False, cmd

            # Horizontal scrolling
            elif cmd == 'scroll-left':
                ctx.skip_cols = max(ctx.skip_cols - 10, 0)
            elif cmd == 'scroll-right':
                # Unbounded on the right
                ctx.skip_cols += 10

            # Input editing
            elif cmd == 'backspace':
                prefix = ctx.filter[:ctx.curs_col-1] if ctx.curs_col > 0 else ''
                ctx.filter = prefix + ctx.filter[ctx.curs_col:]
                ctx.curs_col -= 1
                update_filter(ctx)
            elif cmd == 'delete':
                ctx.filter = ctx.filter[:ctx.curs_col] + ctx.filter[ctx.curs_col+1:]
                update_filter(ctx)
            elif cmd == 'kill-word-back':
                # Find the first non-space going backwards, then first space
                kill_point = ctx.curs_col - 1
                while kill_point >= 0 and ctx.filter[kill_point] == ' ':
                    kill_point -= 1
                kill_point = ctx.filter.rfind(' ', 0, max(kill_point, 0))
                ctx.filter = ctx.filter[:kill_point+1] + ctx.filter[ctx.curs_col:]
                ctx.curs_col = kill_point + 1
                update_filter(ctx)
            elif cmd == 'kill-line-fwd':
                ctx.filter = ctx.filter[:ctx.curs_col]
                update_filter(ctx)
            elif cmd == 'kill-line-back':
                ctx.filter = ctx.filter[ctx.curs_col:]
                ctx.curs_col = 0
                update_filter(ctx)

            elif cmd == 'cursor-left':
                ctx.curs_col -= 1
            elif cmd == 'cursor-right':
                ctx.curs_col += 1
            elif cmd == 'cursor-home':
                ctx.curs_col = 0
            elif cmd == 'cursor-end':
                ctx.curs_col = len(ctx.filter)

            # Data source switching. Just swap sources and return True, flagging
            # that we want to re-run the UI
            elif cmd == 'switch-data-source':
                args.show_uops = not args.show_uops
                args.filter = ctx.filter
                return True

            elif cmd == 'quit':
                return

            else:
                assert False, cmd
        # Resize
        elif key == 'KEY_RESIZE':
            curses.update_lines_cols()
        # Count (for vim-like movement)
        elif ctx.mode == Mode.BROWSE and key.isdigit():
            ctx.count += key
        elif ctx.mode == Mode.BROWSE and key == ESC and ctx.count:
            ctx.count = ''
        # Filter text input
        elif ctx.mode == Mode.FILTER and key.isprintable() and len(key) == 1:
            ctx.filter = ctx.filter[:ctx.curs_col] + key + ctx.filter[ctx.curs_col:]
            ctx.curs_col += 1
            update_filter(ctx)
        # Mouse input
        elif key == 'KEY_MOUSE':
            [_, x, y, _, bstate] = curses.getmouse()
            if bstate & curses.BUTTON2_PRESSED:
                scroll(ctx, 3, screen_lines, move_cursor=True)
            elif bstate & curses.BUTTON4_PRESSED:
                scroll(ctx, -3, screen_lines, move_cursor=True)
        # Scroll
        elif key in SCROLL_KEYS:
            count = int(ctx.count) if ctx.count else 1
            ctx.count = ''
            offset = count * get_offset(SCROLL_KEYS, key)
            scroll(ctx, offset, screen_lines, move_cursor=True)
        # Move cursor
        elif key in CURS_KEYS:
            count = int(ctx.count) if ctx.count else 1
            # Override g/G behavior with count to go to an absolute line
            if key.lower() == 'g' and ctx.count:
                ctx.curs_row_id = count - 1
            else:
                ctx.curs_row_id += count * get_offset(CURS_KEYS, key)
            ctx.curs_row_id = clip(ctx.curs_row_id, 0, len(ctx.filtered_data))
            ctx.move_flag = True
            ctx.count = ''
        else:
            ctx.flash_error = 'Unknown key: %r' % key
            ctx.count = ''

################################################################################
## Data gathering helpers ######################################################
################################################################################

DATASETS = [
    Dataset(name='intrinsics',
        parse_fn=parse_intrinsics_guide,
        base_url='https://software.intel.com/content/dam/develop/public/us/en/include/intrinsics-guide',
        path='data-latest.xml'),
    Dataset(name='uops_info',
        parse_fn=parse_uops_info,
        base_url='https://www.uops.info',
        path='instructions.xml'),
]

def download_data(args):
    # Make sure output directory exists
    os.makedirs(args.data_dir, exist_ok=True)

    allow_dl = args.update or args.download

    for dataset in DATASETS:
        # Check if the XML file already exists, unless --update was passed
        if not args.update and os.path.exists(dataset.full_path):
            print('%s already exists, skipping download...' % dataset.path)
        else:
            if not allow_dl:
                print('Could not find data in the data directory (%s).' % args.data_dir)
                print('Do you want to download data from intel.com and uops.info to this directory?')
                print('A JSON cache will also be written to this directory to improve startup time.')
                print('If not, no files will be written and no network accesses will be performed.')
                resp = input('Download? [y/N] ')
                allow_dl = (resp.strip().lower() in {'yes', 'y'})
                if not allow_dl:
                    sys.exit(1)

            # Download file
            url = '%s/%s' % (dataset.base_url, dataset.path)
            print('Downloading %s...' % url)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as f:
                data = f.read()

            # Write output file
            with open(dataset.full_path, 'wb') as f:
                f.write(data)

# We store preprocessed data from the XML to have a faster startup time.
# The schema of the preprocessed data might change in the future, in
# which case we'll increment this so we know to regenerate the JSON.
JSON_CACHE_VERSION = 2

def get_info(args):
    # Set full path for each dataset
    for dataset in DATASETS:
        dataset.full_path = '%s/%s' % (args.data_dir, dataset.path)

    # Check for JSON cache (if not passed --update or --update-cache)
    json_path = '%s/cache.json.gz' % args.data_dir
    found_version = None
    cache = None
    if not args.update and not args.update_cache and os.path.exists(json_path):
        with gzip.open(json_path) as f:
            cache = json.load(f)

    # Download the XML data if necessary and build the cache
    if not cache or cache['cache_version'] != JSON_CACHE_VERSION:
        download_data(args)

        # Parse the XML data
        cache = {'cache_version': JSON_CACHE_VERSION, 'datasets': {}}
        for dataset in DATASETS:
            [version, data] = dataset.parse_fn(dataset.full_path)
            cache['datasets'][dataset.name] = {'version': version, 'data': data}

        # Write JSON cache
        print('Writing cache to %s...' % json_path)
        with gzip.open(json_path, 'wb') as f:
            # json.dump() writes as a string, not bytes, which gzip.open needs
            f.write(json.dumps(cache).encode('ascii'))

    return cache

def get_cache():
    base_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_dir = '%s/data' % base_dir

    # Command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('filter', nargs='*', help='set an optional initial filter')
    parser.add_argument('--light', action='store_false', dest='dark_mode',
            help='use a light color scheme')
    parser.add_argument('-u', '--uops', action='store_true', dest='show_uops',
            help='show performance data for all instructions from uops.info '
            'as the main view instead of intrinsics')
    parser.add_argument('--arch', action='append', help='filter uops data to show only '
            'architectures in this list. Can separate arches by commas.\n'
            'Known arches: %s' % ' '.join(ALL_ARCHES))

    group = parser.add_argument_group('data source options')
    group.add_argument('--download', action='store_true', help='download the '
            'necessary XML files from intel.com and uops.info into the data directory '
            'if they don\'t exist')
    group.add_argument('--update', action='store_true', help='force a '
            're-download of the XML files')
    group.add_argument('--update-cache', action='store_true', help='re-build '
            'the JSON cache from the already-downloaded XML files')
    group.add_argument('-p', '--data-dir', default=data_dir, help='where to '
            'store downloaded XML files, and the JSON cache generated from them')

    args = parser.parse_args()

    args.filter = ' '.join(args.filter) or None

    if args.arch:
        args.arch = sum((arch.lower().split(',') for arch in args.arch), [])

    args.data_dir = os.path.abspath(args.data_dir)
    return args, get_info(args)

def main():
    args, cache = get_cache()

    intr_data = cache['datasets']['intrinsics']['data']
    uops_info = cache['datasets']['uops_info']['data']

    # Set a 50ms timeout on parsing escape sequences. This has to be done with
    # an environment variable (?!) because curses and its Python interface are awful.
    # We don't overwrite the variable if the user has already set it.
    if 'ESCDELAY' not in os.environ:
        os.environ['ESCDELAY'] = '50'

    # Run the UI within a loop. This is just a hacky way to easily support data
    # source switching--when the user requests a switch, we just change some
    # options and restart. We use curses.wrapper to make sure to clean up the
    # terminal on exit, even with exceptions etc
    while True:
        should_rerun = curses.wrapper(run_ui, args, intr_data, uops_info)
        if not should_rerun:
            break
    # Make sure cursor is visible back in the terminal
    curses.curs_set(1)

if __name__ == '__main__':
    main()
