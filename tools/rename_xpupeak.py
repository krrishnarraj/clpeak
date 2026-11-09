#!/usr/bin/env python3
"""The 4.0 rename sweep: clpeak -> xpupeak, XpuPeak where case shows, in every
name only the tree sees.

    tools/rename_xpupeak.py            rename, git-mv, then print the residue
    tools/rename_xpupeak.py --dry-run  say what would change, touch nothing
    tools/rename_xpupeak.py --check    exit 1 if anything is left to rename
    tools/rename_xpupeak.py --report   the residue report only

Exits 2 when the report finds a mention that is neither renamed nor protected.

Over tracked text files it gives each spelling of the name its 4.0 one.  The
name is XpuPeak wherever case can show, and xpupeak wherever an identifier is
lowercase by rule or by convention (RENAME_PLAN.md, D11):

    CLPEAK                        -> XPUPEAK   macros, CMake options, env vars
    Clpeak, Xpupeak               -> XpuPeak   types: XpuPeakEventCallback
    clpeak, xpupeak + a capital   -> xpuPeak   camelCase: xpuPeakLaunch
    clpeak, xpupeak, bare, in a comment or in Markdown text
                                  -> XpuPeak   the product named in prose
    clpeak anywhere else          -> xpupeak   commands, files, snake, kebab

Bare leaves out the name in code format or in quotes, the name as part of a
path, a file name, a URL, a namespace or a longer word, and a command line
(xpupeak --list-devices): those keep their spelling.  So prose that means
the command, a binary or a target puts it in code format, as D11 asks:
`xpupeak`.  Strings are code here: a message that names the product says
XpuPeak by hand.  The mapping is case-exact, so clPeak (the OpenCL backend's
class) never matches.  Tracked paths that carry the name are git-mv'd the
same way.

Some names have someone on the other end: result files and history already
on users' disks, store and platform IDs, URLs, icons.  Each of those belongs
to a later task (RENAME_PLAN.md, T2-T6), so the sweep never touches the
protected paths and tokens below -- except the uppercase spelling.  CLPEAK
only ever names a CMake option or variable, a macro or an env var, and 4.0
keeps no 3.x spelling of any of them (the clpeak command is the one name
that stays), so it is renamed in the protected paths too, all but the saved
runs.  A name can be both an internal one and a
persisted key (clpeak_version is the C function and also the run document's
key).  The sweep renames the name, keeps the key, and the report lists each
such pair.

The script is idempotent.  After merging a 3.x fix into 4.0.x, a re-run
renames only what the merge brought in.  It ends with a residue report:
every clpeak still in the tree, grouped by the task that owns it.

A name renamed from clpeak grows by one character (from xpupeak it keeps its
length), so in C-family sources the sweep also re-aligns what was aligned to
a column after one: continuation lines under an open bracket, trailing
comments aligned with a neighbour's, and macro continuation backslashes.

Delete this script once 4.0.0 ships.
"""

import argparse
import bisect
import os
import re
import subprocess
import sys
import textwrap
from collections import OrderedDict, defaultdict

SELF = 'tools/rename_xpupeak.py'

# The spellings the sweep looks at: the three 3.x ones, and the two 4.0 ones
# that a case can still change (spelling() decides).  XPUPEAK is final.
TOKEN = re.compile(r'clpeak|Clpeak|CLPEAK|xpupeak|Xpupeak')
OLD = re.compile(r'clpeak|Clpeak|CLPEAK')    # the 3.x spellings
ANY = re.compile(r'clpeak', re.IGNORECASE)   # what `git grep -i` sees

OWNERS = OrderedDict([
    ('T2', 'xpupeak CLI and the clpeak command'),
    ('T3', 'result format v4 and data on disk'),
    ('T4', 'app identity'),
    ('T5', 'packaging and stores'),
    ('T6', 'docs, site, README'),
    ('stays', 'stays clpeak'),
])

# Paths that belong whole to another task: never rewritten.  First match
# wins, so a file comes before its directory.
PROTECTED_PATHS = [
    ('results/', 'stays', 'saved reference runs (data; the format redesign converts them)'),
    ('docs/format-v4.md', 'stays', 'the format doc, whose changes from v3 name the 3.x keys'),
    ('docs/', 'T6', 'the site'),
    ('README.md', 'T6', 'README'),
    ('.github/ISSUE_TEMPLATE/', 'T6', 'issue templates'),
    ('snap/', 'T5', 'snap packaging'),
    ('packaging/flatpak/', 'T5', 'Flatpak packaging'),
    ('packaging/homebrew/', 'T5', 'Homebrew formula'),
    ('packaging/signpath/', 'T2', 'Windows signing configuration'),
    ('tools/icons/', 'T4', 'icon master and generator'),
    ('app/android/privacy.txt', 'T4', 'Play privacy text'),
    (SELF, 'stays', 'this script (deleted once 4.0.0 ships)'),
]

# The protected paths whose uppercase CLPEAK stays too: data, and this script.
UPPER_KEPT = ('results/', SELF)
UPPER = re.compile(r'CLPEAK')

# Protected tokens inside swept files.  An occurrence of the name inside a
# match is kept.  The first rule that covers it names its owner in the report.
# Owner '@app' means T4 inside app/ and T6 elsewhere.
RULES = [
    ('url', '@app', 'URLs naming the repo, site or stores',
     re.compile(r'krrishnarraj/clpeak|github\.io/clpeak|snapcraft\.io/clpeak|id=kr\.clpeak')),
    ('flathub-id', 'T5', 'the Flathub app ID',
     re.compile(r'io\.github\.krrishnarraj\.clpeak')),
    ('reverse-dns', 'T4', 'kr.clpeak IDs, method channels, the Kotlin package dir',
     re.compile(r'\bkr[./]clpeak\b(?:[./][\w${}.-]+)*')),
    ('icon', 'T4', 'icon files and references to them',
     re.compile(r'\bic_stat_clpeak\b|\bclpeak_icon\b|\bclpeak_master_\d+\b')),
    ('schema', 'T3', 'run-document schemas, and <documents>/clpeak/runs',
     re.compile(r'\bclpeak/run[\w-]*')),
    ('version-key', 'T3', 'the run-document key clpeak_version',
     re.compile(r'\bclpeak_version\b(?!\s*\()')),
    ('index-key', 'T3', 'the history-index key clpeakVersion',
     re.compile(r'(?<=[\'"])clpeakVersion(?=[\'"])')),
    ('suffix', 'T3', 'file suffixes and folders (.clpeak.json, ~/.clpeak, ...)',
     re.compile(r'\.clpeak\b(?:\.\w+)*')),
    ('suffix', 'T3', 'file suffixes and folders (.clpeak.json, ~/.clpeak, ...)',
     re.compile(r'(?<![\w.])clpeak\.(?:partial\.json|resume\.json|json|log)\b')),
    ('mobile-folder', 'T3', 'the mobile history folder in RunHistoryStore.baseDirectory()',
     re.compile(r'docs\.path,\s*\'clpeak\'')),
    ('coreml-temp', 'T3', 'the Core ML temp-file prefix',
     re.compile(r'\bclpeak-coreml-')),
    ('history', 'stays', 'old version numbers and asset names',
     re.compile(r'\bclpeak[ -]v?\d+\.\d+(?:\.\d+)*')),
    # With the schema rule above, what 4.0 needs to refuse a 3.x file by name.
    ('refusal', 'stays', 'the 3.x refusal: a file clpeak 3.x wrote is refused by name',
     re.compile(r'\bclpeak 3\.x\b')),
    ('command', 'stays', 'the clpeak command: xpupeak under its 3.x name (D1)',
     lambda text: second_name(text)),
]


# The clpeak command (D1): the same program as xpupeak, under its 3.x name.
# A 3.x line names the program clpeak too, and there it becomes xpupeak, so
# the rule keeps only what no 3.x line says: the command on a line that names
# xpupeak as well, next to the words that say which name it is, or where
# only 4.0 puts it -- the unpacked zip's $bin/clpeak, the install link, the
# second CTest, the second --version in CI.
_COMMAND_WORD = re.compile(r'(?<![\w.-])clpeak(?![\w-])')
_SAYS_WHICH = re.compile(r'3\.x command|3\.x name|second name|both commands|`clpeak` says')
_ONLY_4 = re.compile(r'\$bin/clpeak|INSTALL_BINDIR\}/clpeak|^ *add_test\(clpeak_test_run clpeak\)'
                     r'|^ *\./build/clpeak --version$', re.M)


def second_name(text):
    out = [(m.start(), m.end()) for m in _ONLY_4.finditer(text)]
    for m in _COMMAND_WORD.finditer(text):
        line = text[text.rfind('\n', 0, m.start()) + 1:_line_end(text, m.start())]
        if 'xpupeak' in line or _SAYS_WHICH.search(line):
            out.append((m.start(), m.end()))
    return out

# Spans renamed even inside a protected token: the Dart FFI lookups, whose
# 'clpeak_version' is the C function, not the run-document key.
OVERRIDES = [
    re.compile(r'lookupFunction<[^>]*>\(\s*\'(clpeak_\w+)\''),
]

C_FAMILY = ('.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.m', '.mm',
            '.cu', '.cuh', '.hip', '.metal', '.cl', '.h.in')


def git(*args):
    r = subprocess.run(['git'] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode:
        sys.exit('git %s: %s' % (' '.join(args), r.stderr.decode(errors='replace').strip()))
    return r.stdout


def path_owner(path):
    for prefix, owner, what in PROTECTED_PATHS:
        if path == prefix or (prefix.endswith('/') and path.startswith(prefix)):
            return owner, what, prefix
    return None


def resolve(owner, path):
    if owner == '@app':
        return 'T4' if path.startswith('app/') else 'T6'
    return owner


# Filled in by kept_file_rules(): the names of tracked files whose own path
# keeps the name (packaging/homebrew/clpeak.rb until T5 renames it, the
# icons), so a mention of one still names a file that exists.
KEPT_FILES = []


def protected_spans(text):
    spans = []
    for name, owner, what, rx in RULES + KEPT_FILES:
        found = rx(text) if callable(rx) else ((m.start(), m.end()) for m in rx.finditer(text))
        for s, e in found:
            spans.append((s, e, name, owner, what))
    return spans


def kept_file_rules(paths):
    by_owner = OrderedDict()
    for path in paths:
        base = path.rsplit('/', 1)[-1]
        if not OLD.search(base):
            continue
        po = path_owner(path)
        if po:
            owner = po[0]
        else:
            edits, kept = classify(path, protected_spans(path), [])
            if edits or not kept:
                continue                  # git-mv'd by this script
            owner = resolve(kept[0][1][3], path)
        by_owner.setdefault(owner, set()).add(base)
    return [('kept-file', owner, 'names of files that keep the name until their task renames them',
             re.compile(r'(?<![\w.-])(?:%s)(?![\w-])' % '|'.join(re.escape(b) for b in sorted(names))))
            for owner, names in by_owner.items()]


def override_spans(text):
    out = []
    for rx in OVERRIDES:
        for m in rx.finditer(text):
            out.append((m.start(1), m.end(1)))
    return out


def classify(text, spans, overrides, prose=None):
    """Split TOKEN matches into edits (start, end, new spelling) and the 3.x
    spellings kept by a rule, as (offset, rule) pairs.  `prose` is
    prose_spans()'s answer for `text`; None for text that has no prose."""
    edits, kept = [], []
    for m in TOKEN.finditer(text):
        a, b = m.start(), m.end()
        if not any(s <= a and b <= e for s, e in overrides):
            rule = next((sp for sp in spans if sp[0] <= a and b <= sp[1]), None)
            if rule is not None:
                if OLD.fullmatch(m.group(0)):
                    kept.append((a, rule))
                continue
        new = spelling(text, a, b, prose is not None and prose.covers(a))
        if new != m.group(0):
            edits.append((a, b, new))
    return edits, kept


def respell(name):
    """A name's 4.0 spelling, outside prose: what the sweep makes of it in code."""
    edits, _ = classify(name, [], [])
    return rename_plain(name, edits)


def rename_plain(text, edits):
    out, last = [], 0
    for a, b, new in edits:
        out.append(text[last:a])
        out.append(new)
        last = b
    out.append(text[last:])
    return ''.join(out)


# --- the spelling (D11) -------------------------------------------------------

# What may stand right before a bare name in prose, and right after it.  Any
# other neighbour makes it part of something longer: a path (src/xpupeak,
# ~/.xpupeak), a file name, a flag, a key (name=xpupeak), a placeholder, a
# word, or a literal in quotes or code format.
_BEFORE_OK = set(' \t\n\r([*>|,;!?\u2014')
_AFTER_OK = set(' \t\n\r)]*.,;:!?\'|<\u2014')
_COMMAND = re.compile(r' --?[A-Za-z0-9]')    # xpupeak --list-devices, xpupeak -o run.json


def spelling(text, a, b, in_prose):
    """The 4.0 spelling of the name at text[a:b]."""
    old = text[a:b]
    if old == 'CLPEAK':
        return 'XPUPEAK'
    if old in ('Clpeak', 'Xpupeak'):
        return 'XpuPeak'
    if 'A' <= text[b:b + 1] <= 'Z':
        return 'xpuPeak'
    if in_prose and product_named(text, a, b):
        return 'XpuPeak'
    return 'xpupeak'


def product_named(text, a, b):
    """Whether the lowercase name at text[a:b], in a comment or in Markdown
    text, names the product, rather than the command, a file, a path, a URL,
    a namespace or a literal."""
    before = text[a - 1] if a else '\n'
    after = text[b] if b < len(text) else '\n'
    if before not in _BEFORE_OK or after not in _AFTER_OK:
        return False
    nxt = text[b + 1:b + 2]
    if after == '.' and nxt and (nxt.isalnum() or nxt in '_-/'):
        return False                           # xpupeak.json, but not a full stop
    if after == ':' and nxt == ':':
        return False                           # xpupeak::memoryBudget()
    if after == "'" and nxt != 's':
        return False                           # 'xpupeak', but not xpupeak's
    if text[max(0, a - 10):a].endswith(('namespace ', '$ ')):
        return False                           # } // namespace xpupeak; $ xpupeak
    if _COMMAND.match(text, b):
        return False                           # a command line
    return True


# --- where the prose is ---------------------------------------------------------

class Spans:
    """Disjoint [start, end) ranges of a text, sorted."""

    def __init__(self, ranges):
        self.ranges = ranges
        self.starts = [s for s, _ in ranges]

    def covers(self, i):
        k = bisect.bisect_right(self.starts, i) - 1
        return k >= 0 and i < self.ranges[k][1]


def _line_end(text, i):
    j = text.find('\n', i)
    return len(text) if j < 0 else j


def _merge(ranges):
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        elif s < e:
            out.append((s, e))
    return out


def _minus(ranges, holes):
    """`ranges` with `holes` cut out of them; both merged."""
    out, k = [], 0
    for s, e in ranges:
        while k < len(holes) and holes[k][1] <= s:
            k += 1
        j = k
        while j < len(holes) and holes[j][0] < e:
            if holes[j][0] > s:
                out.append((s, holes[j][0]))
            s = max(s, holes[j][1])
            j += 1
        if s < e:
            out.append((s, e))
    return out


_TICKS = re.compile(r'`+')


def _code_span_end(text, m, b):
    """Where the code span that the run of backticks `m` opens ends: at the
    next run of the same length, within the paragraph and before b.  None
    when nothing closes it."""
    stop = text.find('\n\n', m.end(), b)
    stop = b if stop < 0 else stop
    return next((c.end() for c in _TICKS.finditer(text, m.end(), stop)
                 if len(c.group(0)) == len(m.group(0))), None)


def _code_spans(text, a, b):
    """The code spans (`...`) in text[a:b]."""
    out, i = [], a
    while True:
        m = _TICKS.search(text, i, b)
        if not m:
            return out
        end = _code_span_end(text, m, b)
        if end is None:
            i = m.end()
        else:
            out.append((m.start(), end))
            i = end


def _skip_quoted(text, i, cxx):
    """The offset just past the string or character literal opening at text[i]."""
    q, n = text[i], len(text)
    k = i
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] == '_'):
        k -= 1
    prefix = text[k:i]
    if cxx:
        if q == "'" and prefix and prefix not in ('u', 'U', 'L', 'u8'):
            return i + 1                       # a digit separator, or an apostrophe
        m = re.match(r'([^()\\\s"]{0,16})\(', text[i + 1:i + 18]) if q == '"' else None
        if prefix in _RAW_PREFIX and m:
            close = ')' + m.group(1) + '"'
            j = text.find(close, i + 1 + m.end())
            return n if j < 0 else j + len(close)
    else:
        if text.startswith(q * 3, i):          # ''' and """: Dart, Kotlin, Swift, Python
            j = text.find(q * 3, i + 3)
            return n if j < 0 else j + 3
        if prefix == 'r':                      # a Dart or Python raw string: no escapes
            j, e = text.find(q, i + 1), _line_end(text, i)
            return e if j < 0 or j > e else j + 1
    j = i + 1
    while j < n and text[j] != q and text[j] != '\n':
        j += 2 if text[j] == '\\' else 1
    return j + 1


_C_START = re.compile(r'//|/\*|["\']')


def c_comments(text, cxx):
    """The comments of C, C++, Objective-C, CUDA, Metal, GLSL, Dart, Kotlin, Swift ..."""
    spans, i, n = [], 0, len(text)
    while True:
        m = _C_START.search(text, i)
        if not m:
            return spans
        a = m.start()
        if m.group(0) == '//':
            i = _line_end(text, a)
            spans.append((a, i))
        elif m.group(0) == '/*':
            j = text.find('*/', a + 2)
            i = n if j < 0 else j + 2
            spans.append((a, i))
        else:
            i = _skip_quoted(text, a, cxx)


_HASH_START = re.compile(r'#|["\']|\[=*\[|<#')


def hash_comments(text, lang):
    """The # comments of CMake, shell, YAML, PowerShell, Python ...  Outside
    CMake, a # starts a comment only at the start of a word."""
    spans, i, n = [], 0, len(text)
    while True:
        m = _HASH_START.search(text, i)
        if not m:
            return spans
        a, tok = m.start(), m.group(0)
        prev = text[a - 1] if a else '\n'
        if tok == '<#':
            if lang == 'ps1':                  # <# a block comment #>
                j = text.find('#>', a + 2)
                i = n if j < 0 else j + 2
                spans.append((a, i))
            else:
                i = a + 1
        elif tok == '#':
            if lang == 'cmake':
                b = re.match(r'#\[(=*)\[', text[a:a + 40])
                if b:                          # #[[ a bracket comment ]]
                    j = text.find(']' + b.group(1) + ']', a)
                    i = n if j < 0 else j + len(b.group(1)) + 2
                    spans.append((a, i))
                    continue
            elif not (prev.isspace() or (lang == 'shell' and prev in ';|&(')):
                i = a + 1                      # $#, ${#x}, a URL's #fragment
                continue
            i = _line_end(text, a)
            spans.append((a, i))
        elif tok[0] == '[':
            if lang == 'cmake':                # [[ a bracket argument ]]
                close = ']' + '=' * (len(tok) - 2) + ']'
                j = text.find(close, a)
                i = n if j < 0 else j + len(close)
            else:
                i = a + 1
        elif tok == "'" and (lang == 'cmake' or prev.isalnum()):
            i = a + 1                          # not a quote in CMake; an apostrophe
        elif lang in ('cmake', 'shell'):       # quotes that span lines
            j = a + 1
            while j < n and text[j] != tok:
                j += 2 if text[j] == '\\' and tok == '"' else 1
            i = j + 1
        else:
            i = _skip_quoted(text, a, cxx=False)


def xml_comments(text):
    return [(m.start(), m.end()) for m in re.finditer(r'<!--.*?-->', text, re.S)]


_FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$', re.M)
_MD_START = re.compile(r'`+|\]\(|<(?=[A-Za-z/])|https?://')
_URL_END = re.compile(r'[^\s)>\]]*')


def markdown_prose(text):
    """Markdown text outside code: fenced blocks, code spans, link targets,
    HTML tags and autolinks, bare URLs.  An HTML comment is text."""
    code, fence = [], None
    for m in _FENCE.finditer(text):
        if fence is None:
            fence = m
        elif m.group(1)[0] == fence.group(1)[0] and len(m.group(1)) >= len(fence.group(1)) \
                and not m.group(2).strip():
            code.append((fence.start(), m.end()))
            fence = None
    if fence is not None:
        code.append((fence.start(), len(text)))
    fences, i = Spans(code[:]), 0
    while True:
        m = _MD_START.search(text, i)
        if not m:
            break
        a, tok = m.start(), m.group(0)
        if fences.covers(a):
            i = next(e for s, e in fences.ranges if s <= a < e)
        elif tok[0] == '`':
            end = _code_span_end(text, m, len(text))
            if end is None:
                i = m.end()
            else:
                code.append((a, end))
                i = end
        elif tok == '](':
            j = text.find(')', a)
            i = j + 1 if 0 <= j <= _line_end(text, a) else a + 2
            code.append((a + 2, i))
        elif tok == '<':
            j = text.find('>', a)
            i = j + 1 if 0 <= j <= _line_end(text, a) else a + 1
            code.append((a, i))
        else:
            i = _URL_END.match(text, a).end()
            code.append((a, i))
    return _minus([(0, len(text))], _merge(code))


C_LIKE = C_FAMILY + ('.comp', '.glsl', '.rc', '.xcconfig', '.pbxproj')
SCRIPT_C_LIKE = ('.dart', '.kt', '.kts', '.swift', '.java', '.gradle')
HASH = [(('.sh',), 'shell'), (('.yml', '.yaml'), 'yaml'), (('.ps1',), 'ps1'),
        (('.py', '.rb', '.pro', '.properties', '.desktop', '.toml', '.gitignore',
          '.gitattributes', '.gitmodules'), 'hash')]
XML = ('.xml', '.plist', '.entitlements', '.storyboard', '.xib', '.xcscheme',
       '.xcworkspacedata', '.xcsettings', '.html', '.manifest', '.svg')


def prose_spans(path, text):
    """Where in `text` (the file at `path`) a name is prose: the comments, or
    a Markdown file's text, less what is in code format.  None for a file
    with no prose."""
    if path.rsplit('/', 1)[-1] == 'CMakeLists.txt' or path.endswith('.cmake'):
        spans = hash_comments(text, 'cmake')
    elif path.endswith(C_LIKE):
        spans = c_comments(text, cxx=True)
    elif path.endswith(SCRIPT_C_LIKE):
        spans = c_comments(text, cxx=False)
    elif path.endswith(XML):
        spans = xml_comments(text)
    elif path.endswith('.md'):
        return Spans(markdown_prose(text))
    else:
        lang = next((lang for ends, lang in HASH if path.endswith(ends)), None)
        if lang is None:
            return None
        spans = hash_comments(text, lang)
    # In a comment, code format is backticks, as in Markdown.
    spans = _merge(spans)
    ticks = _merge(cs for s, e in spans for cs in _code_spans(text, s, e))
    return Spans(_minus(spans, ticks))


# --- C-family re-alignment --------------------------------------------------

_RAW_PREFIX = ('R', 'u8R', 'uR', 'UR', 'LR')


class CLex:
    """Just enough C/C++/ObjC lexing to see code apart from strings and comments."""

    def __init__(self):
        self.block = False    # inside /* */
        self.raw = None       # closing delimiter of a raw string, while inside one

    def inside(self):
        return self.block or self.raw is not None

    def scan(self, s):
        """Advance over line `s`.

        Returns the brackets outside strings and comments, as ('(' or ')',
        column); the column of the line's // comment, or None; and the columns
        where a token starts (after a space or an open bracket, not inside a
        string or a comment).
        """
        brackets, starts, i, n = [], set(), 0, len(s)
        while i < n:
            if self.block:
                j = s.find('*/', i)
                if j < 0:
                    return brackets, None, starts
                i, self.block = j + 2, False
                continue
            if self.raw is not None:
                j = s.find(self.raw, i)
                if j < 0:
                    return brackets, None, starts
                i, self.raw = j + len(self.raw), None
                continue
            c = s[i]
            if c != ' ' and (i == 0 or s[i - 1] in ' ([{<'):
                starts.add(i)
            if s.startswith('//', i):
                return brackets, i, starts
            if s.startswith('/*', i):
                self.block, i = True, i + 2
                continue
            if c == '"':
                k = i
                while k > 0 and (s[k - 1].isalnum() or s[k - 1] == '_'):
                    k -= 1
                m = re.match(r'([^()\\\s"]{0,16})\(', s[i + 1:])
                if s[k:i] in _RAW_PREFIX and m:
                    self.raw = ')' + m.group(1) + '"'
                    i += 1 + m.end()
                    continue
                i += 1
                while i < n and s[i] != '"':
                    i += 2 if s[i] == '\\' else 1
                i += 1
                continue
            if c == "'":
                if i > 0 and s[i - 1].isalnum() and not (s[i - 1] in 'uUL' or s[i - 2:i] == 'u8'):
                    i += 1            # digit separator, or an apostrophe in a word
                    continue
                i += 1
                while i < n and s[i] != "'":
                    i += 2 if s[i] == '\\' else 1
                i += 1
                continue
            if c in '([{':
                brackets.append(('(', i))
            elif c in ')]}':
                brackets.append((')', i))
            i += 1
        return brackets, None, starts


def rename_c(text, edits):
    """Rename, then keep the columns that were aligned after a renamed name.

    A continuation line that starts one past an open bracket of an earlier
    line (an aligned argument list) moves with that bracket.  One that starts
    where a token of the line above starts (a ternary's ':' under its '?')
    moves with that token.  Anything else -- block indentation, an operand
    continued four columns in -- stays where it is.  Lines map one to one, so
    a column's shift is its line's re-indent plus what each name renamed
    before it grew by.
    """
    lines = text.split('\n')
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    per_line = defaultdict(list)     # line -> [(column, old length, new spelling)]
    li = 0
    for a, b, new in edits:
        while li + 1 < len(starts) and starts[li + 1] <= a:
            li += 1
        per_line[li].append((a - starts[li], b - a, new))

    def growth(cols, c):
        return sum(len(new) - n for x, n, new in cols if x < c)

    lex = CLex()
    stack = []    # open brackets: [old column after it, new column after it, aligns]
    above = None  # the line above: (re-indent, rename columns, token starts)
    out = []
    for i, raw_line in enumerate(lines):
        cr = raw_line.endswith('\r')
        body = raw_line[:-1] if cr else raw_line
        cols = per_line.get(i, [])
        began_inside = lex.inside()
        if not body.strip() and not began_inside:
            stack, above = [], None
        indent = len(body) - len(body.lstrip(' '))
        delta = 0
        if indent and body.strip() and not began_inside and '\t' not in body[:indent]:
            for old_col, new_col, aligns in reversed(stack):
                if aligns and old_col == indent:
                    delta = new_col - old_col
                    break
            else:
                if above and indent in above[2]:
                    delta = above[0] + growth(above[1], indent)

        def grow(c):
            return delta + growth(cols, c)

        brackets, comment_col, token_starts = lex.scan(body)
        for kind, c in brackets:
            if kind == '(':
                rest = body[c + 1:].strip()
                aligns = bool(rest) and not rest.startswith(('//', '/*'))
                stack.append([c + 1, c + 1 + grow(c + 1), aligns])
            elif stack:
                stack.pop()
        if body.strip():
            above = (delta, cols, token_starts)

        line_edits = list(cols)
        if delta:
            line_edits.append((0, indent, ' ' * (indent + delta)))
        # A trailing // comment aligned with a neighbour's keeps its column.
        if comment_col is not None and body[:comment_col].strip():
            g = grow(comment_col)
            run = len(body[:comment_col]) - len(body[:comment_col].rstrip(' '))
            neighbours = [lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
            aligned = any(nb[comment_col:comment_col + 2] == '//' and nb[:comment_col].strip()
                          for nb in neighbours)
            if g and aligned and run >= 2:
                cut = min(g, run - 1)
                line_edits.append((comment_col - run, run, ' ' * (run - cut)))
        # So does a macro's continuation backslash.
        if body.endswith('\\') and comment_col is None and not lex.block:
            end = len(body) - 1
            run = len(body[:end]) - len(body[:end].rstrip(' '))
            g = grow(end)
            if g and run >= 2 and body[:end].strip():
                cut = min(g, run - 1)
                line_edits.append((end - run, run, ' ' * (run - cut)))
        new = body
        for c, length, repl in sorted(line_edits, reverse=True):
            new = new[:c] + repl + new[c + length:]
        out.append(new + ('\r' if cr else ''))
    return '\n'.join(out)


# --- the sweep ----------------------------------------------------------------

def tracked():
    """Every tracked file's path; gitlinks (submodules) and symlinks left out."""
    out = []
    for entry in git('ls-files', '-s', '-z').split(b'\0'):
        if not entry:
            continue
        meta, path = entry.split(b'\t', 1)
        mode = meta.split()[0]
        if mode in (b'160000', b'120000'):
            continue
        out.append(path.decode('utf-8', 'surrogateescape'))
    return out


def read_text(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        return None
    if b'\0' in data:
        return None
    return data.decode('utf-8', 'surrogateescape')


def sweep(paths, write):
    changed, count = [], 0
    for path in paths:
        po = path_owner(path)
        if po and po[2] in UPPER_KEPT:
            continue
        text = read_text(path)
        if text is None or not TOKEN.search(text):
            continue
        if po:
            edits = [(m.start(), m.end(), 'XPUPEAK') for m in UPPER.finditer(text)]
        else:
            edits, _ = classify(text, protected_spans(text), override_spans(text),
                                prose_spans(path, text))
        if not edits:
            continue
        is_c = path.endswith(C_FAMILY)
        new = rename_c(text, edits) if is_c else rename_plain(text, edits)
        if new == text:
            continue
        count += len(edits)
        changed.append((path, len(edits)))
        if write:
            with open(path, 'wb') as f:
                f.write(new.encode('utf-8', 'surrogateescape'))
    return changed, count


def moves(paths):
    out, existing = [], set(paths)
    for path in paths:
        if path_owner(path) or not TOKEN.search(path):
            continue
        edits, _ = classify(path, protected_spans(path), [])
        if edits:
            new = rename_plain(path, edits)
            out.append((path, new, new in existing or os.path.lexists(new)))
    return out


def report(paths):
    groups = OrderedDict()     # (owner, rule) -> {'what', 'files': {path: n}, 'forms': set}
    unexpected = []
    dual = OrderedDict()

    def add(owner, rule, what, path, form):
        g = groups.setdefault((owner, rule), {'what': what, 'files': OrderedDict(), 'forms': set()})
        g['files'][path] = g['files'].get(path, 0) + 1
        if form:
            g['forms'].add(form)

    corpus = []
    total, files = 0, set()
    for path in paths:
        text = read_text(path)
        if text is None:
            continue
        corpus.append((path, text))
        if not ANY.search(text):
            continue
        po = path_owner(path)
        spans = protected_spans(text) if not po else []
        overrides = override_spans(text) if not po else []
        for m in ANY.finditer(text):
            total += 1
            files.add(path)
            a, b = m.start(), m.end()
            if po and not (text[a:b] == 'CLPEAK' and po[2] not in UPPER_KEPT):
                add(po[0], 'path:' + po[2], po[1], path, None)
                continue
            if text[a:b] == 'clPeak':
                add('stays', 'clPeak', 'clPeak, the OpenCL backend class', path, None)
                continue
            rule = None
            if not any(s <= a and b <= e for s, e in overrides):
                rule = next((sp for sp in spans if sp[0] <= a and b <= sp[1]), None)
            if rule is None:
                line = text.count('\n', 0, a) + 1
                unexpected.append('%s:%d: %s' % (path, line, text[max(0, a - 30):b + 30].replace('\n', ' ')))
                continue
            s, e, name, owner, what = rule
            form = text[s:e].strip()
            if len(form) > 60:
                form = form[:57] + '...'
            add(resolve(owner, path), name, what, path, form)
            word = re.match(r'\w*', text[a:]).group(0)
            head = re.search(r'\w*$', text[:a]).group(0)
            ident = head + word
            if ident.lower() != 'clpeak' and name not in ('history', 'command'):
                dual.setdefault(ident, set()).add(path)

    # A kept identifier whose renamed spelling also exists is a name/key pair.
    pairs = []
    swept = '\n'.join(t for p, t in corpus if not path_owner(p))
    for ident, where in dual.items():
        new = respell(ident)
        if re.search(r'\b%s\b' % re.escape(new), swept):
            pairs.append((ident, new, sorted(where)))

    lines = ['Residue: %d mentions of clpeak (git grep -n -i clpeak) in %d files.' % (total, len(files))]
    for owner, title in OWNERS.items():
        mine = [(k, g) for k, g in groups.items() if k[0] == owner]
        if not mine:
            continue
        n = sum(sum(g['files'].values()) for _, g in mine)
        lines.append('')
        lines.append('%s -- %s: %d' % (owner, title, n))
        for (o, rule), g in mine:
            fs = g['files']
            lines.append('  %s: %d in %d file%s' % (g['what'], sum(fs.values()), len(fs),
                                                   '' if len(fs) == 1 else 's'))
            if g['forms']:
                lines.extend(textwrap.wrap('forms: ' + ', '.join(sorted(g['forms'])), 76,
                                           initial_indent='    ', subsequent_indent='      '))
            # A no-break space keeps each count on its path's line.
            lines.extend(l.replace('\u00a0', ' ') for l in textwrap.wrap(
                ', '.join('%s\u00a0(%d)' % kv for kv in fs.items()), 76,
                initial_indent='    ', subsequent_indent='    ', break_on_hyphens=False))
    if pairs:
        lines.append('')
        lines.append('Renamed as a name, kept as a persisted key:')
        for old, new, where in pairs:
            lines.extend(textwrap.wrap('%s -> %s; the key %s stays in %s' % (old, new, old, ', '.join(where)),
                                       76, initial_indent='  ', subsequent_indent='    ',
                                       break_on_hyphens=False))
    still = [p for p in paths if ANY.search(p)]
    if still:
        lines.append('')
        lines.append('Paths that keep the name:')
        for p in still:
            po = path_owner(p)
            owner = po[0] if po else next((resolve(sp[3], p) for sp in protected_spans(p)), '?')
            lines.append('  %s (%s)' % (p, owner))
    if unexpected:
        lines.append('')
        lines.append('UNEXPECTED (not protected, not renamed): %d' % len(unexpected))
        lines.extend('  ' + u for u in unexpected[:50])
    return '\n'.join(lines), unexpected


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='say what would change, touch nothing')
    ap.add_argument('--check', action='store_true', help='exit 1 if anything is left to rename')
    ap.add_argument('--report', action='store_true', help='print the residue report only')
    args = ap.parse_args()

    os.chdir(git('rev-parse', '--show-toplevel').decode().strip())
    paths = tracked()
    KEPT_FILES[:] = kept_file_rules(paths)
    write = not (args.dry_run or args.check or args.report)

    if not args.report:
        changed, count = sweep(paths, write)
        mv = moves(paths)
        verb = 'renamed' if write else 'would rename'
        print('%s %d names in %d files' % (verb, count, len(changed)))
        for path, n in changed:
            print('  %s (%d)' % (path, n))
        for old, new, clash in mv:
            if clash:
                print('  CONFLICT: %s and %s both exist; merge them by hand' % (old, new))
            elif write:
                if os.path.dirname(new):
                    os.makedirs(os.path.dirname(new), exist_ok=True)
                git('mv', old, new)
                print('  git mv %s %s' % (old, new))
            else:
                print('  would git mv %s %s' % (old, new))
        if args.check:
            sys.exit(1 if (changed or mv) else 0)
        if write:
            paths = tracked()

    text, unexpected = report(paths)
    print()
    print(text)
    if unexpected:
        sys.exit(2)


if __name__ == '__main__':
    main()
