#!/usr/bin/env python3
"""The 4.0 rename sweep: clpeak -> xpupeak for every name only the tree sees.

    tools/rename_xpupeak.py            rename, git-mv, then print the residue
    tools/rename_xpupeak.py --dry-run  say what would change, touch nothing
    tools/rename_xpupeak.py --check    exit 1 if anything is left to rename
    tools/rename_xpupeak.py --report   the residue report only

Exits 2 when the report finds a mention that is neither renamed nor protected.

Over tracked text files it maps clpeak -> xpupeak, Clpeak -> Xpupeak and
CLPEAK -> XPUPEAK.  The mapping is case-exact, so clPeak (the OpenCL
backend's class) never matches.  Tracked paths that carry the name are
git-mv'd the same way.

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

Renamed names grow by one character, so in C-family sources the sweep also
re-aligns what was aligned to a column after them: continuation lines under
an open bracket, trailing comments aligned with a neighbour's, and macro
continuation backslashes.

Delete this script once 4.0.0 ships.
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap
from collections import OrderedDict, defaultdict

SELF = 'tools/rename_xpupeak.py'

# The three spellings that are renamed.  Six characters each; every
# replacement adds exactly one.
TOKEN = re.compile(r'clpeak|Clpeak|CLPEAK')
MAP = {'clpeak': 'xpupeak', 'Clpeak': 'Xpupeak', 'CLPEAK': 'XPUPEAK'}
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
]

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
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), name, owner, what))
    return spans


def kept_file_rules(paths):
    by_owner = OrderedDict()
    for path in paths:
        base = path.rsplit('/', 1)[-1]
        if not TOKEN.search(base):
            continue
        po = path_owner(path)
        if po:
            owner = po[0]
        else:
            renamed, kept = classify(path, protected_spans(path), [])
            if renamed or not kept:
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


def classify(text, spans, overrides):
    """Split TOKEN matches into renamed offsets and kept (offset, rule) pairs."""
    renamed, kept = [], []
    for m in TOKEN.finditer(text):
        a, b = m.start(), m.end()
        if any(s <= a and b <= e for s, e in overrides):
            renamed.append(a)
            continue
        rule = next((sp for sp in spans if sp[0] <= a and b <= sp[1]), None)
        if rule is None:
            renamed.append(a)
        else:
            kept.append((a, rule))
    return renamed, kept


def rename_plain(text, renamed):
    out, last = [], 0
    for a in renamed:
        out.append(text[last:a])
        out.append(MAP[text[a:a + 6]])
        last = a + 6
    out.append(text[last:])
    return ''.join(out)


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


def rename_c(text, renamed):
    """Rename, then keep the columns that were aligned after a renamed name.

    A continuation line that starts one past an open bracket of an earlier
    line (an aligned argument list) moves with that bracket.  One that starts
    where a token of the line above starts (a ternary's ':' under its '?')
    moves with that token.  Anything else -- block indentation, an operand
    continued four columns in -- stays where it is.  Lines map one to one, so
    a column's shift is its line's re-indent plus one per name renamed before it.
    """
    lines = text.split('\n')
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    per_line = defaultdict(list)
    li = 0
    for a in renamed:
        while li + 1 < len(starts) and starts[li + 1] <= a:
            li += 1
        per_line[li].append(a - starts[li])

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
                    delta = above[0] + sum(1 for x in above[1] if x < indent)

        def grow(c):
            return delta + sum(1 for x in cols if x < c)

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

        edits = [(c, 6, MAP[body[c:c + 6]]) for c in cols]
        if delta:
            edits.append((0, indent, ' ' * (indent + delta)))
        # A trailing // comment aligned with a neighbour's keeps its column.
        if comment_col is not None and body[:comment_col].strip():
            g = grow(comment_col)
            run = len(body[:comment_col]) - len(body[:comment_col].rstrip(' '))
            neighbours = [lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
            aligned = any(nb[comment_col:comment_col + 2] == '//' and nb[:comment_col].strip()
                          for nb in neighbours)
            if g and aligned and run >= 2:
                cut = min(g, run - 1)
                edits.append((comment_col - run, run, ' ' * (run - cut)))
        # So does a macro's continuation backslash.
        if body.endswith('\\') and comment_col is None and not lex.block:
            end = len(body) - 1
            run = len(body[:end]) - len(body[:end].rstrip(' '))
            g = grow(end)
            if g and run >= 2 and body[:end].strip():
                cut = min(g, run - 1)
                edits.append((end - run, run, ' ' * (run - cut)))
        new = body
        for c, length, repl in sorted(edits, reverse=True):
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
            renamed = [m.start() for m in UPPER.finditer(text)]
        else:
            renamed, _ = classify(text, protected_spans(text), override_spans(text))
        if not renamed:
            continue
        is_c = path.endswith(C_FAMILY)
        new = rename_c(text, renamed) if is_c else rename_plain(text, renamed)
        if new == text:
            continue
        count += len(renamed)
        changed.append((path, len(renamed)))
        if write:
            with open(path, 'wb') as f:
                f.write(new.encode('utf-8', 'surrogateescape'))
    return changed, count


def moves(paths):
    out, existing = [], set(paths)
    for path in paths:
        if path_owner(path) or not TOKEN.search(path):
            continue
        renamed, _ = classify(path, protected_spans(path), [])
        if renamed:
            new = rename_plain(path, renamed)
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
            form = text[s:e]
            if len(form) > 60:
                form = form[:57] + '...'
            add(resolve(owner, path), name, what, path, form)
            word = re.match(r'\w*', text[a:]).group(0)
            head = re.search(r'\w*$', text[:a]).group(0)
            ident = head + word
            if ident.lower() != 'clpeak' and name != 'history':
                dual.setdefault(ident, set()).add(path)

    # A kept identifier whose renamed spelling also exists is a name/key pair.
    pairs = []
    swept = '\n'.join(t for p, t in corpus if not path_owner(p))
    for ident, where in dual.items():
        new = TOKEN.sub(lambda m: MAP[m.group(0)], ident)
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
