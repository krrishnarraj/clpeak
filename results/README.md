# Reference runs

Saved runs, one file per machine, named by vendor and device. They are the
baselines to check a number against when it looks wrong -- see the root
`AGENTS.md` ("Is this number plausible?") -- and the data behind the README's
sample output and the screenshots (`tools/screenshots.sh`).

Produced by:

```
xpupeak -o results/<vendor>/<Device_Name>.json
```

Format: `docs/format-v4.md`. The runs saved here are still format 3, written
by clpeak 3.x, until they are run again: `--compare` and the app's Import
(History) refuse them until then. A run from hardware not listed here is
welcome as a pull request.
