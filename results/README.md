# Reference runs

Saved clpeak runs, one file per machine, named by vendor and device. They are
the baselines to check a number against when it looks wrong -- see the root
`AGENTS.md` ("Is this number plausible?") -- and the data behind the README's
sample output and the screenshots (`tools/screenshots.sh`).

Produced by:

```
clpeak -o results/<vendor>/<Device_Name>.json
```

Format: `docs/format-v3.md`. A run from hardware not listed here is welcome as
a pull request.
