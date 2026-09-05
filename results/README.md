# Reference runs

Saved clpeak runs, one file per device, used as the sanity check when a number
looks wrong — see the root `AGENTS.md` ("Is this number plausible?").

Produced by:

```
clpeak -o results/<vendor>/<Device_Name>.clpeak.json
```

The v2 XML files that lived here were removed with the format-v3 change; they
are being regenerated on real hardware, so the directory is sparse until they
land. Format schema: `docs/format-v3.md`.
