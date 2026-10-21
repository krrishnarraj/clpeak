---
title: Code signing policy
---

# Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io/),
certificate by [SignPath Foundation](https://signpath.org/).

## What is signed

The Windows release zips on the
[releases page](https://github.com/{{ site.repo }}/releases). In each zip the
binaries built from this repository are signed: `bin\clpeak.exe`,
`gui\clpeak-gui.exe`, `gui\clpeak-engine.exe` and `gui\clpeak_ffi.dll`. The
Flutter engine and plugin DLLs next to them are third-party code and ship as
their authors built them.

Every signed binary is built from this repository's source by its
[GitHub Actions workflow](https://github.com/{{ site.repo }}/blob/master/.github/workflows/build.yml),
on GitHub-hosted runners, from a release tag. Nothing built elsewhere is
submitted for signing.

## Team roles

- **Committers and reviewers:** [Krishnaraj Bhat](https://github.com/krrishnarraj)
- **Approvers:** [Krishnaraj Bhat](https://github.com/krrishnarraj)

Changes from outside contributors are reviewed by a committer before they are
merged. Each signing request is approved by hand.

## Privacy policy

This program will not transfer any information to other networked systems
unless specifically requested by the user or the person installing or
operating it.

clpeak's results stay on the machine they were measured on. The app opens a
web page only when you click a link (the project page, or "report a bug"),
and shares a result only when you choose to.
