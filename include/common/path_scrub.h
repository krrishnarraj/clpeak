#ifndef XPUPEAK_PATH_SCRUB_H
#define XPUPEAK_PATH_SCRUB_H

#include <string>
#include <vector>

namespace xpupeak {

// ── Paths without the person in them ───────────────────────────────────────
//
// A result file is made to be shared, and a path is where a user's name, or
// an app install's random id, still reaches one: a home directory is named
// after its user, a macOS temporary folder is the user's own, Android
// installs an app under random directory names and iOS under a UUID.
// scrubPaths() rewrites those parts of every path in `text` and keeps the
// rest, which is what makes a path worth having in a log:
//
//   the home directory ($HOME; on Windows %USERPROFILE%)   ~
//   the temporary directory, where it is the user's own    $TMPDIR
//   any other home: /Users/<name>, /home/<name>,
//     C:\Users\<name>                                      /Users/<user> ...
//   macOS's per-user /var/folders/<a>/<b>                  /var/folders/<user>
//   Android's /data/app/~~<id>/<package>-<id>             /data/app/<install>
//   iOS's .../Application/<UUID>, .../AppGroup/<UUID>     .../Application/<install>
//
// RunLog applies it to every entry, the document's and the sidecar's alike
// (include/common/run_log.h), and a result's and a sidecar's
// `invocation.argv` go through scrubArgv().  A checkpoint keeps argv as it
// was given: it is not shared, and --resume parses it again.
std::string scrubPaths(const std::string &text);

// argv as a result records it: the program by its file name alone, every
// argument after it through scrubPaths().
std::vector<std::string> scrubArgv(const std::vector<std::string> &argv);

} // namespace xpupeak

#endif // XPUPEAK_PATH_SCRUB_H
