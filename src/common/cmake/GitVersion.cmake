# cmake/GitVersion.cmake
# The single definition of "what version is this build", shared by version.cmake
# (configure time) and GenVersion.cmake (build time) so the two can never
# disagree about it.
#
# Why this is not just `git describe --dirty`:
#
#   `flutter build` rewrites the files the SDK owns -- pubspec.lock and the
#   per-platform generated plugin registrants -- whenever the SDK differs at all
#   from the one that produced the committed copies.  That is routine in CI,
#   where the runner tracks Flutter stable while the checked-in files came from
#   whatever a developer had installed.  It is pure tool churn: not one of those
#   files is an input to the native binaries.
#
#   It still ended up stamped on them, because CPack runs a `preinstall` pass
#   before staging the install tree.  That re-runs the version check -- now
#   against a tree the GUI build has dirtied -- rewrites version.h, and relinks
#   the CLI and clpeak_ffi as -dirty on the way into the package.  A pristine
#   tag checkout shipped binaries claiming modified sources.
#
#   So the dirty flag is computed here over everything *except* those generated
#   files.  A real edit -- including one to the GUI's Dart or runner sources --
#   still marks the build dirty.
set(CLPEAK_GIT_DIRTY_EXCLUDES
    ":(exclude)app/pubspec.lock"
    ":(exclude)app/linux/flutter/generated_plugin_registrant.cc"
    ":(exclude)app/linux/flutter/generated_plugin_registrant.h"
    ":(exclude)app/linux/flutter/generated_plugins.cmake"
    ":(exclude)app/windows/flutter/generated_plugin_registrant.cc"
    ":(exclude)app/windows/flutter/generated_plugin_registrant.h"
    ":(exclude)app/windows/flutter/generated_plugins.cmake"
    ":(exclude)app/macos/Flutter/GeneratedPluginRegistrant.swift"
)

# clpeak_git_describe(<out_var> <git_executable> <git_root> <fallback>)
#
# Sets <out_var> in the caller's scope to the git-describe string with a
# "-dirty" suffix when the tree carries real modifications, or to <fallback>
# when there is no usable git checkout (tarball builds).
function(clpeak_git_describe out_var git_exe git_root fallback)
    set(_version "${fallback}")

    if(git_exe AND EXISTS "${git_root}/.git")
        execute_process(
            COMMAND ${git_exe} describe --tags --always
            WORKING_DIRECTORY ${git_root}
            OUTPUT_VARIABLE _describe
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _describe_result
        )
        if(_describe_result EQUAL 0)
            # Strip optional leading 'v' (tags mix v1.0 and 1.1.7)
            string(REGEX REPLACE "^v" "" _describe "${_describe}")

            # `git status` (not diff-index) so the index stat cache is refreshed
            # first -- a fresh CI checkout is stat-dirty and would otherwise
            # report phantom modifications.  -uno keeps untracked files out of
            # it, matching what --dirty counts.  The leading "." is required:
            # a pathspec of only :(exclude) entries matches nothing.
            execute_process(
                COMMAND ${git_exe} status --porcelain --untracked-files=no
                        -- . ${CLPEAK_GIT_DIRTY_EXCLUDES}
                WORKING_DIRECTORY ${git_root}
                OUTPUT_VARIABLE _status
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _status_result
            )
            if(_status_result EQUAL 0 AND NOT "${_status}" STREQUAL "")
                set(_describe "${_describe}-dirty")
            endif()

            set(_version "${_describe}")
        endif()
    endif()

    set(${out_var} "${_version}" PARENT_SCOPE)
endfunction()
