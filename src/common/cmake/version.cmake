# cmake/version.cmake
# Derives CLPEAK_VERSION_STR from git-describe ONCE, at configure time, and
# generates version.h from it.  Falls back to a hardcoded version when building
# from a tarball (no .git).
#
# Deliberately not re-derived during the build.  It used to be: a custom target
# re-ran git-describe on every build so the string picked up new commits without
# reconfiguring.  That tied the version to the state of the tree *while the build
# was running* -- and the GUI build modifies tracked files, because the Flutter
# SDK rewrites pubspec.lock, analysis_options.yaml and the generated plugin
# registrants whenever its version differs from the one that produced the
# committed copies (routine in CI, which tracks Flutter stable).  CPack's
# preinstall pass then rebuilt against that dirtied tree and relinked the
# binaries as "-dirty" -- from a pristine tag checkout.  Excluding the
# offending files one by one was whack-a-mole: the list was incomplete the
# first time CI ran it.  Deriving once, before anything else has run, closes
# the whole class of problem instead of the instances of it.
#
# Consequence: the string is fixed at configure time, so after committing,
# re-run `cmake -B build` to refresh it.  The configure summary prints it.
#
# Callers may set CLPEAK_GIT_ROOT before including this file to override the
# git working directory; it otherwise resolves to the repo root relative to
# this file, which is correct for the mobile superprojects too (their
# CMAKE_SOURCE_DIR points deep into the NDK/Xcode project tree).

set(_CLPEAK_VERSION_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(CLPEAK_VERSION_FALLBACK "2.0.16")

if(NOT DEFINED CLPEAK_GIT_ROOT)
    get_filename_component(CLPEAK_GIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

find_package(Git QUIET)

set(CLPEAK_VERSION_STR "${CLPEAK_VERSION_FALLBACK}")
if(GIT_FOUND AND EXISTS "${CLPEAK_GIT_ROOT}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${CLPEAK_GIT_ROOT}
        OUTPUT_VARIABLE _git_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_result
    )
    if(_git_result EQUAL 0)
        # Strip optional leading 'v' (tags mix v1.0 and 1.1.7)
        string(REGEX REPLACE "^v" "" _git_version "${_git_version}")
        set(CLPEAK_VERSION_STR "${_git_version}")
    endif()
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
configure_file(
  "${_CLPEAK_VERSION_CMAKE_DIR}/version.h.in"
  "${CMAKE_BINARY_DIR}/generated/version.h"
  @ONLY
)

# Give <target> access to the generated version.h.
function(clpeak_setup_version target)
  target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()
