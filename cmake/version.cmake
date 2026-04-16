# cmake/version.cmake
# Derives CLPEAK_VERSION_STR from git-describe at build time.
# Falls back to a hardcoded version when building from a tarball (no .git).

set(_CLPEAK_VERSION_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(CLPEAK_VERSION_FALLBACK "1.1.7")

find_package(Git QUIET)

# --- Configure-time: seed an initial version.h so the tree always has one ---
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty --long
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE _git_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result
  )
  if(_git_result EQUAL 0)
    # Strip optional leading 'v' (tags mix v1.0 and 1.1.7)
    string(REGEX REPLACE "^v" "" _git_version "${_git_version}")
    set(CLPEAK_VERSION_STR "${_git_version}")
  else()
    set(CLPEAK_VERSION_STR "${CLPEAK_VERSION_FALLBACK}")
  endif()
else()
  set(CLPEAK_VERSION_STR "${CLPEAK_VERSION_FALLBACK}")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
configure_file(
  "${_CLPEAK_VERSION_CMAKE_DIR}/version.h.in"
  "${CMAKE_BINARY_DIR}/generated/version.h"
  @ONLY
)

# --- Build-time: regenerate version.h on every build (write-if-different) ---
# The GenVersion.cmake script only overwrites the file when the string changes,
# so only options.cpp recompiles when the version actually changes.
function(clpeak_setup_version target)
  if(NOT TARGET clpeak_version_gen)
    add_custom_target(clpeak_version_gen ALL
      COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTEMPLATE_DIR=${_CLPEAK_VERSION_CMAKE_DIR}
        -DFALLBACK=${CLPEAK_VERSION_FALLBACK}
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -P ${_CLPEAK_VERSION_CMAKE_DIR}/GenVersion.cmake
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Checking clpeak version..."
    )
  endif()

  add_dependencies(${target} clpeak_version_gen)
  target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()
