# cmake/GenVersion.cmake
# Invoked at build time via cmake -P to regenerate version.h only when needed.
# Variables passed in via -D: SOURCE_DIR, BINARY_DIR, TEMPLATE_DIR, FALLBACK, GIT_EXECUTABLE

set(CLPEAK_VERSION_STR "${FALLBACK}")

if(GIT_EXECUTABLE AND EXISTS "${SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty --long
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE _git_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result
  )
  if(_git_result EQUAL 0)
    string(REGEX REPLACE "^v" "" _git_version "${_git_version}")
    set(CLPEAK_VERSION_STR "${_git_version}")
  endif()
endif()

# Write to a temporary file, then compare with the existing version.h.
# Only overwrite if changed — this keeps the mtime stable and avoids
# unnecessary recompilation of everything that includes version.h.
set(_version_h_path "${BINARY_DIR}/generated/version.h")
set(_version_h_tmp  "${BINARY_DIR}/generated/version.h.tmp")

configure_file("${TEMPLATE_DIR}/version.h.in" "${_version_h_tmp}" @ONLY)

if(EXISTS "${_version_h_path}")
  file(READ "${_version_h_path}" _old_contents)
  file(READ "${_version_h_tmp}"  _new_contents)
  if("${_old_contents}" STREQUAL "${_new_contents}")
    file(REMOVE "${_version_h_tmp}")
    return()
  endif()
endif()

file(RENAME "${_version_h_tmp}" "${_version_h_path}")
