# android/app/src/main/cpp/version_android.cmake
# Specialized version helper for Android that finds the Git root correctly.

set(_ANDROID_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
get_filename_component(_GIT_ROOT "${_ANDROID_CMAKE_DIR}/../../../../.." ABSOLUTE)
set(CLPEAK_VERSION_FALLBACK "1.1.7")

find_package(Git QUIET)

# Re-run version detection with correct working directory
if(GIT_FOUND AND EXISTS "${_GIT_ROOT}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty --long
    WORKING_DIRECTORY ${_GIT_ROOT}
    OUTPUT_VARIABLE _git_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result
  )
  if(_git_result EQUAL 0)
    string(REGEX REPLACE "^v" "" _git_version "${_git_version}")
    set(CLPEAK_VERSION_STR "${_git_version}")
  else()
    set(CLPEAK_VERSION_STR "${CLPEAK_VERSION_FALLBACK}")
  endif()
else()
  set(CLPEAK_VERSION_STR "${CLPEAK_VERSION_FALLBACK}")
endif()

# Override the build-time generation to use the correct SOURCE_DIR
function(clpeak_setup_version_android target)
  if(NOT TARGET clpeak_version_gen)
    add_custom_target(clpeak_version_gen ALL
      COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${_GIT_ROOT}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTEMPLATE_DIR=${CLPEAK_ROOT}/cmake
        -DFALLBACK=${CLPEAK_VERSION_FALLBACK}
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -P ${CLPEAK_ROOT}/cmake/GenVersion.cmake
      WORKING_DIRECTORY ${_GIT_ROOT}
      COMMENT "Checking clpeak version (Android)..."
    )
  endif()

  add_dependencies(${target} clpeak_version_gen)
  target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()
