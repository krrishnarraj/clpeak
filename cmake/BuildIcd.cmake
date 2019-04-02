
set(ICD_ROOT "${CMAKE_CURRENT_BINARY_DIR}/icd_install")

set(CMAKE_LIST_CONTENT "
  cmake_minimum_required(VERSION 3.5)
  include(ExternalProject)

  ExternalProject_add(  icd_clone
    PREFIX              icd
    SOURCE_DIR          icd/src/icd
    GIT_REPOSITORY      https://github.com/krrishnarraj/OpenCL-ICD-Loader
    GIT_TAG             master
    GIT_SHALLOW         1
    GIT_PROGRESS        1
    CONFIGURE_COMMAND   \"\"
    BUILD_COMMAND       \"\"
    INSTALL_COMMAND     \"\"
  )

  ExternalProject_add(  headers_clone
    DEPENDS             icd_clone
    PREFIX              icd
    SOURCE_DIR          icd/src/icd/inc
    GIT_REPOSITORY      https://github.com/KhronosGroup/OpenCL-Headers
    GIT_TAG             master
    GIT_SHALLOW         1
    GIT_PROGRESS        1
    CONFIGURE_COMMAND   \"\"
    BUILD_COMMAND       \"\"
    INSTALL_COMMAND     \"\"
  )

  ExternalProject_add(  icd_build
    PREFIX              icd
    DEPENDS             icd_clone headers_clone
    SOURCE_DIR          icd/src/icd
    DOWNLOAD_COMMAND    \"\"
    UPDATE_COMMAND      \"\"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${ICD_ROOT}
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_SHARED_LIBS=OFF
      -DENABLE_TESTS=OFF
  )
")

set(DEPS "${CMAKE_CURRENT_BINARY_DIR}/icd")
set(DEPS_BUILD "${DEPS}/build")
file(MAKE_DIRECTORY ${DEPS} ${DEPS_BUILD})
file(WRITE "${DEPS}/CMakeLists.txt" "${CMAKE_LIST_CONTENT}")
execute_process(WORKING_DIRECTORY "${DEPS_BUILD}" COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ..)
execute_process(WORKING_DIRECTORY "${DEPS_BUILD}" COMMAND ${CMAKE_COMMAND} --build .)

set(ENV{OCL_ROOT} "${ICD_ROOT}")

if(UNIX)
  set(ENV{AMDAPPSDKROOT} "${ICD_ROOT}")
elseif(WIN32)
  set(ENV{AMDAPPSDKROOT} "${ICD_ROOT}/lib")
endif()
