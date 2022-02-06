set(SDK_ROOT "${CMAKE_CURRENT_BINARY_DIR}/sdk_install")

set(CMAKE_LIST_CONTENT "
  cmake_minimum_required(VERSION 3.5)
  project(opencl_sdk)
  include(ExternalProject)

  ExternalProject_add(  opencl_sdk
    GIT_REPOSITORY      https://github.com/KhronosGroup/OpenCL-SDK
    GIT_TAG             main
    GIT_SHALLOW         1
    GIT_PROGRESS        1
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${SDK_ROOT}
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_DOCS=OFF
      -DBUILD_EXAMPLES=OFF
      -DBUILD_TESTS=OFF
      -DOPENCL_SDK_BUILD_SAMPLES=OFF
      -DOPENCL_SDK_TEST_SAMPLES=OFF
  )
")

set(DEPS "${CMAKE_CURRENT_BINARY_DIR}/opencl_sdk")
set(DEPS_BUILD "${DEPS}/build")
file(MAKE_DIRECTORY ${DEPS} ${DEPS_BUILD})
file(WRITE "${DEPS}/CMakeLists.txt" "${CMAKE_LIST_CONTENT}")
execute_process(WORKING_DIRECTORY "${DEPS_BUILD}" COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} ..)
execute_process(WORKING_DIRECTORY "${DEPS_BUILD}" COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE})

set(ENV{OCL_ROOT} "${SDK_ROOT}")

if(UNIX)
  set(ENV{AMDAPPSDKROOT} "${SDK_ROOT}")
elseif(WIN32)
  set(ENV{AMDAPPSDKROOT} "${SDK_ROOT}/lib")
endif()
