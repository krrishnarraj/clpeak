
set(CLHPP_ROOT "${CMAKE_CURRENT_BINARY_DIR}/clhpp_install")

set(CMAKE_LIST_CONTENT "
  cmake_minimum_required(VERSION 3.5)
  include(ExternalProject)

  ExternalProject_add(  hpp_headers
    PREFIX              hpp
    GIT_REPOSITORY      https://github.com/KhronosGroup/OpenCL-CLHPP
    GIT_TAG             master
    GIT_SHALLOW         1
    GIT_PROGRESS        1
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CLHPP_ROOT}
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_DOCS=OFF
      -DBUILD_EXAMPLES=OFF
      -DBUILD_TESTS=OFF
  )
")

set(DEPS "${CMAKE_CURRENT_BINARY_DIR}/clhpp")
set(DEPS_BUILD "${DEPS}/build")
file(MAKE_DIRECTORY ${DEPS} ${DEPS_BUILD})
file(WRITE "${DEPS}/CMakeLists.txt" "${CMAKE_LIST_CONTENT}")
execute_process(WORKING_DIRECTORY "${DEPS_BUILD}" COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ..)
execute_process(WORKING_DIRECTORY "${DEPS_BUILD}" COMMAND ${CMAKE_COMMAND} --build .)

