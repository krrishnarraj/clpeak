add_library(peak_common STATIC
    peak.cpp
    common.cpp
    console_mute.cpp
    dynlib.cpp
    inventory.cpp
    logger.cpp
    logger_text.cpp
    options.cpp
    host_info.cpp
    json.cpp
    keep_awake.cpp
    run_document.cpp
    run_log.cpp
    units.cpp
)
# dlopen lives in libdl on older glibc; ${CMAKE_DL_LIBS} is empty where it's in libc.
target_link_libraries(peak_common PUBLIC ${CMAKE_DL_LIBS})
# Core ML's compile cache has to be cleared by whichever backend fills it
# (include/common/coreml_cache.h); the Objective-C++ implementation is Apple
# only and the header is a no-op stub elsewhere.
if(APPLE)
    target_sources(peak_common PRIVATE coreml_cache.mm)
    set_source_files_properties(coreml_cache.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    find_library(COMMON_FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(peak_common PUBLIC ${COMMON_FOUNDATION_FRAMEWORK})
else()
    target_sources(peak_common PRIVATE coreml_cache_stub.cpp)
endif()
# A run holds the machine out of idle sleep (include/common/keep_awake.h):
# Windows asks powrprof whether this is a Modern Standby machine; macOS takes
# an IOKit power assertion, which iOS does not offer (the app holds the screen
# there instead).  Linux reaches logind and the desktop through a libsystemd
# loaded at run time, so it links nothing.
if(WIN32)
    target_link_libraries(peak_common PUBLIC powrprof)
elseif(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    find_library(COMMON_IOKIT_FRAMEWORK IOKit)
    find_library(COMMON_COREFOUNDATION_FRAMEWORK CoreFoundation)
    target_link_libraries(peak_common PUBLIC
        ${COMMON_IOKIT_FRAMEWORK} ${COMMON_COREFOUNDATION_FRAMEWORK})
endif()
# logger.cpp (base) + logger_text.cpp (shared text formatter, writes to an
# injectable std::ostream) live here.  Consumers that need a non-text channel
# (Android JNI, iOS callbacks) subclass these in their own tree.
target_include_directories(peak_common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../include/common
)
if(MSVC)
    # Covers clang-cl too: -fPIC is meaningless on Windows (warns), and
    # clang-cl aliases -Wall to MSVC /Wall (≈ -Weverything spam), so route the
    # GNU-style warnings through the /clang: passthrough instead.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(peak_common PRIVATE
            /clang:-Wall /clang:-Wextra /clang:-Wno-unused-parameter)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
    target_compile_options(peak_common PRIVATE -fPIC -Wall -Wextra -Wno-unused-parameter)
endif()

# version.h is generated once, at configure time
include(${CMAKE_CURRENT_LIST_DIR}/cmake/version.cmake)
clpeak_setup_version(peak_common)
