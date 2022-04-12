cmake_minimum_required(VERSION 3.22.0)

project(vs-wnnm VERSION 0.1 LANGUAGES CXX)

set(VCL_HOME "" CACHE PATH "Path to vector class v2 headers")

find_package(MKL CONFIG REQUIRED)

add_library(wnnm SHARED source/wnnm.cpp)

target_include_directories(wnnm PRIVATE ${VCL_HOME})

target_link_libraries(wnnm PRIVATE $<LINK_ONLY:MKL::MKL>)

set_target_properties(wnnm PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

find_package(PkgConfig QUIET MODULE)

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)

    if(VS_FOUND)
        message(STATUS "Found VapourSynth r${VS_VERSION}")

        cmake_path(APPEND install_dir ${VS_LIBDIR} vapoursynth)
        target_include_directories(wnnm PRIVATE ${VS_INCLUDE_DIRS})

        install(TARGETS wnnm LIBRARY DESTINATION ${install_dir})
    endif()
endif()

if(NOT VS_FOUND)
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")

    if(VS_INCLUDE_DIR STREQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    target_include_directories(wnnm PRIVATE ${VS_INCLUDE_DIR})

    install(TARGETS wnnm LIBRARY RUNTIME)
endif()

find_package(Git QUIET)

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    if(VCS_TAG)
        string(STRIP ${VCS_TAG} VCS_TAG)
    endif()
endif()

if(VCS_TAG)
    message(STATUS "vs-nlm-ispc ${VCS_TAG}")
else()
    message(WARNING "unknown plugin version")
    set(VCS_TAG "unknown")
endif()

configure_file(source/config.h.in config.h)

target_include_directories(wnnm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
