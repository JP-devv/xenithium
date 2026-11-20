cmake_minimum_required(VERSION 3.28)
project(xenithium VERSION 0.0.1 LANGUAGES CXX)

# Xenithium: native C++20 inference engine for Intel Arc Pro B-series (Battlemage, Xe2).
# SYCL (icpx, AOT for bmg) + ESIMD for kernels, Level Zero directly for memory and queues, no oneCCL.
# Layout (docs/architecture.md): src/ the engine's libraries, tools/ the executables, tests/ the gates; model-specific
# code lives in src/arch/<id>/ only.

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  message(FATAL_ERROR "xenithium needs icpx (oneAPI DPC++), got ${CMAKE_CXX_COMPILER_ID}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# No linker dependency files: under -fsycl the driver links temporary device objects ($TMPDIR/icpx-*/...o) that
# the linker records in link.d and then deletes, so every SYCL target looked out of date and every build (even a
# no-op) relinked all of them, AOT device compile included (~55 s, 5 CPU-minutes).
set(CMAKE_LINK_DEPENDS_USE_LINKER OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# Every executable and the 256-GRF kernel library directly in build/ (scripts run ./build/xenithium-*; the binaries'
# RUNPATH finds libxenithium_kernels_xl.so there).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(XENITHIUM_SYCL_TARGET "spir64_gen" CACHE STRING "SYCL AOT target")
set(XENITHIUM_SYCL_DEVICE "bmg" CACHE STRING "AOT device (bmg = Battlemage)")
# The device compiler (ocloc / IGC) prints at every link of device code a remark per loop it does not vectorise, two
# module-link warnings per device image and its per-kernel notes: about 37,000 lines per build that bury the C++
# compiler's own warnings. Link commands run through a filter that drops exactly those lines (errors and everything
# else pass; the command and its exit status are unchanged). ON shows them, e.g. to read a kernel's register spills.
option(XENITHIUM_DEVICE_COMPILER_NOTES "Print the device compiler's remarks and per-kernel notes at link time" OFF)
if(NOT XENITHIUM_DEVICE_COMPILER_NOTES)
  set(CMAKE_CXX_LINKER_LAUNCHER ${PROJECT_SOURCE_DIR}/scripts/quiet-device-link.sh)
endif()
option(XENITHIUM_BUILD_TESTS "Build the gates" ON)
# Device-code AOT at link time runs one backend compile per split module; the jobs run them in parallel (the same
# device images, a much shorter link).
set(XENITHIUM_LINK_JOBS "16" CACHE STRING "parallel device-code compiles per SYCL link")

find_package(Threads REQUIRED)
find_package(LevelZero CONFIG QUIET)

include(cmake/Xenithium.cmake)

add_subdirectory(src)
add_subdirectory(tools)
if(XENITHIUM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
