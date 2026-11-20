# Xenithium build helpers: the two kinds of targets (SYCL device code, host-only code) and the one-line forms for
# tools and tests. Every target in the tree is declared through these, so a flag change lands everywhere at once.

# AOT: no JIT at start-up. The backend arguments are separate argv entries; one string with an embedded space is
# re-split by the Makefile generator and swallows the next flag.
set(XENITHIUM_SYCL_FLAGS
    -fsycl
    "-fsycl-targets=${XENITHIUM_SYCL_TARGET}"
    "-Xsycl-target-backend=${XENITHIUM_SYCL_TARGET}"
    "-device ${XENITHIUM_SYCL_DEVICE}"
    -Wall -Wextra -Wno-unused-parameter)

# A target that includes a SYCL / Level Zero header. Every such target needs this itself: linking a SYCL library does
# not propagate the flags.
function(xenithium_sycl target)
  target_compile_options(${target} PRIVATE ${XENITHIUM_SYCL_FLAGS})
  target_link_options(${target} PRIVATE ${XENITHIUM_SYCL_FLAGS} -fsycl-max-parallel-link-jobs=${XENITHIUM_LINK_JOBS})
  target_include_directories(${target} PUBLIC ${PROJECT_SOURCE_DIR}/include)
  if(LevelZero_FOUND)
    target_link_libraries(${target} PRIVATE LevelZero::ze_loader)
  else()
    target_link_libraries(${target} PRIVATE ze_loader)
  endif()
endfunction()

# Host-only code: no -fsycl. icpx defaults to fast math for host code too, so precise floating point is requested
# explicitly (host references must round like IEEE).
function(xenithium_host target)
  target_include_directories(${target} PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(${target} PRIVATE -O2 -Wall -Wextra -fp-model=precise)
endfunction()

# xenithium_tool(NAME [HOST] SOURCES f.cpp ... LIBS lib ...): an executable in build/. SYCL unless HOST.
function(xenithium_tool name)
  cmake_parse_arguments(T "HOST" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  if(T_HOST)
    xenithium_host(${name})
  else()
    xenithium_sycl(${name})
  endif()
  target_link_libraries(${name} PRIVATE ${T_LIBS})
endfunction()

# xenithium_test(NAME [GPU] [TIMEOUT s] [SOURCE f.cpp] LIBS lib ...): the gate binary xenithium-test-NAME (source
# tests/test_<NAME with - as _>.cpp unless SOURCE) registered with ctest under the label `host` or `gpu`
# (scripts/test.sh runs `gpu` only when the cards are free). GPU tests are SYCL targets; host tests are host-only.
function(xenithium_test name)
  cmake_parse_arguments(T "GPU" "TIMEOUT;SOURCE" "LIBS" ${ARGN})
  string(REPLACE "-" "_" stem ${name})
  if(NOT T_SOURCE)
    set(T_SOURCE test_${stem}.cpp)
  endif()
  if(NOT T_TIMEOUT)
    set(T_TIMEOUT 300)
  endif()
  set(target xenithium-test-${name})
  add_executable(${target} ${T_SOURCE})
  if(T_GPU)
    xenithium_sycl(${target})
    set(label gpu)
  else()
    xenithium_host(${target})
    set(label host)
  endif()
  target_link_libraries(${target} PRIVATE ${T_LIBS})
  # from the repository root, as scripts/test.sh always ran them (tests read tests/data/... relative to it)
  add_test(NAME ${name} COMMAND ${target} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(${name} PROPERTIES LABELS ${label} TIMEOUT ${T_TIMEOUT})
endfunction()

# Per-arch source lists (src/arch/<id>/sources.cmake): xenithium_arch_sources(HOST ... DEVICE ... SERVE ...).
#   HOST    host-only model code (config, tensor roles, budget, placement data): xenithium_model
#   DEVICE  code that touches the cards (loaders, the model's step): xenithium_arch
#   SERVE   the core::StepModel adapter and the serving backend: xenithium_gpu_engine
# Paths are relative to the arch directory.
macro(xenithium_arch_sources)
  cmake_parse_arguments(A "" "" "HOST;DEVICE;SERVE" ${ARGN})
  get_filename_component(_xen_arch_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
  foreach(kind HOST DEVICE SERVE)
    foreach(f IN LISTS A_${kind})
      list(APPEND XENITHIUM_ARCH_${kind} ${_xen_arch_dir}/${f})
    endforeach()
  endforeach()
endmacro()

# Per-family kernel source lists (src/kernels/CMakeLists.txt): xenithium_kernel_family(DIR SOURCES ... XL ... REF ...).
#   SOURCES  device code at 128 GRF: xenithium_kernels (static)
#   XL       device code that spills at 128 GRF, compiled at 256 (-doubleGRF): xenithium_kernels_xl (shared)
#   REF      exact host references, -fp-model=precise: xenithium_pack
macro(xenithium_kernel_family dir)
  cmake_parse_arguments(K "" "" "SOURCES;XL;REF" ${ARGN})
  foreach(kind SOURCES XL REF)
    foreach(f IN LISTS K_${kind})
      list(APPEND XENITHIUM_KERNELS_${kind} ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${f})
    endforeach()
  endforeach()
endmacro()
