#!/usr/bin/env bash
# The build's link launcher (CMAKE_CXX_LINKER_LAUNCHER, unless -DXENITHIUM_DEVICE_COMPILER_NOTES=ON): runs the link
# command it is given and drops the device compiler's notes from its output, nothing else. Those notes are printed at
# every link of device code and no source change removes them:
#   [bmg] remark: .. loop not vectorized ..                     one per loop of every kernel
#   [bmg] warning: Linking two modules of different ..          two per device image
#   [bmg] warning: in kernel .. / undefined value ..            the per-kernel notes (register spills, recompilations)
#   Spill memory used .. / Compiling kernel with spill code ..  the same notes from the code generator
#   Compilation from IR .. / Build succeeded.                   its progress lines
# Errors ("[bmg] error: ..", the linker's own messages) pass through, and the exit status is the link command's.
"$@" 2>&1 | grep -v -E \
  -e '^\[[A-Za-z0-9_]+\] remark: ' \
  -e '^\[[A-Za-z0-9_]+\] warning: Linking two modules of different (target triples|data layouts): ' \
  -e '^\[[A-Za-z0-9_]+\] warning: in kernel ' \
  -e '^\[[A-Za-z0-9_]+\] warning:  undefined value is referenced after decomposition$' \
  -e '^\[[A-Za-z0-9_]+\] in function .*PrintVerboseGenericControlFlowLog' \
  -e '^Spill memory used = [0-9]+ bytes for kernel ' \
  -e '^ Compiling kernel with spill code may degrade performance\.' \
  -e '^Compilation from IR - skipping loading of FCL$' \
  -e '^Build succeeded\.$' \
  -e '^$' >&2
exit "${PIPESTATUS[0]}"
