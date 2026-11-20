#!/usr/bin/env bash
# The engine's toolchain environment: oneAPI 2026.2 with its own oneDNN (scripts/install-deps.sh installs both). Source
# it, do not run it:
#   source scripts/oneapi-env.sh
# XEN_ONEAPI=DIR picks another oneAPI (default /opt/intel/oneapi/2026.2); XEN_DNNL_ROOT=DIR another oneDNN install
# (unset = oneAPI's). Its lib dir goes FIRST in LD_LIBRARY_PATH: the binaries' RUNPATH yields to it, and oneapi-vars.sh
# puts oneAPI's libdnnl.so.3 (same soname) there. Changing it: delete build/CMakeCache.txt and rebuild.
case $- in *u*) _xen_nounset=1; set +u ;; *) _xen_nounset=0 ;; esac  # oneapi-vars.sh reads unset variables
_xen_oneapi=${XEN_ONEAPI:-/opt/intel/oneapi/2026.2}
_xen_dnnl=${XEN_DNNL_ROOT:-}
if [ -f "$_xen_oneapi/oneapi-vars.sh" ]; then
  . "$_xen_oneapi/oneapi-vars.sh" --force >/dev/null 2>&1
elif [ -f /opt/intel/oneapi/setvars.sh ]; then
  # component layout without the versioned umbrella dir (apt packages alone, e.g. the Docker image): each component's
  # newest version
  . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi
if [ -n "$_xen_dnnl" ] && [ -f "$_xen_dnnl/lib/libdnnl.so" ]; then
  export DNNLROOT=$_xen_dnnl
  export LD_LIBRARY_PATH=$_xen_dnnl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export CPATH=$_xen_dnnl/include${CPATH:+:$CPATH}
fi
[ "$_xen_nounset" = 1 ] && set -u
unset _xen_oneapi _xen_dnnl _xen_nounset
