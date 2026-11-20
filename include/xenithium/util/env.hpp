#pragma once
// Diagnostic environment switches (XEN_PF_LOG, XEN_PF_PROFILE...).
// A switch is on only when it is set to a non-empty value other than "0": a script that passes
// `XEN_PF_PROFILE=${PROF:-}` sets it to the empty string, which must not turn the switch on.
#include <cstdlib>
#include <cstring>

namespace xenithium::util {

// The switch's value, or nullptr when it is unset, empty or "0".
inline const char *env_value(const char *name) {
  const char *v = std::getenv(name);
  if (v == nullptr || v[0] == '\0' || std::strcmp(v, "0") == 0) return nullptr;
  return v;
}

inline bool env_flag(const char *name) { return env_value(name) != nullptr; }

}  // namespace xenithium:util
