// Routing counts on disk (XRC1): see model/route_counts.hpp.
#include "xenithium/model/route_counts.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace xenithium::model {

std::vector<double> read_route_counts(const std::string &path, int layers, int experts) {
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0;
  int32_t l = 0, e = 0;
  if (!f || !f.read(reinterpret_cast<char *>(&magic), 4) || !f.read(reinterpret_cast<char *>(&l), 4) || !f.read(reinterpret_cast<char *>(&e), 4) ||
      magic != 0x31435258u)
    throw std::runtime_error("routing counts: cannot read " + path);
  if (l != layers || e != experts) throw std::runtime_error("routing counts: " + path + " is for another model shape");
  std::vector<double> c(static_cast<size_t>(l) * e);
  if (!f.read(reinterpret_cast<char *>(c.data()), static_cast<std::streamsize>(c.size() * 8))) throw std::runtime_error("routing counts: short file " + path);
  return c;
}

void write_route_counts(const std::string &path, int layers, int experts, const std::vector<double> &counts) {
  std::ofstream f(path, std::ios::binary);
  const uint32_t magic = 0x31435258u;
  const int32_t l = layers, e = experts;
  f.write(reinterpret_cast<const char *>(&magic), 4);
  f.write(reinterpret_cast<const char *>(&l), 4);
  f.write(reinterpret_cast<const char *>(&e), 4);
  f.write(reinterpret_cast<const char *>(counts.data()), static_cast<std::streamsize>(counts.size() * 8));
  if (!f) throw std::runtime_error("routing counts: cannot write " + path);
}

}  // namespace xenithium:model
