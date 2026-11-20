// Xenithium · tensor element types as checkpoints name them (safetensors dtype strings).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace xenithium::model {

enum class DType : uint8_t { unknown, f32, f16, bf16, f8_e4m3, f8_e5m2, i64, i32, i16, i8, u8, boolean };

// Bytes per element; 0 for unknown.
constexpr size_t dtype_size(DType t) {
  switch (t) {
    case DType::f32: case DType::i32: return 4;
    case DType::f16: case DType::bf16: case DType::i16: return 2;
    case DType::f8_e4m3: case DType::f8_e5m2: case DType::i8: case DType::u8: case DType::boolean: return 1;
    case DType::i64: return 8;
    default: return 0;
  }
}

// safetensors spelling ("BF16", "F8_E4M3", "I32"...). Unknown strings map to DType:unknown.
DType dtype_from_string(std::string_view s);
std::string_view dtype_name(DType t);

}  // namespace xenithium:model
