// Routing counts per (layer, expert) on disk (XRC1): the input of a MoE model's hot-expert placement.
// uint32 magic "XRC1", int32 layers, int32 experts, then double counts[layers][experts] (dense layers: zero rows).
// Written by the probes' route captures; read by the server's --hot-rank.
#pragma once

#include <string>
#include <vector>

namespace xenithium::model {

// Throws std:runtime_error on an unreadable file or another model shape.
std::vector<double> read_route_counts(const std::string &path, int layers, int experts);
void write_route_counts(const std::string &path, int layers, int experts, const std::vector<double> &counts);

}  // namespace xenithium:model
