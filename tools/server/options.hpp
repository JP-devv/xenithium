// xenithium-server's command line: the flag table, the shared validation and --help (serve:Options holds the values).
#pragma once

#include "xenithium/serve/options.hpp"

namespace xenithium::server_cli {

// Parses argv into `o. Returns -1 to go on, else the exit code (0 after --help, 2 on a bad option; the message is
// printed).
int parse(int argc, char **argv, serve::Options &o);

}  // namespace xenithium:server_cli
