#pragma once
// PRIVATE header -- must stay under src/, NOT include/: CMake ships include/
// wholesale via install(DIRECTORY), so anything there becomes public API next to
// the stable C FFI surface (peacock_gpu.h).
//
// Scaffolding shared by more than one TU: the `fb` namespace alias and the
// debug/trace facility.

#include "generated/gpu_plan_generated.h"

#include <cstdio>

namespace peacock {

// Declared in exactly ONE place. Redeclaring it at a different scope (global in
// one header, peacock:: in another) makes every `fb::` use site ambiguous.
namespace fb = peacock::plan;

// DEFINED ONCE, in expr.cpp. Deliberately NOT `static inline`: that would give
// every TU its own Meyers singleton and re-run getenv per TU.
bool debug_enabled();

// No-op unless debug is on; otherwise syncs the default stream and checks for
// errors, to localize async CUDA faults. Defined once, in expr.cpp.
void debug_sync(const char* tag);

// A macro, so it cannot cross translation units -- hence this shared header
// rather than a copy in every .cpp that traces.
#define PCK_TRACE(...) do {                                  \
    if (debug_enabled()) {                                   \
      std::fprintf(stderr, "[peacock] " __VA_ARGS__);        \
      std::fprintf(stderr, "\n");                            \
    }                                                        \
  } while (0)

}  // namespace peacock
