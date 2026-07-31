#pragma once
// PRIVATE header -- deliberately under src/, NOT include/. CMakeLists ships
// include/ wholesale via install(DIRECTORY include/), so anything placed there
// becomes public API sitting next to the stable C FFI surface (peacock_gpu.h).
// These are internal executor guts and are not part of that contract.
//
// Shared scaffolding needed by more than one TU after the Inc4b split: the `fb`
// namespace alias and the debug/trace facility.

#include "generated/gpu_plan_generated.h"

#include <cstdio>

namespace peacock {

// Declared in exactly ONE place. Redeclaring it at a different scope (global in
// one header, peacock:: in another) makes every `fb::` use site ambiguous.
namespace fb = peacock::plan;

// Declared here, DEFINED ONCE (expr.cpp). Deliberately NOT `static inline`: that
// would give every TU its own Meyers singleton and re-run getenv per TU. Same
// observable behaviour today, but it is exactly the per-TU duplication this split
// has to be careful about -- see the thread_local hazard Inc4a removed.
bool debug_enabled();

// Synchronize the default stream and check for errors. When debug is on we always
// sync (to localize async CUDA faults); when off this is a no-op. Defined once.
void debug_sync(const char* tag);

// A macro, so it does not cross translation units -- it has to live in this shared
// header rather than be duplicated into every .cpp that traces.
#define PCK_TRACE(...) do {                                  \
    if (debug_enabled()) {                                   \
      std::fprintf(stderr, "[peacock] " __VA_ARGS__);        \
      std::fprintf(stderr, "\n");                            \
    }                                                        \
  } while (0)

}  // namespace peacock
