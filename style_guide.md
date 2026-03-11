# C++ Style Guide

This guide applies to all C++ code in PeacockDB, primarily in `libpeacock_gpu` and the FFI boundary layer.

---

## Formatting

### Indentation
- **2 spaces** per indent level. No tabs.

### Namespaces
- Do **not** indent namespace contents.
- Add a comment after the closing brace.

```cpp
namespace peacock {
namespace gpu {

class executor {
  // ...
};

} // namespace gpu
} // namespace peacock
```

### Braces
- Opening brace on the same line (K&R style), except for namespace definitions.
- Always use braces for control flow, even single-line bodies.

```cpp
if (condition) {
  do_something();
}
```

### Line length
- Soft limit of 100 characters.

### Pointers and references
- Attach `*` and `&` to the type, not the name.

```cpp
int* ptr;
const buffer& buf;
```

---

## Naming

All names use **snake_case**.

| Entity | Convention | Example |
|---|---|---|
| Files | `snake_case.cpp` / `.h` | `gpu_executor.cpp` |
| Types (classes, structs, enums) | `snake_case` | `physical_plan`, `chunk_iterator` |
| Functions and methods | `snake_case` | `execute_plan()` |
| Variables | `snake_case` | `row_count` |
| Constants and enumerators | `snake_case` | `max_chunk_size` |
| Macros | `UPPER_SNAKE_CASE` | `PEACOCK_CHECK_CUDA` |
| Namespaces | `snake_case` | `peacock::gpu` |
| Private member variables | trailing `_` | `device_memory_` |
| Template parameters | `PascalCase` | `template <typename T>` |

---

## Headers

- Use `#pragma once` (not include guards).
- Order of includes:
  1. Corresponding `.h` for the `.cpp`
  2. Other project headers
  3. Third-party headers (`cudf/`, `rmm/`)
  4. System / standard library headers

Separate each group with a blank line.

```cpp
#include "gpu_executor.h"

#include "physical_plan.h"
#include "chunk_iterator.h"

#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <vector>
```

---

## Types

- Prefer `int32_t`, `uint64_t`, etc. over `int`, `long` for types crossing FFI boundaries.
- Use `std::size_t` for sizes and counts.
- Prefer `std::string_view` over `const std::string&` for read-only string parameters.

---

## Functions

- Prefer returning values over output parameters.
- Do not use non-const reference parameters. Input parameters are passed by value or `const&`; output and input/output parameters are passed by pointer.
- Use `[[nodiscard]]` on functions whose return value must not be ignored (especially CUDA/cuDF status codes and resource handles).
- Keep functions short and focused on a single responsibility.

```cpp
// Good: input by const&, output by pointer, result by return value
[[nodiscard]] std::unique_ptr<cudf::table> execute_plan(
  const physical_plan& plan,
  execution_stats* out_stats);

// Bad: non-const reference hides mutation at call site
void execute_plan(const physical_plan& plan, execution_stats& out_stats);
```

---

## Error Handling

- No exceptions across the FFI boundary. Use error codes or a result struct.
- Within C++-only code, use exceptions or assertions as appropriate.
- CUDA errors must be checked immediately. Use a helper macro:

```cpp
#define PEACOCK_CHECK_CUDA(expr) \
  do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
      throw std::runtime_error(cudaGetErrorString(_err)); \
    } \
  } while (0)
```

---

## Memory

- Prefer `rmm::device_buffer` and `rmm::device_uvector` over raw `cudaMalloc`.
- Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) for host-side ownership.
- Clearly document ownership transfer at FFI boundaries with comments.

---

## Classes

- Public interface first, then protected, then private.
- Rule of zero or rule of five — do not mix.
- Mark single-argument constructors `explicit` unless implicit conversion is intentional.

```cpp
class gpu_executor {
public:
  explicit gpu_executor(rmm::cuda_stream_view stream);

  [[nodiscard]] execution_result execute(const physical_plan& plan);

private:
  rmm::cuda_stream_view stream_;
  std::size_t memory_limit_;
};
```

---

## Comments

- Use `//` for all comments. Reserve `/* */` for temporarily disabling code.
- Inside function body, use comments sparingly. Only comment non-obvious code.
- Document non-obvious GPU constraints inline (e.g., memory budget assumptions, kernel launch limits).

---

## Miscellaneous

- Prefer `nullptr` over `NULL` or `0` for pointers.
- Prefer range-based `for` loops over index-based when the index is not needed.
- Do not use `using namespace`.
- Mark overrides with `override`, never with `virtual` alone.
