# libkdtree++

Header-only **k-dimensional tree** for C++20: spatial ordering, nearest-neighbor search, and axis-aligned range queries. This tree is derived from **libkdtree++** (Martin F. Krafft, Sylvain Bougerel, Paul Harris) with a modernized API and internals.

## Requirements

- **C++20** or later
- CMake **3.20+** (for the bundled test target)

## Version

- CMake package version: **0.7.1**
- Macros in `kdtree.hpp`: `KDTREE_VERSION` (701), `KDTREE_LIB_VERSION` (`"0_7_1"`)

## Usage

Include the umbrella header and link nothing (header-only):

```cpp
#include "kdtree.hpp"  // adjust path to this directory

libkdtree::KDTree<2, MyPoint> tree;
tree.insert(point);
auto it = tree.findNearest(query);
```

### Main types

| Type | Role |
|------|------|
| `libkdtree::KDTree<K, Val, Acc, Dist, Cmp, Alloc>` | K-dimensional tree |
| `libkdtree::BracketAccessor<Val>` | Default accessor: `val[dim]` |
| `libkdtree::SquaredDifference<…>` | Default per-axis squared difference (Euclidean-style nearest) |
| `libkdtree::KdTreeRegion<…>` | Axis-aligned box for range queries |
| `libkdtree::TreeIterator<…>` | Bidirectional iterator over stored values |

### Typical operations

- **Insert / clear**: `insert`, `clear`, range constructors, `assignOptimized` (bulk rebuild using a vector in place)
- **Lookup**: `find` (by coordinates), `findExact` (by `operator==` on value)
- **Nearest**: `findNearest`, optional max radius, `findNearestIf` with a predicate
- **Range**: `countWithinRange`, `visitWithinRange`, `findWithinRange` (L∞-style boxes in accessor space by default)
- **Erase**: `erase` (iterator or coordinate-equivalent value), `eraseExact`
- **Maintenance**: `optimize` (rebuild balanced from sorted order)

Accessors on the tree: `valueAccessor`, `valueComparator`, `valueDistance`, `getAllocator`, `maxSize`.

## CMake

**Standalone** (tests on by default; fetches GoogleTest):

```bash
cmake -S . -B build -DLIBKDTREE_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

Disable tests:

```bash
cmake -S . -B build -DLIBKDTREE_BUILD_TESTS=OFF
```

**In another project** (`add_subdirectory`):

```cmake
add_subdirectory(path/to/libkdtree)
target_link_libraries(your_target PRIVATE libkdtree::libkdtree)
```

The interface target exports include directories for the libkdtree source directory.

## Tests

Unit tests live in `tests/test.cpp` (GoogleTest). With `LIBKDTREE_BUILD_TESTS=ON`, the executable `libkdtree_tests` is registered with CTest (label `libkdtree`).

## Optional compile-time defines

| Define | Effect |
|--------|--------|
| `KDTREE_DEFINE_OSTREAM_OPERATORS` | `operator<<` for nodes and trees (adds `<ostream>` / `<stack>` usage) |
| `KDTREE_CHECK_PERFORMANCE` | Global distance-call counter (`numDistCalcs`) when using instrumented functors |
| `KDTREE_CHECK_PERFORMANCE_COUNTERS` | Extra profiling hooks in the tree |

## Documentation

A **Doxygen** configuration is provided (`Doxyfile`). Run `doxygen` from this directory to generate API docs from the headers.

## License

Original libkdtree++ is distributed under the **Artistic License 2.0**. See the copyright blocks in the source files and any `COPYING` file shipped with the upstream distribution.

## Files

| File | Contents |
|------|----------|
| `kdtree.hpp` | `KDTree` public interface |
| `node.hpp` | `TreeNode`, `NodeBase`, nearest-neighbor helpers |
| `region.hpp` | `KdTreeRegion` |
| `iterator.hpp` | `TreeIterator` |
| `allocator.hpp` | `AllocBase`, RAII node allocation |
| `function.hpp` | `BracketAccessor`, `SquaredDifference`, helpers |
