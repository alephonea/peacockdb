#include "peacock/partitioning.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/unary.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

namespace peacock::partitioning {
namespace {

// --- Spark murmur3 (bit-identical to comet spark_compatible_murmur3_hash) ------
// All math in uint32_t: two's-complement wrapping mul/add and rotate match Rust's
// i32 mul_wrapping/add_wrapping/rotate_left bit-for-bit; xor-shifts use the
// unsigned shift comet casts to (`h1 as u32 >> k`).
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int r) {
  return (x << r) | (x >> (32 - r));
}
__device__ __forceinline__ uint32_t mix_k1(uint32_t k1) {
  k1 *= 0xcc9e2d51u;
  k1 = rotl32(k1, 15);
  k1 *= 0x1b873593u;
  return k1;
}
__device__ __forceinline__ uint32_t mix_h1(uint32_t h1, uint32_t k1) {
  h1 ^= k1;
  h1 = rotl32(h1, 13);
  h1 = h1 * 5u + 0xe6546b64u;
  return h1;
}
__device__ __forceinline__ uint32_t fmix32(uint32_t h1, uint32_t len) {
  h1 ^= len;
  h1 ^= h1 >> 16;
  h1 *= 0x85ebca6bu;
  h1 ^= h1 >> 13;
  h1 *= 0xc2b2ae35u;
  h1 ^= h1 >> 16;
  return h1;
}

// Hash a byte span with a running seed — matches comet: 4-byte LE chunks via
// mix_h1(mix_k1(int)), then EACH tail byte SIGN-EXTENDED (i8->i32) through
// mix_h1(mix_k1(byte)) (Spark's per-byte tail, not standard murmur3's), then fmix.
__device__ uint32_t spark_hash_bytes(char const* data, int len, uint32_t seed) {
  uint32_t h1          = seed;
  int const len_align4 = len - (len & 3);
  for (int i = 0; i < len_align4; i += 4) {
    uint32_t k1 = static_cast<uint32_t>(static_cast<uint8_t>(data[i])) |
                  (static_cast<uint32_t>(static_cast<uint8_t>(data[i + 1])) << 8) |
                  (static_cast<uint32_t>(static_cast<uint8_t>(data[i + 2])) << 16) |
                  (static_cast<uint32_t>(static_cast<uint8_t>(data[i + 3])) << 24);
    h1 = mix_h1(h1, mix_k1(k1));
  }
  for (int i = len_align4; i < len; ++i) {
    int32_t const b = static_cast<int32_t>(static_cast<int8_t>(data[i]));  // sign-extend
    h1              = mix_h1(h1, mix_k1(static_cast<uint32_t>(b)));
  }
  return fmix32(h1, static_cast<uint32_t>(len));
}

// One thread per row; folds one STRING key column into the running hash.
// Null rows are SKIPPED (no update) — matches comet's "no update for Null".
__global__ void spark_hash_string_col_kernel(cudf::column_device_view col,
                                             uint32_t* hashes,
                                             cudf::size_type n) {
  auto const row = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n) return;
  if (col.is_null(row)) return;
  auto const s   = col.element<cudf::string_view>(row);
  hashes[row]    = spark_hash_bytes(s.data(), s.size_bytes(), hashes[row]);
}

// One thread per row; folds one FIXED-WIDTH key column into the running hash.
// comet hashes `value.to_le_bytes()` of the natural-width representation through the
// SAME murmur3; on a little-endian GPU the value's in-memory bytes ARE its LE bytes,
// so hashing sizeof(T) bytes at &v is bit-identical to comet (Int32/Date-as-i32 → 4B,
// Int64/Timestamp-as-i64 → 8B — one/two 4-byte blocks, no tail since sizeof%4==0).
// Null rows are SKIPPED (running hash unchanged), matching comet's is_null skip.
template <typename T>
__global__ void spark_hash_fixed_col_kernel(cudf::column_device_view col,
                                            uint32_t* hashes,
                                            cudf::size_type n) {
  auto const row = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n) return;
  if (col.is_null(row)) return;
  T const v   = col.element<T>(row);
  hashes[row] = spark_hash_bytes(reinterpret_cast<char const*>(&v),
                                 static_cast<int>(sizeof(T)), hashes[row]);
}

// pmod (positive modulo), NOT raw % — negative hashes must wrap into [0, parts).
__global__ void pmod_kernel(uint32_t const* hashes,
                            int32_t* pid,
                            cudf::size_type n,
                            int32_t parts) {
  auto const row = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n) return;
  int32_t const h = static_cast<int32_t>(hashes[row]);
  pid[row]        = ((h % parts) + parts) % parts;
}

}  // namespace

std::unique_ptr<cudf::column> spark_partition_ids(cudf::table_view const& input,
                                                 std::vector<cudf::size_type> const& key_cols,
                                                 cudf::size_type num_partitions,
                                                 uint32_t seed,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(num_partitions > 0, "peacock spark_partition_ids: num_partitions must be > 0");
  auto const n = input.num_rows();

  rmm::device_uvector<uint32_t> hashes(n, stream, mr);
  thrust::fill(rmm::exec_policy(stream), hashes.begin(), hashes.end(), seed);

  constexpr int block = 256;
  auto const grid     = (n + block - 1) / block;
  // Decoded dictionary key columns are kept alive here until all hash kernels for this
  // call have been enqueued (their device views feed async kernels below).
  std::vector<std::unique_ptr<cudf::column>> decoded_keep;
  for (auto const ci : key_cols) {
    cudf::column_view col = input.column(ci);
    // (#18) Normalize a dict-encoded key to its STRING values FOR HASHING ONLY. A
    // dict-encoded parquet string arrives as cuDF DICTIONARY32, which no kernel
    // below handles; decoding yields the identical bytes the comet CPU path hashes
    // (it sees Utf8View→Utf8). The SCATTERED output keeps the ORIGINAL column.
    if (col.type().id() == cudf::type_id::DICTIONARY32) {
      decoded_keep.push_back(
          cudf::dictionary::decode(cudf::dictionary_column_view{col}, stream, mr));
      col = decoded_keep.back()->view();
    }
    // (#18) Normalize small-int / date families to INT32 so the 4-byte fixed kernel
    // hashes Spark-identical bytes. Spark widens sub-32-bit ints to i32, so INT8/16
    // need a VALUE cast; TIMESTAMP_DAYS is already int32 days-since-epoch (comet's
    // DATE32 -> i32), so a zero-copy BIT-cast suffices. Hash-only, as above.
    switch (col.type().id()) {
      case cudf::type_id::INT8:
      case cudf::type_id::INT16:
        decoded_keep.push_back(
            cudf::cast(col, cudf::data_type{cudf::type_id::INT32}, stream, mr));
        col = decoded_keep.back()->view();
        break;
      case cudf::type_id::TIMESTAMP_DAYS:
        col = cudf::bit_cast(col, cudf::data_type{cudf::type_id::INT32});
        break;
      default:
        break;
    }
    auto const dcol = cudf::column_device_view::create(col, stream);
    if (n > 0) {
      // Dispatch by cuDF type id. Each column folds into the running (seed-chained)
      // hash in key order, so composite keys work for free. STRING + INT32/INT64
      // only; timestamp/decimal/float keys are pending (#18) and fail loudly below
      // rather than hash a wrong encoding.
      switch (col.type().id()) {
        case cudf::type_id::STRING:
          spark_hash_string_col_kernel<<<grid, block, 0, stream.value()>>>(
              *dcol, hashes.data(), n);
          break;
        case cudf::type_id::INT32:
          spark_hash_fixed_col_kernel<int32_t><<<grid, block, 0, stream.value()>>>(
              *dcol, hashes.data(), n);
          break;
        case cudf::type_id::INT64:
          spark_hash_fixed_col_kernel<int64_t><<<grid, block, 0, stream.value()>>>(
              *dcol, hashes.data(), n);
          break;
        default:
          // Print the exact cuDF type_id: the DataFusion-plan type (e.g. Utf8View)
          // is only a proxy for what actually reaches this kernel.
          CUDF_FAIL(
              "peacock spark_partition_ids: unsupported key column cuDF type_id=" +
              std::to_string(static_cast<int>(col.type().id())) +
              " (supported: STRING, dict-encoded string, INT8/16/32/64, DATE32; "
              "timestamp/decimal/float partition keys pending — extend the kernel + "
              "re-prove comet conformance, see #18/Inc7)");
      }
    }
  }

  auto pid = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, n,
                                       cudf::mask_state::UNALLOCATED, stream, mr);
  if (n > 0) {
    pmod_kernel<<<grid, block, 0, stream.value()>>>(
        hashes.data(), pid->mutable_view().data<int32_t>(), n, num_partitions);
  }
  return pid;
}

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> spark_hash_partition(
    cudf::table_view const& input,
    std::vector<cudf::size_type> const& key_cols,
    cudf::size_type num_partitions,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const pid = spark_partition_ids(input, key_cols, num_partitions, seed, stream, mr);
  return cudf::partition(input, pid->view(), num_partitions, stream, mr);
}

}  // namespace peacock::partitioning
