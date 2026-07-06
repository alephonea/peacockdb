//! Vector search: fp16 vector type + distance UDFs (CPU for now; GPU top-k lands
//! in later tickets). A vector column is an arrow `FixedSizeList<Float16, dim>`.

pub mod cpu;
pub mod types;
pub mod udf;

pub use types::{is_vector_type, vector_dtype_for_dim, VectorMeta, VectorScalar};
pub use udf::{l2_distance_udf, L2Distance};
