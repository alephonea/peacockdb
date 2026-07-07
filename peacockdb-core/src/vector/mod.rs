//! Vector search: fp16 vector type + distance UDFs (CPU for now; GPU top-k lands
//! in later tickets). A vector column is an arrow `FixedSizeList<Float16, dim>`.

pub mod analyzer;
pub mod cpu;
pub mod exec;
pub mod logical;
pub mod optimizer;
pub mod planner;
pub mod types;
pub mod udf;

pub use analyzer::VectorTopKAnalyzerRule;
pub use exec::GpuVectorSearchExec;
pub use logical::VectorTopK;
pub use optimizer::PushFilterIntoVectorTopK;
pub use planner::{VectorQueryPlanner, VectorTopKPlanner};
pub use types::{is_vector_type, vector_dtype_for_dim, VectorMeta, VectorScalar};
pub use udf::{l2_distance_udf, to_vector_udf, L2Distance, ToVector};
