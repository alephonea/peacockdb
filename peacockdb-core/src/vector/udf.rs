//! `l2_distance(a, b)` scalar UDF: exact Euclidean distance between two fp16
//! vectors (`FixedSizeList<Float16, N>`), returned as `Float32`. CPU-only; the
//! GPU top-k path comes in a later ticket.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, FixedSizeListArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{exec_err, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use super::cpu::l2_distance_arrays;
use super::types::is_vector_type;

#[derive(Debug)]
pub struct L2Distance {
    signature: Signature,
}

impl L2Distance {
    pub fn new() -> Self {
        // Two args of any type: the concrete element/dim of a vector is a
        // FixedSizeList<Float16, N> whose N is data-dependent and not expressible
        // as a fixed Signature, so we accept any pair and validate the vector
        // shape in `return_type` (plan time) and `invoke_with_args` (run time).
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for L2Distance {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for L2Distance {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "l2_distance"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [a, b] = arg_types else {
            return exec_err!("l2_distance expects 2 arguments, got {}", arg_types.len());
        };
        let (Some(a), Some(b)) = (is_vector_type(a), is_vector_type(b)) else {
            return exec_err!(
                "l2_distance arguments must be fp16 vectors (FixedSizeList<Float16, N>), got {a:?} and {b:?}"
            );
        };
        if a.dim != b.dim {
            return exec_err!("l2_distance dimension mismatch: {} vs {}", a.dim, b.dim);
        }
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let arrays: Vec<ArrayRef> = args
            .args
            .into_iter()
            .map(|v| v.into_array(rows))
            .collect::<Result<_>>()?;
        let [a, b] = arrays.as_slice() else {
            return exec_err!("l2_distance expects 2 arguments");
        };
        let (Some(a), Some(b)) = (
            a.as_any().downcast_ref::<FixedSizeListArray>(),
            b.as_any().downcast_ref::<FixedSizeListArray>(),
        ) else {
            return exec_err!("l2_distance arguments must be FixedSizeList vectors");
        };
        Ok(ColumnarValue::Array(Arc::new(l2_distance_arrays(a, b)?)))
    }
}

/// The registrable `l2_distance` UDF.
pub fn l2_distance_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(L2Distance::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_type_is_float32_for_matching_vectors() {
        use crate::vector::{vector_dtype_for_dim, VectorScalar};
        let udf = L2Distance::new();
        let v4 = vector_dtype_for_dim(4, VectorScalar::F16);
        assert_eq!(
            udf.return_type(&[v4.clone(), v4.clone()]).unwrap(),
            DataType::Float32
        );
        // Dim mismatch and non-vector args are plan-time errors.
        let v8 = vector_dtype_for_dim(8, VectorScalar::F16);
        assert!(udf.return_type(&[v4.clone(), v8]).is_err());
        assert!(udf.return_type(&[v4, DataType::Float32]).is_err());
    }
}
