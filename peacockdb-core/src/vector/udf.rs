//! `l2_distance(a, b)` scalar UDF: exact Euclidean distance between two fp16
//! vectors (`FixedSizeList<Float16, N>`), returned as `Float32`. CPU-only; the
//! GPU top-k path comes in a later ticket.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, FixedSizeListArray, Float16Array, Float16Builder};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::{exec_err, DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use super::cpu::l2_distance_arrays;
use super::types::{is_vector_type, vector_dtype_for_dim, VectorScalar};

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

/// `to_vector(x0, x1, …, x_{n-1})` — a query-vector literal constructor: builds a
/// `FixedSizeList<Float16, n>` from `n` numeric arguments (each cast to fp16).
/// Variadic (rather than `to_vector([array])`) so the dimension `n` is the arg
/// count — known at plan time, which lets `return_type` produce the concrete
/// FixedSizeList type `l2_distance` requires. SQL can spell it directly, and
/// `to_vector(1,2,3,4)` folds to a constant vector (see the analyzer's const-eval).
#[derive(Debug)]
pub struct ToVector {
    signature: Signature,
}

impl ToVector {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl Default for ToVector {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for ToVector {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "to_vector"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.is_empty() {
            return exec_err!("to_vector needs at least one element");
        }
        // dim == arg count, so the FixedSizeList<Float16, dim> type is known here.
        Ok(vector_dtype_for_dim(arg_types.len() as u32, VectorScalar::F16))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rows = args.number_rows;
        let dim = args.args.len();
        if dim == 0 {
            return exec_err!("to_vector needs at least one element");
        }
        // One fp16 column per element position; cast whatever numeric type arrives.
        let cols: Vec<Float16Array> = args
            .args
            .into_iter()
            .map(|v| {
                let arr = v.into_array(rows)?;
                let f16 = cast(&arr, &DataType::Float16).map_err(DataFusionError::from)?;
                f16.as_any()
                    .downcast_ref::<Float16Array>()
                    .cloned()
                    .ok_or_else(|| {
                        DataFusionError::Execution("to_vector: cast to Float16 failed".to_string())
                    })
            })
            .collect::<Result<_>>()?;

        // Row-major child: row r's list is [col0[r], …, col_{dim-1}[r]].
        let mut child = Float16Builder::with_capacity(rows * dim);
        for r in 0..rows {
            for c in &cols {
                if c.is_null(r) {
                    child.append_null();
                } else {
                    child.append_value(c.value(r));
                }
            }
        }
        let field = Arc::new(Field::new("item", DataType::Float16, true));
        let fsl = FixedSizeListArray::new(field, dim as i32, Arc::new(child.finish()), None);
        Ok(ColumnarValue::Array(Arc::new(fsl)))
    }
}

/// The registrable `to_vector` UDF.
pub fn to_vector_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(ToVector::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_vector_builds_fixed_size_list_float16() {
        use crate::vector::{is_vector_type, VectorMeta};
        use datafusion::arrow::array::Float16Array;

        let udf = ToVector::new();
        // return_type: dim == arg count, element fp16.
        let rt = udf.return_type(&[DataType::Int64, DataType::Int64, DataType::Float64]).unwrap();
        assert_eq!(is_vector_type(&rt), Some(VectorMeta { dim: 3, scalar: VectorScalar::F16 }));
        assert!(udf.return_type(&[]).is_err());

        // invoke: numeric scalars -> a one-row FixedSizeList<Float16,4> [1,2,3,4].
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Scalar(datafusion::common::ScalarValue::Int64(Some(1))),
                ColumnarValue::Scalar(datafusion::common::ScalarValue::Int64(Some(2))),
                ColumnarValue::Scalar(datafusion::common::ScalarValue::Float64(Some(3.0))),
                ColumnarValue::Scalar(datafusion::common::ScalarValue::Float64(Some(4.0))),
            ],
            number_rows: 1,
            return_type: &vector_dtype_for_dim(4, VectorScalar::F16),
        };
        let out = udf.invoke_with_args(args).unwrap().into_array(1).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.len(), 1);
        assert_eq!(fsl.value_length(), 4);
        let vals = fsl.values().as_any().downcast_ref::<Float16Array>().unwrap();
        let got: Vec<f32> = (0..4).map(|i| vals.value(i).to_f32()).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

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
