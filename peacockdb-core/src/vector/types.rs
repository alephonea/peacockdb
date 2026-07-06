//! The logical vector type and its mapping to an arrow `DataType`.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};

/// Scalar element type of a vector column. Only fp16 for the MVP; cosine/IP and
/// wider element types come later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorScalar {
    F16,
}

impl VectorScalar {
    /// The arrow element `DataType` this scalar maps to.
    pub fn arrow_dtype(self) -> DataType {
        match self {
            VectorScalar::F16 => DataType::Float16,
        }
    }
}

/// Logical description of a vector column: element type + fixed dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorMeta {
    pub dim: u32,
    pub scalar: VectorScalar,
}

/// Child-field name arrow uses inside a `FixedSizeList`. Its exact value is
/// irrelevant to `is_vector_type` (which ignores the name), but we keep it stable.
const ITEM_FIELD: &str = "item";

/// Arrow type for a `dim`-dimensional vector of `scalar`: a
/// `FixedSizeList<scalar, dim>`. The child field is nullable (arrow's default),
/// though vector elements are not expected to be null in practice.
pub fn vector_dtype_for_dim(dim: u32, scalar: VectorScalar) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new(ITEM_FIELD, scalar.arrow_dtype(), true)),
        dim as i32,
    )
}

/// `Some(meta)` iff `dt` is a supported vector type — a `FixedSizeList` of a
/// known scalar element type with a non-negative length; `None` otherwise.
/// Ignores the child field's name/nullability so it round-trips with
/// [`vector_dtype_for_dim`] regardless of how the column was constructed.
pub fn is_vector_type(dt: &DataType) -> Option<VectorMeta> {
    let DataType::FixedSizeList(field, len) = dt else {
        return None;
    };
    let scalar = match field.data_type() {
        DataType::Float16 => VectorScalar::F16,
        _ => return None,
    };
    (*len >= 0).then_some(VectorMeta { dim: *len as u32, scalar })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_for_dim_round_trips_through_is_vector_type() {
        for dim in [1u32, 4, 128, 1536] {
            let dt = vector_dtype_for_dim(dim, VectorScalar::F16);
            let meta = is_vector_type(&dt).expect("constructed vector type is recognized");
            assert_eq!(meta, VectorMeta { dim, scalar: VectorScalar::F16 });
        }
    }

    #[test]
    fn non_vector_types_are_rejected() {
        assert_eq!(is_vector_type(&DataType::Float32), None);
        assert_eq!(is_vector_type(&DataType::Int64), None);
        // FixedSizeList of a non-fp16 element is not (yet) a supported vector.
        let int_list = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 4);
        assert_eq!(is_vector_type(&int_list), None);
    }
}
