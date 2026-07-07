//! Pure-Rust CPU distance kernels over fp16 vectors. Straightforward scalar
//! loops accumulating in f32 — no SIMD yet (a later optimization); the GPU path
//! lands in a later ticket. This is also the exact reference the GPU results are
//! checked against.

use datafusion::arrow::array::{Array, FixedSizeListArray, Float16Array, Float32Array};
use datafusion::error::{DataFusionError, Result};
use half::f16;

/// L2 (Euclidean) distance between two equal-length fp16 slices: `sqrt(Σ (a-b)²)`,
/// squared differences accumulated in f32 to limit fp16 rounding.
pub fn l2_distance_row(a: &[f16], b: &[f16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x.to_f32() - y.to_f32();
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Row-wise L2 distance between two `FixedSizeList<Float16, N>` arrays, returned
/// as a `Float32Array` of the same length. Output row `i` is null iff either input
/// list row `i` is null. Errors on dimension/row-count mismatch or non-fp16 elements.
pub fn l2_distance_arrays(a: &FixedSizeListArray, b: &FixedSizeListArray) -> Result<Float32Array> {
    if a.value_length() != b.value_length() {
        return Err(DataFusionError::Execution(format!(
            "l2_distance: vector dimension mismatch ({} vs {})",
            a.value_length(),
            b.value_length()
        )));
    }
    if a.len() != b.len() {
        return Err(DataFusionError::Execution(format!(
            "l2_distance: row-count mismatch ({} vs {})",
            a.len(),
            b.len()
        )));
    }
    let dim = a.value_length() as usize;
    let a_vals = fp16_child(a)?;
    let b_vals = fp16_child(b)?;

    let mut out = Float32Array::builder(a.len());
    for row in 0..a.len() {
        if a.is_null(row) || b.is_null(row) {
            out.append_null();
            continue;
        }
        let base = row * dim;
        out.append_value(l2_distance_row(
            &a_vals[base..base + dim],
            &b_vals[base..base + dim],
        ));
    }
    Ok(out.finish())
}

/// The flat fp16 child buffer of a `FixedSizeList<Float16, N>` array.
fn fp16_child(list: &FixedSizeListArray) -> Result<&[f16]> {
    list.values()
        .as_any()
        .downcast_ref::<Float16Array>()
        .map(|a| a.values().as_ref())
        .ok_or_else(|| {
            DataFusionError::Execution(
                "l2_distance: vector elements must be Float16".to_string(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::FixedSizeListBuilder;
    use datafusion::arrow::array::Float16Builder;

    fn f16s(xs: &[f32]) -> Vec<f16> {
        xs.iter().map(|&x| f16::from_f32(x)).collect()
    }

    #[test]
    fn row_kernel_matches_hand_computed() {
        // a=[1,2,3], b=[4,6,8]: diffs 3,4,5 -> sqrt(9+16+25)=sqrt(50)=7.0710678.
        let a = f16s(&[1.0, 2.0, 3.0]);
        let b = f16s(&[4.0, 6.0, 8.0]);
        let d = l2_distance_row(&a, &b);
        assert!((d - 50.0f32.sqrt()).abs() < 1e-2, "got {d}");
        // Identical vectors -> 0.
        assert_eq!(l2_distance_row(&a, &a), 0.0);
    }

    /// Build a 2-row FixedSizeList<Float16,3> array from row slices.
    fn fsl(rows: &[Vec<f16>]) -> FixedSizeListArray {
        let dim = rows[0].len() as i32;
        let mut b = FixedSizeListBuilder::new(Float16Builder::new(), dim);
        for row in rows {
            for &v in row {
                b.values().append_value(v);
            }
            b.append(true);
        }
        b.finish()
    }

    #[test]
    fn array_kernel_row_wise() {
        let a = fsl(&[f16s(&[1.0, 2.0, 3.0]), f16s(&[0.0, 0.0, 0.0])]);
        let b = fsl(&[f16s(&[4.0, 6.0, 8.0]), f16s(&[0.0, 3.0, 4.0])]);
        let out = l2_distance_arrays(&a, &b).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out.value(0) - 50.0f32.sqrt()).abs() < 1e-2);
        assert!((out.value(1) - 5.0).abs() < 1e-2); // sqrt(0+9+16)=5
    }

    #[test]
    fn dimension_mismatch_is_an_error() {
        let a = fsl(&[f16s(&[1.0, 2.0, 3.0])]);
        let b = fsl(&[f16s(&[1.0, 2.0])]);
        assert!(l2_distance_arrays(&a, &b).is_err());
    }
}
