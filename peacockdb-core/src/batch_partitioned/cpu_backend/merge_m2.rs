//! `merge_m2`: state into state, which none of DataFusion's aggregate modes does.
//!
//! Partial takes values and emits state, Final takes state and emits a value, Single does
//! both at once. This mode stacks two merges — per lane, then across lanes — so the lower
//! one must emit the state the upper one reads. It is the gap `AggregateMode::Merge` was
//! added to the wire to fill.
//!
//! The accumulator is DataFusion's own with one method rewired: what arrives is state, so
//! `update_batch` merges it. No part of Welford is written out here, because the device
//! runs cuDF's MERGE_M2 and two spellings differ in the last digits.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::{Result as DfResult, ScalarValue};
use datafusion::functions_aggregate::stddev::stddev_udaf;
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
};

/// What this aggregate is called. Ours rather than DataFusion's: nothing in a session
/// registers it, and the CPU backend hands the definition to the aggregate it builds.
pub const NAME: &str = "merge_m2";

pub fn udaf() -> Arc<AggregateUDF> {
    Arc::new(AggregateUDF::from(MergeM2::new()))
}

#[derive(Debug)]
struct MergeM2 {
    signature: Signature,
}

impl MergeM2 {
    fn new() -> Self {
        Self {
            // The Welford state as DataFusion declares it, in its order: a count and two
            // doubles. Exact rather than coercible — a state column arriving as another
            // type is a decomposition that has gone wrong upstream, and coercing it would
            // merge the wrong numbers rather than say so.
            signature: Signature::exact(
                vec![DataType::UInt64, DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for MergeM2 {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    /// What a finalize would produce. Nothing reaches it: this mode finalizes in a
    /// project, so the aggregate is only ever asked for its state.
    fn return_type(&self, _args: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::Float64)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> DfResult<Vec<Field>> {
        Ok(vec![
            Field::new(format!("{}[count]", args.name), DataType::UInt64, true),
            Field::new(format!("{}[mean]", args.name), DataType::Float64, true),
            Field::new(format!("{}[m2]", args.name), DataType::Float64, true),
        ])
    }

    fn accumulator(&self, args: AccumulatorArgs) -> DfResult<Box<dyn Accumulator>> {
        Ok(Box::new(Merging {
            inner: stddev_udaf().accumulator(args)?,
        }))
    }
}

/// DataFusion's own stddev accumulator, reading its input as state rather than as values.
#[derive(Debug)]
struct Merging {
    inner: Box<dyn Accumulator>,
}

impl Accumulator for Merging {
    /// The rewired method, and the whole of what this type is for.
    fn update_batch(&mut self, values: &[ArrayRef]) -> DfResult<()> {
        self.inner.merge_batch(values)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> DfResult<()> {
        self.inner.merge_batch(states)
    }

    fn state(&mut self) -> DfResult<Vec<ScalarValue>> {
        self.inner.state()
    }

    fn evaluate(&mut self) -> DfResult<ScalarValue> {
        self.inner.evaluate()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}
