//! What the device hands back for a declared column, worked out before the crossing.
//!
//! The round trip is Arrow -> `fb::DataType` (`plan_serializer::convert_data_type`) ->
//! `cudf::type_id` (`fb_to_type_id`, `cpp/src/expr.cpp`) -> Arrow
//! (`cudf::to_arrow_schema`, from `export_table_to_ipc`). It lives in three files and two
//! languages, so before this nobody could answer what a column comes back as without
//! reading all three. Most of it is identity; the arms that are not are named below.

use datafusion::arrow::datatypes::{
    DECIMAL128_MAX_PRECISION, DataType, Schema as ArrowSchema, TimeUnit,
};

use super::error::PlanError;

/// Why a column does not come back as it was declared. Which of the three the unload
/// casts is the whole reason they are told apart: one is inherent and the other two are
/// gaps with a fix of their own, and casting those would hide them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Divergence {
    /// cuDF has one string type, so `Utf8View` and `LargeUtf8` both arrive as `Utf8`
    /// ([#183](../../../../llm-wiki/tasks/bp-tickets.md#t183)). Neither side can be
    /// changed to avoid it, which is what makes the unload the place to absorb it.
    StringType,
    /// cuDF decimals carry a scale and no precision, and the export passes no precision
    /// through `column_metadata`, so every decimal arrives at the maximum
    /// ([#187](../../../../llm-wiki/tasks/bp-tickets.md#t187)). Predicted here and not
    /// cast: `wire-schema.md` puts the precision on the wire instead.
    DecimalPrecision,
    /// `Date64` is a millisecond timestamp to cuDF and comes back as one. No corpus query
    /// declares a `Date64`, so nothing reaches this.
    DateAsTimestamp,
}

/// What the device will hand back for one declared type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Export {
    Identity,
    Diverges { exported: DataType, why: Divergence },
}

/// The exported type for one declared column, over every type the wire carries.
///
/// `Err` where the wire carries the type and cuDF has no mapping for it: `convert_data_type`
/// serializes all five of those and `fb_to_type_id` answers `EMPTY` for each, so the column
/// would reach the device typeless. Refusing where the plan is built is what writing this
/// down buys — today they are stopped only by whatever fails first on the device.
pub fn export_type_for(declared: &DataType) -> Result<Export, PlanError> {
    Ok(match declared {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::Date32 => Export::Identity,
        DataType::Utf8View | DataType::LargeUtf8 => Export::Diverges {
            exported: DataType::Utf8,
            why: Divergence::StringType,
        },
        DataType::Date64 => Export::Diverges {
            exported: DataType::Timestamp(TimeUnit::Millisecond, None),
            why: Divergence::DateAsTimestamp,
        },
        // The export's fallback is `max_precision<__int128_t>()`, which is this constant.
        // A column declared at it already is what it comes back as, which is why twenty
        // corpus queries of decimal sums went past #187 and the first plain projection
        // did not.
        DataType::Decimal128(precision, scale) if *precision != DECIMAL128_MAX_PRECISION => {
            Export::Diverges {
                exported: DataType::Decimal128(DECIMAL128_MAX_PRECISION, *scale),
                why: Divergence::DecimalPrecision,
            }
        }
        DataType::Decimal128(_, _) => Export::Identity,
        // The five `fb_to_type_id` answers `EMPTY` for. The wire carries them, so the column
        // reaches the device with no type at all rather than being stopped on the way.
        DataType::Null
        | DataType::Float16
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => {
            return Err(PlanError::Unsupported(format!(
                "a {declared:?} column at the sink: the wire carries the type and cuDF maps it \
                 to no type at all, so the device would export a typeless column"
            )));
        }
        // Everything else, and a different sentence because a different thing is true of it:
        // `convert_data_type` has no arm for it either, so the plan is refused at
        // serialization and no sink is ever reached. Separated so that a type added to the
        // wire lands here rather than being absorbed by a refusal that reads as considered.
        other => {
            return Err(PlanError::Unsupported(format!(
                "a {other:?} column at the sink: the wire has no encoding for the type"
            )));
        }
    })
}

/// One column of a sink's input that does not come back as declared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnExport {
    pub ordinal: u32,
    pub exported: DataType,
    pub why: Divergence,
}

/// A sink's columns that diverge, in ordinal order. The identity ones are left out: what
/// the list is for is naming what differs, and every column of every plan would bury it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Exports {
    columns: Vec<ColumnExport>,
}

impl Exports {
    /// Over the schema the sink consumes, which is the schema its rows are compared
    /// against on the way out.
    pub fn of(schema: &ArrowSchema) -> Result<Self, PlanError> {
        let mut columns = Vec::new();
        for (ordinal, field) in schema.fields().iter().enumerate() {
            if let Export::Diverges { exported, why } = export_type_for(field.data_type())? {
                columns.push(ColumnExport {
                    ordinal: ordinal as u32,
                    exported,
                    why,
                });
            }
        }
        Ok(Self { columns })
    }

    pub fn columns(&self) -> &[ColumnExport] {
        &self.columns
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The ordinals the unload casts back to what was declared: the string arm and
    /// nothing else. The concat against the sink's schema is the only thing checking that
    /// the device produced what the plan said it would, and a cast per divergence would
    /// leave nothing checking it at all.
    pub fn cast_ordinals(&self) -> Vec<u32> {
        self.columns
            .iter()
            .filter(|column| column.why == Divergence::StringType)
            .map(|column| column.ordinal)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire's set, parsed out of `convert_data_type`'s match arms. Reading the source
    /// rather than listing the types here, for the reason `node_kind_names` reads
    /// `node_name`: a list written beside the check can miss what the match cannot.
    fn wire_types() -> Vec<&'static str> {
        const SOURCE: &str = include_str!("../plan_serializer.rs");
        let body = SOURCE
            .split_once("pub(crate) fn convert_data_type(")
            .expect("plan_serializer.rs declares convert_data_type")
            .1;
        let body = body
            .split_once("\n}")
            .expect("convert_data_type has a body")
            .0;
        let names: Vec<&str> = body
            .lines()
            .filter_map(|line| line.trim().strip_prefix("ArrowDataType::"))
            .filter_map(|rest| rest.split(['(', ' ']).next())
            .collect();
        // Against every mention, not against the arms: a merged arm — `A | B => …` on one
        // line, or with the `|` wrapped onto the next — drops a name and an arm together, so
        // an arm count balances while the second type leaves the cover. The five that map
        // alike are what anyone would merge, so this is the shape to expect here.
        assert_eq!(
            names.len(),
            body.matches("ArrowDataType::").count(),
            "convert_data_type names {} types and the line scan found {} — a merged arm, and \
             the scan has to take both sides of it",
            body.matches("ArrowDataType::").count(),
            names.len()
        );
        // A different mistake, and still worth its own line: an arm that names no type at all.
        // Two merged arms leave this balanced, which is why it does not replace the count
        // above.
        assert_eq!(
            names.len() + 1,
            body.matches("=>").count(),
            "convert_data_type has {} arms and {} named a type — one arm names none",
            body.matches("=>").count(),
            names.len()
        );
        names
    }

    /// A value per wire type, since a parameterized arm carries no arguments to build from.
    /// Exhaustive on purpose: a type added to the wire fails here naming itself rather than
    /// defaulting to something that happens to pass.
    fn value_of(name: &str) -> DataType {
        match name {
            "Null" => DataType::Null,
            "Boolean" => DataType::Boolean,
            "Int8" => DataType::Int8,
            "Int16" => DataType::Int16,
            "Int32" => DataType::Int32,
            "Int64" => DataType::Int64,
            "UInt8" => DataType::UInt8,
            "UInt16" => DataType::UInt16,
            "UInt32" => DataType::UInt32,
            "UInt64" => DataType::UInt64,
            "Float16" => DataType::Float16,
            "Float32" => DataType::Float32,
            "Float64" => DataType::Float64,
            "Utf8" => DataType::Utf8,
            "LargeUtf8" => DataType::LargeUtf8,
            "Binary" => DataType::Binary,
            "LargeBinary" => DataType::LargeBinary,
            "Date32" => DataType::Date32,
            "Date64" => DataType::Date64,
            "Decimal128" => DataType::Decimal128(15, 2),
            "Utf8View" => DataType::Utf8View,
            "BinaryView" => DataType::BinaryView,
            other => panic!("{other} reaches the wire and has no value here — add one"),
        }
    }

    /// Totality as a property rather than as three examples. The catch-all makes every
    /// unnamed type an `Err`, so a type that should be identity and was forgotten would be
    /// refused with nothing going red; this is what notices.
    #[test]
    fn every_type_the_wire_carries_is_answered_and_the_five_typeless_ones_refuse() {
        let typeless = ["Null", "Float16", "Binary", "LargeBinary", "BinaryView"];
        let mut refused = 0;
        for name in wire_types() {
            let answer = export_type_for(&value_of(name));
            assert_eq!(
                answer.is_err(),
                typeless.contains(&name),
                "{name}: cuDF maps it to {}, and export_type_for {}",
                if typeless.contains(&name) {
                    "nothing"
                } else {
                    "a type"
                },
                if answer.is_err() {
                    "refuses"
                } else {
                    "answers"
                }
            );
            refused += usize::from(answer.is_err());
        }
        assert_eq!(refused, typeless.len(), "every typeless type is reached");
    }
}
