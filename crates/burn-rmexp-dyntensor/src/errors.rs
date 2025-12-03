//! # Common Error Types
use burn::prelude::Shape;
use burn::tensor::Slice;

/// Errors that can occur when checking tensor slices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SlicingError {
    OutOfBounds {
        msg: String,
        shape: Shape,
        slices: Vec<Slice>,
    },
    InvalidRank {
        msg: String,
        shape: Shape,
        slices: Vec<Slice>,
    },
}

/// Common Errors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DynTensorError {
    /// An error occurred while slicing.
    SliceError(SlicingError),

    /// Invalid Arguments.
    InvalidArgument { msg: String },

    /// The tensor rank is not supported for the requested operation.
    UnsupportedRank { msg: String, rank: usize },
}
