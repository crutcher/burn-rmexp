//! # Tensor Operations
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{BasicOps, Slice};

/// Provides a dynamic version of [`Tensor::slice`].
pub fn slice_dyn<B: Backend, const R: usize, K: BasicOps<B>>(
    tensor: Tensor<B, R, K>,
    slices: &[Slice],
) -> Tensor<B, R, K> {
    let mut tensor = tensor;
    for (dim, slice) in slices.iter().enumerate() {
        tensor = tensor.slice_dim(dim, *slice);
    }
    tensor
}
