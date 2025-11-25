//! # Tensor Provider

use burn::tensor::{Slice, TensorData, TensorKind};
use std::future::Future;
use burn::prelude::Backend;
use burn::Tensor;

pub mod metadata;

pub enum TensorProviderError {
    InvalidShape,
    OutOfBounds,
    Unavailable,
    InternalError,
}

pub trait TensorProvider {
    fn typed_shape(&self) -> metadata::TypedShape;

    fn get_slice_data(&self, slice: Slice) -> impl Future<Output=Result<TensorData, TensorProviderError>> + Send;

    fn get_flat_slice_tensor<B, K>(&self, slice: Slice)
    -> impl Future<Output=Result<Tensor<B, 1, K>, TensorProviderError>> + Send
    where B: Backend,
          K: TensorKind<B>;
}

pub struct ConstantTensorProvider<B, const D: usize, K>
where B: Backend, K: TensorKind<B>{
    tensor: Tensor<B, D, K>,
}

impl<B, const D: usize, K> ConstantTensorProvider<B, D, K>
where B: Backend, K: TensorKind<B> {
    pub fn new(tensor: Tensor<B, D, K>) -> Self {
        Self { tensor }
    }
}

impl<B, const D: usize, K> TensorProvider for ConstantTensorProvider<B, D, K>
where B: Backend, K: TensorKind<B> {
    fn typed_shape(&self) -> metadata::TypedShape {
        metadata::TypedShape { dtype: self.tensor.dtype(), shape: self.tensor.shape() }
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn it_works() {
        tokio::time::pause();
        let start = std::time::Instant::now();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        println!("elapsed: {:?}", start.elapsed());
    }

}