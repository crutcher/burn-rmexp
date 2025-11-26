//! # Tensor Provider

use burn::tensor::{BasicOps, Slice, TensorData};
use std::future::Future;
use burn::prelude::{Backend, Bool, Float, Int};
use burn::Tensor;

pub mod metadata;

pub enum FlatTensorWrapper<B: Backend> {
    Float(Tensor<B, 1, Float>),
    Int(Tensor<B, 1, Int>),
    Bool(Tensor<B, 1, Bool>),
}

impl<B: Backend> From<Tensor<B, 1, Float>> for FlatTensorWrapper<B> {
    fn from(value: Tensor<B, 1, Float>) -> Self {
        FlatTensorWrapper::Float(value)
    }
}
impl<B: Backend> From<Tensor<B, 1, Int>> for FlatTensorWrapper<B> {
    fn from(value: Tensor<B, 1, Int>) -> Self {
        FlatTensorWrapper::Int(value)
    }
}
impl<B: Backend> From<Tensor<B, 1, Bool>> for FlatTensorWrapper<B> {
    fn from(value: Tensor<B, 1, Bool>) -> Self {
        FlatTensorWrapper::Bool(value)
    }
}

pub enum TensorProviderError {
    InvalidShape,
    OutOfBounds,
    Unavailable,
    InternalError,
}

pub trait TensorProvider<B: Backend> {
    fn typed_shape(&self) -> metadata::TypedShape;

    fn get_slice_data(&self, slice: Slice) -> impl Future<Output=Result<TensorData, TensorProviderError>> + Send;

    fn get_flat_slice_tensor(&self, slice: Slice)
    -> impl Future<Output=Result<FlatTensorWrapper<B>, TensorProviderError>> + Send;
}

pub struct ConstantTensorProvider<B, const D: usize, K>
where B: Backend, K: BasicOps<B>{
    pub tensor: Tensor<B, D, K>,
}

impl<B, const D: usize, K> ConstantTensorProvider<B, D, K>
where B: Backend, K: BasicOps<B> {
    pub fn new(tensor: Tensor<B, D, K>) -> Self {
        Self { tensor }
    }
}

pub fn flatten<B, const D: usize, K>(tensor: Tensor<B, D, K>) -> Tensor<B, 1, K>
where B: Backend, K: BasicOps<B>
{
    let num_elem = tensor.shape().num_elements();
    tensor.reshape([num_elem])
}

impl<B, const D: usize, K> TensorProvider<B> for ConstantTensorProvider<B, D, K>
where B: Backend, K: BasicOps<B> {
    fn typed_shape(&self) -> metadata::TypedShape {
        metadata::TypedShape { dtype: self.tensor.dtype(), shape: self.tensor.shape() }
    }

    fn get_slice_data(&self, slice: Slice) -> impl Future<Output=Result<TensorData, TensorProviderError>> + Send {
        async move {
            // TODO: check slice bounds
            Ok(self.tensor.clone().slice(slice).into_data())
        }
    }

    fn get_flat_slice_tensor(&self, slice: Slice) -> impl Future<Output=Result<FlatTensorWrapper<B>, TensorProviderError>> + Send {
        async move {
            // TODO: check slice bounds
            let tensor = self.tensor.clone().slice(slice);
            let _tensor: Tensor<B, 1, K> = flatten(tensor);
            todo!()
        }
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