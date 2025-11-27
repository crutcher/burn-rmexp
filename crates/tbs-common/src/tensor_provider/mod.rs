//! # Tensor Provider

use burn::Tensor;
use burn::prelude::{Backend, Bool, Float, Int};
use burn::tensor::{BasicOps, Slice, TensorData, TensorKind};
use std::any::Any;
use std::future::Future;

pub mod metadata;

#[derive(Debug, Clone)]
pub enum FlatTensorWrapper<B: Backend> {
    Float(Tensor<B, 1, Float>),
    Int(Tensor<B, 1, Int>),
    Bool(Tensor<B, 1, Bool>),
}

impl<B: Backend> FlatTensorWrapper<B> {
    /// Wraps a tensor into a `FlatTensorWrapper` if it is a supported tensor kind.
    pub fn wrap<K: TensorKind<B> + 'static>(tensor: Tensor<B, 1, K>) -> Result<Self, String> {
        let any: &dyn Any = &tensor;

        if let Some(t) = any.downcast_ref::<Tensor<B, 1, Float>>() {
            Ok(Self::Float(t.clone()))
        } else if let Some(t) = any.downcast_ref::<Tensor<B, 1, Int>>() {
            Ok(Self::Int(t.clone()))
        } else if let Some(t) = any.downcast_ref::<Tensor<B, 1, Bool>>() {
            Ok(Self::Bool(t.clone()))
        } else {
            Err(format!("Unsupported tensor kind: {:?}", K::name()))
        }
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

    fn get_slice_data(
        &self,
        slice: Slice,
    ) -> impl Future<Output = Result<TensorData, TensorProviderError>> + Send;

    fn get_flat_slice_tensor(
        &self,
        slice: Slice,
    ) -> impl Future<Output = Result<FlatTensorWrapper<B>, TensorProviderError>> + Send;
}

pub struct ConstantTensorProvider<B, const D: usize, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    pub tensor: Tensor<B, D, K>,
}

impl<B, const D: usize, K> ConstantTensorProvider<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    pub fn new(tensor: Tensor<B, D, K>) -> Self {
        Self { tensor }
    }
}

pub fn flatten<B, const D: usize, K>(tensor: Tensor<B, D, K>) -> Tensor<B, 1, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    let num_elem = tensor.shape().num_elements();
    tensor.reshape([num_elem])
}

impl<B, const D: usize, K> TensorProvider<B> for ConstantTensorProvider<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn typed_shape(&self) -> metadata::TypedShape {
        metadata::TypedShape {
            dtype: self.tensor.dtype(),
            shape: self.tensor.shape(),
        }
    }

    fn get_slice_data(
        &self,
        slice: Slice,
    ) -> impl Future<Output = Result<TensorData, TensorProviderError>> + Send {
        async move {
            // TODO: check slice bounds
            Ok(self.tensor.clone().slice(slice).into_data())
        }
    }

    fn get_flat_slice_tensor(
        &self,
        slice: Slice,
    ) -> impl Future<Output = Result<FlatTensorWrapper<B>, TensorProviderError>> + Send {
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
    use super::*;
    use burn::backend::Wgpu;

    #[test]
    fn test_wrap() {
        type B = Wgpu;
        let device = Default::default();

        let tensor: Tensor<B, 1> = Tensor::ones([12], &device);
        let wrapper = FlatTensorWrapper::wrap(tensor.clone()).unwrap();
        match wrapper {
            FlatTensorWrapper::Float(actual) => actual.to_data().assert_eq(&tensor.to_data(), true),
            _ => panic!("Unexpected tensor kind: {:?}", wrapper),
        }

        let tensor: Tensor<B, 1, Int> = Tensor::ones([12], &device);
        let wrapper = FlatTensorWrapper::wrap(tensor.clone()).unwrap();
        match wrapper {
            FlatTensorWrapper::Int(actual) => actual.to_data().assert_eq(&tensor.to_data(), true),
            _ => panic!("Unexpected tensor kind: {:?}", wrapper),
        }

        let tensor: Tensor<B, 1, Bool> = Tensor::ones([12], &device);
        let wrapper = FlatTensorWrapper::wrap(tensor.clone()).unwrap();
        match wrapper {
            FlatTensorWrapper::Bool(actual) => actual.to_data().assert_eq(&tensor.to_data(), true),
            _ => panic!("Unexpected tensor kind: {:?}", wrapper),
        }
    }

    #[tokio::test]
    async fn it_works() {
        tokio::time::pause();
        let start = std::time::Instant::now();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        println!("elapsed: {:?}", start.elapsed());
    }
}
