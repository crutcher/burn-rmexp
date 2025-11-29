use crate::library::metadata::TypedShape;
use burn::prelude::TensorData;
use std::fmt::Debug;
use std::sync::Arc;

pub mod metadata;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProviderError {
    Unavailable,
    InternalError,
}

/// Abstract provider for [`TensorData`].
pub trait TensorDataProvider: Debug {
    /// Get the [`TypedShape`] of the tensor.
    fn typed_shape(&self) -> TypedShape;

    /// Asynchronously get the [`TensorData`] of the tensor.
    ///
    /// # Returns
    /// An `Result<Arc<TensorData>, ProviderError>`.`
    fn get_data(&mut self) -> impl Future<Output = Result<Arc<TensorData>, ProviderError>> + Send;
}

/// A [`TensorDataProvider`] that always returns the same [`TensorData`].
#[derive(Debug, Clone)]
pub struct ConstProvider {
    data: Arc<TensorData>,
}

impl ConstProvider {
    pub fn new(data: impl Into<Arc<TensorData>>) -> Self {
        Self { data: data.into() }
    }

    pub fn data(&self) -> Arc<TensorData> {
        self.data.clone()
    }
}

impl TensorDataProvider for ConstProvider {
    fn typed_shape(&self) -> TypedShape {
        TypedShape {
            dtype: self.data.dtype,
            shape: self.data.shape.clone().into(),
        }
    }

    fn get_data(&mut self) -> impl Future<Output = Result<Arc<TensorData>, ProviderError>> + Send {
        async move { Ok(self.data()) }
    }
}

pub type CatalogueId = String;

/*
pub struct TensorDataCatalogue {}

impl TensorDataCatalogue {
    pub fn get_provider(
        &self,
        id: CatalogueId,
    ) -> impl Future<Output = Result<???, ProviderError>> + Send {
    }
}
 */

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Shape;
    use burn::tensor::DType;

    #[tokio::test]
    async fn test_constant_tensor_data_provider() {
        let data = TensorData::ones::<f32, _>([2, 3]);
        let mut provider = ConstProvider::new(data);

        assert_eq!(
            provider.typed_shape(),
            TypedShape {
                dtype: DType::F32,
                shape: Shape::new([2, 3])
            }
        );

        let fetch = provider.get_data().await.unwrap();
        assert_eq!(fetch, provider.data())
    }
}
