use burn::prelude::Backend;
use burn_ext_dyntensor::DynTensor;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::pin::Pin;

/// Query for a tensor in a [`TensorLibrary`]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorLibraryQuery {
    /// Query by UUID.
    Uuid(uuid::Uuid),

    /// Query by route.
    Route(Vec<String>),

    /// Query by string path.
    Path(String),
}

impl From<uuid::Uuid> for TensorLibraryQuery {
    fn from(uuid: uuid::Uuid) -> Self {
        Self::Uuid(uuid)
    }
}

/// Error returned by [`TensorLibrary`]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorLibraryError {
    InvalidQuery(TensorLibraryQuery),
}

pub trait TensorLibrary<B: Backend>: 'static + Debug {
    /// Query a tensor from the library.
    fn query<'a>(
        &'a mut self,
        query: TensorLibraryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>> + Send + 'a>>;
}

#[derive(Debug)]
pub struct TensorLibraryCollection<B: Backend> {
    libs: Vec<Box<dyn TensorLibrary<B>>>,
}

impl<B: Backend> Default for TensorLibraryCollection<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TensorLibraryCollection<B> {
    /// Create a new empty library collection.
    pub fn new() -> Self {
        Self { libs: Vec::new() }
    }

    /// Add a library to the collection.
    pub fn push(
        &mut self,
        lib: Box<dyn TensorLibrary<B>>,
    ) {
        self.libs.push(lib);
    }

    /// Get a reference to the underlying libraries.
    pub fn libs(&self) -> &[Box<dyn TensorLibrary<B>>] {
        &self.libs
    }

    /// Get a mutable reference to the underlying libraries.
    pub fn libs_mut(&mut self) -> &mut [Box<dyn TensorLibrary<B>>] {
        &mut self.libs
    }
}

impl<B: Backend> TensorLibrary<B> for TensorLibraryCollection<B> {
    fn query<'a>(
        &'a mut self,
        query: TensorLibraryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>> + Send + 'a>>
    {
        // Query each library in parallel.
        let fs = self
            .libs
            .iter_mut()
            .map(|lib| lib.query(query.clone()))
            .collect::<Vec<_>>();

        // Forward the first error; or the first non-None result.
        Box::pin(async move {
            let res = join_all(fs).await;
            res.into_iter().try_fold(None, |acc, result| match result {
                Err(e) => Err(e),
                Ok(Some(val)) if acc.is_none() => Ok(Some(val)),
                Ok(_) => Ok(acc),
            })
        })
    }
}

/// A [`TensorLibrary`] backed by a [`uuid::Uuid`] keyed [`HashMap`].
#[derive(Debug, Clone)]
pub struct UuidMapTensorLibrary<B: Backend> {
    hash_map: HashMap<uuid::Uuid, DynTensor<B>>,
}

impl<B: Backend> From<HashMap<uuid::Uuid, DynTensor<B>>> for UuidMapTensorLibrary<B> {
    fn from(hash_map: HashMap<uuid::Uuid, DynTensor<B>>) -> Self {
        Self { hash_map }
    }
}

impl<B: Backend> Default for UuidMapTensorLibrary<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> UuidMapTensorLibrary<B> {
    /// Create an empty library.
    pub fn new() -> Self {
        Self {
            hash_map: HashMap::new(),
        }
    }

    /// Get a reference to the internal map.
    pub fn hash_map(&self) -> &HashMap<uuid::Uuid, DynTensor<B>> {
        &self.hash_map
    }

    /// Get a mutable reference to the internal map.
    pub fn hash_map_mut(&mut self) -> &mut HashMap<uuid::Uuid, DynTensor<B>> {
        &mut self.hash_map
    }

    /// Insert a tensor into the library.
    /// If a tensor with the same UUID already exists, it will be replaced.
    ///
    /// # Returns
    ///
    /// The previous value, if any.
    pub fn insert<T: Into<DynTensor<B>>>(
        &mut self,
        key: uuid::Uuid,
        value: T,
    ) -> Option<DynTensor<B>> {
        self.hash_map.insert(key, value.into())
    }

    /// Bind a tensor into the library.
    /// Returns the generated UUID.
    pub fn bind<T: Into<DynTensor<B>>>(
        &mut self,
        value: T,
    ) -> uuid::Uuid {
        let key = uuid::Uuid::new_v4();
        self.insert(key, value);
        key
    }

    /// Remove a tensor from the library.
    /// Returns `None` if the tensor was not found.
    pub fn remove(
        &mut self,
        key: &uuid::Uuid,
    ) -> Option<DynTensor<B>> {
        self.hash_map.remove(key)
    }

    /// Clear the library.
    pub fn clear(&mut self) {
        self.hash_map.clear();
    }

    /// Returns the number of tensors in the library.
    pub fn len(&self) -> usize {
        self.hash_map.len()
    }

    /// Returns the size estimate of the library in bytes.
    pub fn size_estimate(&self) -> usize {
        self.hash_map
            .values()
            .map(|tensor| tensor.size_estimate())
            .sum()
    }

    /// Get a tensor ref from the library.
    pub fn get(
        &self,
        key: &uuid::Uuid,
    ) -> Option<&DynTensor<B>> {
        self.hash_map.get(key)
    }
}

impl<B: Backend> TensorLibrary<B> for UuidMapTensorLibrary<B> {
    /// Query a tensor from the library.
    fn query<'a>(
        &'a mut self,
        query: TensorLibraryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>> + Send + 'a>>
    {
        Box::pin(async move {
            match query {
                TensorLibraryQuery::Uuid(uuid) => Ok(self.get(&uuid).cloned()),
                _ => Ok(None),
            }
        })
    }
}

pub trait LazyBuilder<B: Backend>: Debug + Sync + Send + 'static {
    fn build<'a>(
        &'a self,
        query: TensorLibraryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>> + Send + 'a>>;
}

#[derive(Debug, Default)]
pub struct LazyBuilderLibrary<B: Backend> {
    builders: HashMap<uuid::Uuid, Box<dyn LazyBuilder<B>>>,
    cached: UuidMapTensorLibrary<B>,
}

impl<B: Backend> LazyBuilderLibrary<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cached(&self) -> &UuidMapTensorLibrary<B> {
        &self.cached
    }

    pub fn cached_mut(&mut self) -> &mut UuidMapTensorLibrary<B> {
        &mut self.cached
    }

    pub fn register_builder<T: LazyBuilder<B> + 'static>(
        &mut self,
        uuid: uuid::Uuid,
        builder: T,
    ) {
        self.builders.insert(uuid, Box::new(builder));
    }
}

impl<B: Backend> TensorLibrary<B> for LazyBuilderLibrary<B> {
    fn query<'a>(
        &'a mut self,
        query: TensorLibraryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>> + Send + 'a>>
    {
        Box::pin(async move {
            match query {
                TensorLibraryQuery::Uuid(uuid) => {
                    if let Some(tensor) = self.cached.get(&uuid).cloned() {
                        return Ok(Some(tensor));
                    }

                    let builder = self.builders.get(&uuid);
                    if builder.is_none() {
                        return Ok(None);
                    }

                    let qr = builder.unwrap().build(query.clone()).await?;
                    if qr.is_some() {
                        self.cached.insert(uuid, qr.as_ref().unwrap().clone());
                    }
                    Ok(qr)
                }
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use burn::prelude::Shape;
    use burn_ext_dyntensor::{DynTensor, KindFlag};

    #[tokio::test]
    async fn test_map_library() {
        type B = Wgpu;
        let device = Default::default();

        let mut library = UuidMapTensorLibrary::new();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Default::default(), &device);

        assert!(
            library
                .query(uuid::Uuid::new_v4().into())
                .await
                .expect("query failed")
                .is_none()
        );

        let id = library.bind(source.clone());

        assert_eq!(library.len(), 1);
        assert_eq!(
            library.size_estimate(),
            1 * source.shape().num_elements() * source.dtype().size()
        );

        let _dup = library.bind(source.clone());

        assert_eq!(library.len(), 2);
        assert_eq!(
            library.size_estimate(),
            2 * source.shape().num_elements() * source.dtype().size()
        );

        let dyn_tensor = library
            .query(id.into())
            .await
            .expect("query failed")
            .expect("tensor not found");

        dyn_tensor
            .to_data()
            .unwrap()
            .assert_eq(&source.to_data(), true);
    }

    #[tokio::test]
    async fn test_lazy_builder_library() {
        type B = Wgpu;
        let device: WgpuDevice = Default::default();

        #[derive(Debug)]
        struct RandomBuilder<B: Backend, const R: usize> {
            pub shape: [usize; R],
            pub device: B::Device,
        }

        impl<B: Backend, const R: usize> LazyBuilder<B> for RandomBuilder<B, R> {
            fn build<'a>(
                &'a self,
                _query: TensorLibraryQuery,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>>
                        + Send
                        + 'a,
                >,
            > {
                Box::pin(async move {
                    Ok(Some(
                        Tensor::<B, R>::random(
                            self.shape.clone(),
                            Default::default(),
                            &self.device,
                        )
                        .into(),
                    ))
                })
            }
        }

        let mut library: LazyBuilderLibrary<B> = LazyBuilderLibrary::new();
        let id = uuid::Uuid::new_v4();

        library.register_builder(
            id,
            RandomBuilder {
                shape: [2, 3],
                device: device.clone(),
            },
        );

        let dyn_tensor = library
            .query(id.into())
            .await
            .expect("query failed")
            .expect("tensor not found");

        assert_eq!(dyn_tensor.rank(), 2);
        assert_eq!(dyn_tensor.shape(), Shape::new([2, 3]));

        assert_eq!(dyn_tensor.kind(), KindFlag::Float);
        assert_eq!(dyn_tensor.device(), device);
    }
}
