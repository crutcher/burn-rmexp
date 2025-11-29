use crate::dyn_tensor::DynTensor;
use burn::prelude::Backend;
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

    /// Query by path.
    Path(Vec<String>),
}

impl From<uuid::Uuid> for TensorLibraryQuery {
    fn from(uuid: uuid::Uuid) -> Self {
        Self::Uuid(uuid)
    }
}

impl TensorLibraryQuery {
    fn path_from_sep<S: AsRef<str>>(
        path: S,
        sep: &str,
    ) -> TensorLibraryQuery {
        let path = path
            .as_ref()
            .split(sep)
            .map(String::from)
            .collect::<Vec<_>>();
        TensorLibraryQuery::Path(path)
    }

    /// Parse a dotted path.
    pub fn parse_dotted_path<S: AsRef<str>>(path: S) -> TensorLibraryQuery {
        Self::path_from_sep(path, ".")
    }

    /// Parse a slashed path.
    pub fn parse_slashed_path<S: AsRef<str>>(path: S) -> TensorLibraryQuery {
        Self::path_from_sep(path, "/")
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
        &'a self,
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
        &'a self,
        query: TensorLibraryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>> + Send + 'a>>
    {
        // Query each library in parallel.
        let fs = self
            .libs
            .iter()
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

#[derive(Debug, Clone)]
pub struct UuidMapTensorLibrary<B: Backend> {
    hash_map: HashMap<uuid::Uuid, DynTensor<B>>,
}

impl<B: Backend> Default for UuidMapTensorLibrary<B> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<B: Backend> UuidMapTensorLibrary<B> {
    /// Create an empty library.
    pub fn empty() -> Self {
        Self {
            hash_map: Default::default(),
        }
    }

    /// Create a new library from a map.
    pub fn new(map: HashMap<uuid::Uuid, DynTensor<B>>) -> Self {
        Self { hash_map: map }
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

    /// Get a tensor from the library.
    /// Returns `None` if the tensor was not found.
    pub fn get_clone(
        &self,
        key: &uuid::Uuid,
    ) -> Option<DynTensor<B>> {
        self.hash_map.get(key).cloned()
    }
}

impl<B: Backend> TensorLibrary<B> for UuidMapTensorLibrary<B> {
    /// Query a tensor from the library.
    fn query<'a>(
        &'a self,
        query: TensorLibraryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Option<DynTensor<B>>, TensorLibraryError>> + Send + 'a>>
    {
        Box::pin(async move {
            match query {
                TensorLibraryQuery::Uuid(uuid) => Ok(self.get_clone(&uuid)),
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
    #[tokio::test]
    async fn test_map_library() {
        type B = Wgpu;
        let device = Default::default();

        let mut library = UuidMapTensorLibrary::empty();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Default::default(), &device);

        assert!(
            library
                .query(uuid::Uuid::new_v4().into())
                .await
                .expect("query failed")
                .is_none()
        );

        let id = library.bind(source.clone());

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
}
