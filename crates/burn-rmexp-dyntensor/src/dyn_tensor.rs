use crate::clone_box::CloneBox;
use crate::errors::DynTensorError;
use crate::kind::KindFlag;
use crate::operations;
use crate::rank_dispatch::RankHandler;
use crate::{indexing, rank_dispatch};
use burn::Tensor;
use burn::prelude::{Backend, Bool, Float, Int, Shape, SliceArg, TensorData};
use burn::tensor::{BasicOps, DType, Slice};

/// Values conversion trait for [`DynTensor::slice_assign`].
pub trait ValuesArg<B: Backend>: Sized {
    /// Convert to a [`DynTensor`] on a given device.
    fn into_values(
        self,
        device: &B::Device,
    ) -> Result<DynTensor<B>, DynTensorError>;
}

impl<B: Backend, T: Into<DynTensor<B>>> ValuesArg<B> for T {
    fn into_values(
        self,
        device: &B::Device,
    ) -> Result<DynTensor<B>, DynTensorError> {
        self.into().to_device(device)
    }
}

impl<B: Backend> ValuesArg<B> for TensorData {
    fn into_values(
        self,
        device: &B::Device,
    ) -> Result<DynTensor<B>, DynTensorError> {
        DynTensor::from_data(self, device)
    }
}

/// A dynamic [`Tensor`] wrapper that can be sliced.
#[derive(Debug, Clone)]
pub struct DynTensor<B: Backend> {
    shape: Shape,
    dtype: DType,
    kind: KindFlag,
    device: B::Device,
    tensor: Box<dyn CloneBox>,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, const R: usize, K> From<Tensor<B, R, K>> for DynTensor<B>
where
    K: 'static + BasicOps<B>,
{
    fn from(val: Tensor<B, R, K>) -> Self {
        DynTensor::new(val)
    }
}

impl<B: Backend> DynTensor<B> {
    /// Create a new `TensorStub` from a tensor.
    pub fn new<const R: usize, K>(tensor: Tensor<B, R, K>) -> Self
    where
        K: BasicOps<B> + 'static,
    {
        Self {
            shape: tensor.shape(),
            dtype: tensor.dtype(),
            kind: tensor.dtype().into(),
            device: tensor.device(),
            tensor: Box::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }

    /// Get the tensor rank.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Get the number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Returns the size estimate of the tensor in bytes.
    ///
    /// This is `self.dtype().size() * self.num_elements()`.
    pub fn size_estimate(&self) -> usize {
        self.dtype.size() * self.num_elements()
    }

    /// Get the tensor data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the tensor kind.
    pub fn kind(&self) -> KindFlag {
        self.kind
    }

    /// Get the tensor device.
    pub fn device(&self) -> B::Device {
        self.device.clone()
    }

    /// Downcasts the tensor to a specific rank and kind.
    ///
    /// # Result
    /// - `Some(Tensor<B, R, K>)`: if the params are correct,
    /// - `None`: otherwise.
    pub fn downcast_clone<const R: usize, K>(&self) -> Option<Tensor<B, R, K>>
    where
        K: 'static + BasicOps<B>,
    {
        self.tensor.downcast_ref::<Tensor<B, R, K>>().cloned()
    }

    /// Downcasts to a static tensor.
    ///
    /// # Result
    /// - the static tensor: if the params are correct,
    ///
    /// # Panics
    /// If the types are incorrect.
    pub fn unwrap_clone<const R: usize, K>(&self) -> Tensor<B, R, K>
    where
        K: 'static + BasicOps<B>,
    {
        self.downcast_clone::<R, K>()
            .expect("downcast_clone failed")
    }

    /// Slice the stub tensor.
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Arguments
    /// - `slices`: a `SliceArg<R>`.
    ///
    /// # Result
    /// - `Ok(DynTensor)`: the sliced tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice<const R: usize, S>(
        self,
        slices: S,
    ) -> Result<Self, DynTensorError>
    where
        S: SliceArg<R>,
    {
        let rank = self.rank();
        let slices = self.shape().into_slices(slices);

        indexing::check_slices_bounds(&self.shape(), &slices)
            .map_err(DynTensorError::SliceError)?;

        struct SliceHandler<B: Backend, const R: usize> {
            this: DynTensor<B>,
            slices: [Slice; R],
        }
        impl<B: Backend, const R: usize> RankHandler for SliceHandler<B, R> {
            type Output = DynTensor<B>;
            fn call<const R2: usize>(self) -> Result<Self::Output, DynTensorError> {
                Ok(match self.this.kind {
                    KindFlag::Float => self
                        .this
                        .unwrap_clone::<R, Float>()
                        .slice(self.slices)
                        .into(),
                    KindFlag::Int => self.this.unwrap_clone::<R, Int>().slice(self.slices).into(),
                    KindFlag::Bool => self
                        .this
                        .unwrap_clone::<R, Bool>()
                        .slice(self.slices)
                        .into(),
                })
            }
        }
        rank_dispatch::dispatch_rank(rank, SliceHandler { this: self, slices })
    }

    /// A dynamic version of [`DynTensor::slice`].
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Arguments
    /// - `slices`: a dynamic slice of `Slice`.
    ///
    /// # Result
    /// - `Ok(DynTensor)`: the sliced tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice_dyn(
        self,
        slices: &[Slice],
    ) -> Result<Self, DynTensorError> {
        let rank = self.rank();

        indexing::check_slices_bounds(&self.shape(), slices).map_err(DynTensorError::SliceError)?;

        struct SliceDynHandler<'a, B: Backend> {
            this: DynTensor<B>,
            slices: &'a [Slice],
        }
        impl<'a, B: Backend> RankHandler for SliceDynHandler<'a, B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                Ok(match self.this.kind {
                    KindFlag::Float => {
                        operations::slice_dyn(self.this.unwrap_clone::<R, Float>(), self.slices)
                            .into()
                    }
                    KindFlag::Int => {
                        operations::slice_dyn(self.this.unwrap_clone::<R, Int>(), self.slices)
                            .into()
                    }
                    KindFlag::Bool => {
                        operations::slice_dyn(self.this.unwrap_clone::<R, Bool>(), self.slices)
                            .into()
                    }
                })
            }
        }
        rank_dispatch::dispatch_rank(rank, SliceDynHandler { this: self, slices })
    }

    /// Assign values to a slice.
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Arguments
    /// - `slices`: a `SlicesArg<R2>`.
    /// - `values`: a coercible value; see [`ValuesArg`].
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice_assign<const R2: usize, S, V>(
        self,
        slices: S,
        values: V,
    ) -> Result<Self, DynTensorError>
    where
        S: SliceArg<R2>,
        V: ValuesArg<B>,
    {
        let rank = self.rank();
        let slices = self.shape().into_slices(slices);
        let values: DynTensor<B> = values.into_values(&self.device())?;

        indexing::check_slices_bounds(&self.shape(), &slices)
            .map_err(DynTensorError::SliceError)?;

        if rank != values.rank() {
            return Err(DynTensorError::InvalidArgument {
                msg: format!(
                    "slice of rank ({}) cannot be assigned to tensor of rank ({})",
                    values.rank(),
                    rank
                ),
            });
        }

        let values = values.cast(self.dtype())?;

        // TODO: check that slices shape == source.shape

        struct SliceAssignHandler<B: Backend, const R2: usize> {
            this: DynTensor<B>,
            slices: [Slice; R2],
            values: DynTensor<B>,
        }
        impl<B: Backend, const R2: usize> RankHandler for SliceAssignHandler<B, R2> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                Ok(match self.this.kind {
                    KindFlag::Float => self
                        .this
                        .unwrap_clone::<R, Float>()
                        .slice_assign(self.slices, self.values.unwrap_clone())
                        .into(),
                    KindFlag::Int => self
                        .this
                        .unwrap_clone::<R, Int>()
                        .slice_assign(self.slices, self.values.unwrap_clone())
                        .into(),
                    KindFlag::Bool => self
                        .this
                        .unwrap_clone::<R, Bool>()
                        .slice_assign(self.slices, self.values.unwrap_clone())
                        .into(),
                })
            }
        }
        rank_dispatch::dispatch_rank(
            rank,
            SliceAssignHandler {
                this: self.clone(),
                slices,
                values,
            },
        )
    }

    /// Dynamic slice rank version of [`DynTensor::slice_assign`].
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Arguments
    /// - `slices`: a dynamic slice of `Slice`.
    /// - `values`: a coercible value; see [`ValuesArg`].
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice_assign_dyn<V>(
        self,
        slices: &[Slice],
        values: V,
    ) -> Result<Self, DynTensorError>
    where
        V: ValuesArg<B>,
    {
        struct SliceAssignDynHandler<'a, B: Backend> {
            this: DynTensor<B>,
            slices: &'a [Slice],
            values: DynTensor<B>,
        }
        impl<'a, B: Backend> RankHandler for SliceAssignDynHandler<'a, B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                let slices: [Slice; R] = self.slices.try_into().unwrap();
                self.this.slice_assign(slices, self.values)
            }
        }
        let values = values.into_values(&self.device())?;
        rank_dispatch::dispatch_rank(
            self.rank(),
            SliceAssignDynHandler {
                this: self,
                slices,
                values,
            },
        )
    }

    /// Flatten the tensor.
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a flattened (rank=1) tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn flatten(self) -> Result<Self, DynTensorError> {
        struct FlattenHandler<B: Backend> {
            tensor: DynTensor<B>,
        }
        impl<B: Backend> RankHandler for FlattenHandler<B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                Ok(match self.tensor.kind {
                    KindFlag::Float => self
                        .tensor
                        .unwrap_clone::<R, Float>()
                        .flatten::<1>(0, self.tensor.rank() - 1)
                        .into(),
                    KindFlag::Int => self
                        .tensor
                        .unwrap_clone::<R, Int>()
                        .flatten::<1>(0, self.tensor.rank() - 1)
                        .into(),
                    KindFlag::Bool => self
                        .tensor
                        .unwrap_clone::<R, Bool>()
                        .flatten::<1>(0, self.tensor.rank() - 1)
                        .into(),
                })
            }
        }
        rank_dispatch::dispatch_rank(self.rank(), FlattenHandler { tensor: self })
    }

    /// Cast the tensor.
    ///
    /// Auto-converts kind if necessary.
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Arguments
    /// - `dtype`: the target data type.
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Result<Self, DynTensorError> {
        struct CastHandler<B: Backend> {
            this: DynTensor<B>,
            dtype: DType,
        }
        impl<B: Backend> RankHandler for CastHandler<B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                let target_kind: KindFlag = self.dtype.into();
                Ok(match self.this.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.this.unwrap_clone();
                        match target_kind {
                            KindFlag::Float => tensor.cast(self.dtype).into(),
                            KindFlag::Int => tensor.int().cast(self.dtype).into(),
                            KindFlag::Bool => tensor.bool().into(),
                        }
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.this.unwrap_clone();
                        match target_kind {
                            KindFlag::Float => tensor.float().cast(self.dtype).into(),
                            KindFlag::Int => tensor.cast(self.dtype).into(),
                            KindFlag::Bool => tensor.bool().into(),
                        }
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.this.unwrap_clone();
                        match target_kind {
                            KindFlag::Float => tensor.float().cast(self.dtype).into(),
                            KindFlag::Int => tensor.int().cast(self.dtype).into(),
                            KindFlag::Bool => self.this,
                        }
                    }
                })
            }
        }
        rank_dispatch::dispatch_rank(self.rank(), CastHandler { this: self, dtype })
    }

    /// Move the tensor to the given device.
    ///
    /// Moving to the same device is an inexpensive no-op.
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Arguments
    /// - `device`: the target device.
    ///
    /// # Result
    /// - `Ok(DynTensor<B>)`: the moved tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn to_device(
        self,
        device: &B::Device,
    ) -> Result<Self, DynTensorError> {
        if &self.device() == device {
            return Ok(self);
        }

        struct ToDeviceHandler<'a, B: Backend> {
            this: DynTensor<B>,
            device: &'a B::Device,
        }
        impl<'a, B: Backend> RankHandler for ToDeviceHandler<'a, B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                Ok(match self.this.kind {
                    KindFlag::Float => self
                        .this
                        .unwrap_clone::<R, Float>()
                        .to_device(self.device)
                        .into(),
                    KindFlag::Int => self
                        .this
                        .unwrap_clone::<R, Int>()
                        .to_device(self.device)
                        .into(),
                    KindFlag::Bool => self
                        .this
                        .unwrap_clone::<R, Bool>()
                        .to_device(self.device)
                        .into(),
                })
            }
        }
        rank_dispatch::dispatch_rank(self.rank(), ToDeviceHandler { this: self, device })
    }

    /// Convert a [`TensorData`] to a [`DynTensor`].
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Arguments
    /// - `data`: source [`TensorData`].
    /// - `device`: the target device.
    ///
    /// # Result
    /// - `Ok(DynTensor<B>)`: the converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn from_data(
        data: TensorData,
        device: &B::Device,
    ) -> Result<Self, DynTensorError> {
        struct FromDataHandler<'a, B: Backend> {
            data: TensorData,
            device: &'a B::Device,
        }
        impl<'a, B: Backend> RankHandler for FromDataHandler<'a, B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                let kind: KindFlag = self.data.dtype.into();
                Ok(match kind {
                    KindFlag::Float => {
                        Tensor::<B, R, Float>::from_data(self.data, self.device).into()
                    }
                    KindFlag::Int => Tensor::<B, R, Int>::from_data(self.data, self.device).into(),
                    KindFlag::Bool => {
                        Tensor::<B, R, Bool>::from_data(self.data, self.device).into()
                    }
                })
            }
        }
        rank_dispatch::dispatch_rank(data.rank(), FromDataHandler { data, device })
    }

    /// Convert the tensor to a [`TensorData`].
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Result
    /// - `Ok(TensorData)`: the converted data.
    /// - `Err(DynTensorError)`: an error.
    pub fn into_data(self) -> Result<TensorData, DynTensorError> {
        struct ToDataHandler<B: Backend> {
            this: DynTensor<B>,
        }
        impl<B: Backend> RankHandler for ToDataHandler<B> {
            type Output = TensorData;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                Ok(match self.this.kind {
                    KindFlag::Float => self.this.unwrap_clone::<R, Float>().into_data(),
                    KindFlag::Int => self.this.unwrap_clone::<R, Int>().into_data(),
                    KindFlag::Bool => self.this.unwrap_clone::<R, Bool>().into_data(),
                })
            }
        }
        rank_dispatch::dispatch_rank(self.rank(), ToDataHandler { this: self })
    }

    /// Convert the tensor to a [`TensorData`].
    ///
    /// Dispatches via [`rank_dispatch::dispatch_rank`].
    ///
    /// # Result
    /// - `Ok(TensorData)`: the converted data.
    /// - `Err(DynTensorError)`: an error.
    pub fn to_data(self) -> Result<TensorData, DynTensorError> {
        self.clone().into_data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::prelude::s;
    use burn::tensor::Distribution;

    fn assert_send<T: Send>() {}

    #[test]
    fn test_send() {
        type B = Wgpu;
        assert_send::<DynTensor<B>>();
    }

    #[test]
    fn test_stub_float() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        assert_eq!(stub.rank(), 2);
        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.num_elements(), 6);

        assert_eq!(stub.dtype(), source.dtype());
        assert_eq!(
            stub.size_estimate(),
            stub.num_elements() * source.dtype().size()
        );

        assert_eq!(stub.kind(), KindFlag::Float);

        assert_eq!(stub.device(), device);

        assert!(stub.downcast_clone::<2, Int>().is_none());
        assert!(stub.downcast_clone::<2, Bool>().is_none());

        assert!(stub.downcast_clone::<3, Float>().is_none());

        let clone = stub.downcast_clone::<2, Float>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);

        stub.clone()
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().flatten::<1>(0, 1).to_data(), true);
    }

    #[test]
    fn test_stub_int() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);
        let source = source.int();

        let stub = DynTensor::new(source.clone());

        assert_eq!(stub.rank(), 2);
        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.num_elements(), 6);

        assert_eq!(stub.dtype(), source.dtype());
        assert_eq!(
            stub.size_estimate(),
            stub.num_elements() * source.dtype().size()
        );

        assert_eq!(stub.kind(), KindFlag::Int);

        assert_eq!(stub.device(), device);

        assert!(stub.downcast_clone::<2, Float>().is_none());
        assert!(stub.downcast_clone::<2, Bool>().is_none());

        assert!(stub.downcast_clone::<3, Int>().is_none());

        let clone = stub.downcast_clone::<2, Int>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);

        stub.clone()
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().flatten::<1>(0, 1).to_data(), true);
    }

    #[test]
    fn test_stub_bool() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Bernoulli(0.5), &device);
        let source = source.bool();

        let stub = DynTensor::new(source.clone());

        assert_eq!(stub.rank(), 2);
        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.num_elements(), 6);

        assert_eq!(stub.dtype(), source.dtype());
        assert_eq!(
            stub.size_estimate(),
            stub.num_elements() * source.dtype().size()
        );

        assert_eq!(stub.kind(), KindFlag::Bool);

        assert_eq!(stub.device(), device);

        assert!(stub.downcast_clone::<2, Int>().is_none());
        assert!(stub.downcast_clone::<2, Float>().is_none());

        assert!(stub.downcast_clone::<3, Bool>().is_none());

        let clone = stub.downcast_clone::<2, Bool>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);

        stub.clone()
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().flatten::<1>(0, 1).to_data(), true);
    }

    #[test]
    fn test_clone() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        let stub_clone = stub.clone();

        assert!(stub_clone.downcast_clone::<3, Float>().is_none());
        assert!(stub_clone.downcast_clone::<2, Int>().is_none());
        let clone = stub_clone.downcast_clone::<2, Float>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);
    }

    #[test]
    fn test_slice() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        let slice = stub.slice(s![.., 1..]).unwrap();
        assert_eq!(slice.shape(), [2, 2].into());
        slice
            .downcast_clone::<2, Float>()
            .unwrap()
            .to_data()
            .assert_eq(&source.clone().slice(s![.., 1..]).to_data(), true);
    }

    #[test]
    fn test_slice_dyn() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        let slice = stub
            .slice_dyn(&vec![Slice::new(0, None, 1), Slice::new(1, None, 1)])
            .unwrap();
        assert_eq!(slice.shape(), [2, 2].into());
        slice
            .downcast_clone::<2, Float>()
            .unwrap()
            .to_data()
            .assert_eq(&source.clone().slice(s![.., 1..]).to_data(), true);
    }
}
