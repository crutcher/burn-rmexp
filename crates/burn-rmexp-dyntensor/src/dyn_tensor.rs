use crate::clone_box::CloneBox;
use crate::indexing;
use crate::indexing::SlicesError;
use crate::kind::KindFlag;
use crate::operations;
use burn::Tensor;
use burn::prelude::{Backend, Bool, Float, Int, Shape, SliceArg, TensorData};
use burn::tensor::{BasicOps, DType, Slice};

/// [`DynTensor`] accessor errors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DynTensorError {
    /// An error occurred while slicing.
    SliceError(SlicesError),

    /// Invalid Arguments.
    InvalidArgument { msg: String },

    /// The tensor rank is not supported for the requested operation.
    UnsupportedRank { msg: String, rank: usize },
}

/// Dynamic to static rank dispatch handler.
trait RankHandler {
    type Output;
    fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError>;
}

/// Dynamic rank dispatch.
fn dispatch_rank<H: RankHandler>(
    rank: usize,
    handler: H,
) -> Result<H::Output, DynTensorError> {
    match rank {
        1 => handler.call::<1>(),
        2 => handler.call::<2>(),
        3 => handler.call::<3>(),
        4 => handler.call::<4>(),
        5 => handler.call::<5>(),
        6 => handler.call::<6>(),
        7 => handler.call::<7>(),
        8 => handler.call::<8>(),
        9 => handler.call::<9>(),
        10 => handler.call::<10>(),
        11 => handler.call::<11>(),
        12 => handler.call::<12>(),
        _ => Err(DynTensorError::UnsupportedRank {
            msg: "unsupported rank".to_string(),
            rank,
        }),
    }
}

/// Values conversion trait for [`DynTensor::slice_assign`].
#[derive(Debug, Clone)]
pub enum ValuesArg<B: Backend> {
    /// A [`DynTensor`] wrapper.
    Dyn(DynTensor<B>),

    /// A [`TensorData`] wrapper.
    Data(TensorData),
}

impl<B: Backend> ValuesArg<B> {
    /// Convert to a [`DynTensor`] on a given device.
    fn to_dyn_device(
        self,
        device: &B::Device,
    ) -> Result<DynTensor<B>, DynTensorError> {
        match self {
            ValuesArg::Dyn(val) => val.to_device(device),
            ValuesArg::Data(val) => DynTensor::from_data(val, device),
        }
    }
}

impl<B: Backend> From<DynTensor<B>> for ValuesArg<B> {
    fn from(val: DynTensor<B>) -> Self {
        ValuesArg::Dyn(val)
    }
}

impl<B: Backend> From<TensorData> for ValuesArg<B> {
    fn from(val: TensorData) -> Self {
        ValuesArg::Data(val)
    }
}

impl<B: Backend, const R: usize, K: BasicOps<B> + 'static> From<Tensor<B, R, K>> for ValuesArg<B> {
    fn from(val: Tensor<B, R, K>) -> Self {
        ValuesArg::Dyn(DynTensor::new(val))
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

impl<B: Backend, const R: usize, K: BasicOps<B> + 'static> From<Tensor<B, R, K>> for DynTensor<B> {
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
    ///
    /// Either a `Some(Tensor)` clone (if the downcast is correct), or `None`.
    pub fn downcast_clone<const R: usize, K>(&self) -> Option<Tensor<B, R, K>>
    where
        K: 'static + BasicOps<B>,
    {
        self.tensor.downcast_ref::<Tensor<B, R, K>>().cloned()
    }

    /// Slice the stub tensor.
    ///
    /// Generated up to rank 12.
    ///
    /// # Arguments
    /// - `slices`: see [`Tensor::slice`].
    ///
    /// # Returns
    ///
    /// A sliced `TensorStub`, or an error.
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
            tensor: DynTensor<B>,
            slices: [Slice; R],
        }
        impl<'a, B: Backend, const R: usize> RankHandler for SliceHandler<B, R> {
            type Output = DynTensor<B>;
            fn call<const R2: usize>(self) -> Result<Self::Output, DynTensorError> {
                match self.tensor.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(tensor.slice(self.slices)))
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(tensor.slice(self.slices)))
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(tensor.slice(self.slices)))
                    }
                }
            }
        }
        dispatch_rank(
            rank,
            SliceHandler {
                tensor: self,
                slices,
            },
        )
    }

    /// A dynamic version of [`DynTensor::slice`].
    ///
    /// Generated up to rank 12.
    pub fn slice_dyn(
        self,
        slices: &[Slice],
    ) -> Result<Self, DynTensorError> {
        let rank = self.rank();

        indexing::check_slices_bounds(&self.shape(), slices).map_err(DynTensorError::SliceError)?;

        struct SliceDynHandler<'a, B: Backend> {
            tensor: DynTensor<B>,
            slices: &'a [Slice],
        }
        impl<'a, B: Backend> RankHandler for SliceDynHandler<'a, B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                match self.tensor.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(operations::slice_dyn(tensor, self.slices)))
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(operations::slice_dyn(tensor, self.slices)))
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(operations::slice_dyn(tensor, self.slices)))
                    }
                }
            }
        }
        dispatch_rank(
            rank,
            SliceDynHandler {
                tensor: self,
                slices,
            },
        )
    }

    /// Assign values to a slice.
    ///
    /// Generated up to rank 12.
    pub fn slice_assign<const R2: usize, S, V>(
        self,
        slices: S,
        values: V,
    ) -> Result<Self, DynTensorError>
    where
        S: SliceArg<R2>,
        V: Into<ValuesArg<B>>,
    {
        let rank = self.rank();
        let slices = self.shape().into_slices(slices);
        let values: DynTensor<B> = values.into().to_dyn_device(&self.device())?;

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
            tensor: DynTensor<B>,
            slices: [Slice; R2],
            values: DynTensor<B>,
        }
        impl<B: Backend, const R2: usize> RankHandler for SliceAssignHandler<B, R2> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                match self.tensor.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.tensor.downcast_clone().unwrap();
                        let values: Tensor<B, R, Float> = self.values.downcast_clone().unwrap();
                        Ok(DynTensor::new(tensor.slice_assign(self.slices, values)))
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.tensor.downcast_clone().unwrap();
                        let values: Tensor<B, R, Int> = self.values.downcast_clone().unwrap();
                        Ok(DynTensor::new(tensor.slice_assign(self.slices, values)))
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.tensor.downcast_clone().unwrap();
                        let values: Tensor<B, R, Bool> = self.values.downcast_clone().unwrap();
                        Ok(DynTensor::new(tensor.slice_assign(self.slices, values)))
                    }
                }
            }
        }
        dispatch_rank(
            rank,
            SliceAssignHandler {
                tensor: self.clone(),
                slices,
                values,
            },
        )
    }

    /// Dynamic slice rank version of [`DynTensor::slice_assign`].
    ///
    /// Generated up to rank=12.
    pub fn slice_assign_dyn<V>(
        self,
        slices: &[Slice],
        values: V,
    ) -> Result<Self, DynTensorError>
    where
        V: Into<ValuesArg<B>>,
    {
        struct SliceAssignDynHandler<'a, B: Backend> {
            tensor: DynTensor<B>,
            slices: &'a [Slice],
            values: ValuesArg<B>,
        }
        impl<'a, B: Backend> RankHandler for SliceAssignDynHandler<'a, B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                let slices: [Slice; R] = self.slices.try_into().unwrap();
                self.tensor.slice_assign(slices, self.values)
            }
        }
        dispatch_rank(
            self.rank(),
            SliceAssignDynHandler {
                tensor: self,
                slices,
                values: values.into(),
            },
        )
    }

    /// Flatten the tensor.
    ///
    /// Generated up to rank 12.
    pub fn flatten(self) -> Result<Self, DynTensorError> {
        struct FlattenHandler<B: Backend> {
            tensor: DynTensor<B>,
        }
        impl<B: Backend> RankHandler for FlattenHandler<B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                match self.tensor.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(
                            tensor.flatten::<1>(0, self.tensor.rank() - 1),
                        ))
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(
                            tensor.flatten::<1>(0, self.tensor.rank() - 1),
                        ))
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.tensor.downcast_clone().unwrap();
                        Ok(DynTensor::new(
                            tensor.flatten::<1>(0, self.tensor.rank() - 1),
                        ))
                    }
                }
            }
        }
        dispatch_rank(self.rank(), FlattenHandler { tensor: self })
    }

    /// Cast the tensor.
    ///
    /// Auto-converts kind if necessary.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Result<Self, DynTensorError> {
        struct CastHandler<B: Backend> {
            tensor: DynTensor<B>,
            dtype: DType,
        }
        impl<B: Backend> RankHandler for CastHandler<B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                let target_kind: KindFlag = self.dtype.into();
                match self.tensor.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.tensor.downcast_clone().unwrap();
                        Ok(match target_kind {
                            KindFlag::Float => DynTensor::new(tensor.cast(self.dtype)),
                            KindFlag::Int => DynTensor::new(tensor.int().cast(self.dtype)),
                            KindFlag::Bool => DynTensor::new(tensor.bool()),
                        })
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.tensor.downcast_clone().unwrap();
                        Ok(match target_kind {
                            KindFlag::Float => DynTensor::new(tensor.float().cast(self.dtype)),
                            KindFlag::Int => DynTensor::new(tensor.cast(self.dtype)),
                            KindFlag::Bool => DynTensor::new(tensor.bool()),
                        })
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.tensor.downcast_clone().unwrap();
                        Ok(match target_kind {
                            KindFlag::Float => DynTensor::new(tensor.float().cast(self.dtype)),
                            KindFlag::Int => DynTensor::new(tensor.int().cast(self.dtype)),
                            KindFlag::Bool => self.tensor,
                        })
                    }
                }
            }
        }
        dispatch_rank(
            self.rank(),
            CastHandler {
                tensor: self,
                dtype,
            },
        )
    }

    /// Move the tensor to the given device.
    ///
    /// Generated up to rank 12.
    pub fn to_device(
        self,
        device: &B::Device,
    ) -> Result<Self, DynTensorError> {
        if &self.device() == device {
            return Ok(self);
        }

        struct ToDeviceHandler<'a, B: Backend> {
            tensor: DynTensor<B>,
            device: &'a B::Device,
        }
        impl<'a, B: Backend> RankHandler for ToDeviceHandler<'a, B> {
            type Output = DynTensor<B>;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                match self.tensor.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.tensor.downcast_clone().unwrap();
                        let tensor = tensor.to_device(self.device);
                        Ok(DynTensor::new(tensor))
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.tensor.downcast_clone().unwrap();
                        let tensor = tensor.to_device(self.device);
                        Ok(DynTensor::new(tensor))
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.tensor.downcast_clone().unwrap();
                        let tensor = tensor.to_device(self.device);
                        Ok(DynTensor::new(tensor))
                    }
                }
            }
        }
        dispatch_rank(
            self.rank(),
            ToDeviceHandler {
                tensor: self,
                device,
            },
        )
    }

    /// Convert a [`TensorData`] to a [`DynTensor`].
    ///
    /// Generated up to rank 12.
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
                match kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = Tensor::from_data(self.data, self.device);
                        Ok(DynTensor::new(tensor))
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = Tensor::from_data(self.data, self.device);
                        Ok(DynTensor::new(tensor))
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = Tensor::from_data(self.data, self.device);
                        Ok(DynTensor::new(tensor))
                    }
                }
            }
        }
        dispatch_rank(data.rank(), FromDataHandler { data, device })
    }

    /// Convert the tensor to a [`TensorData`].
    ///
    /// Generated up to rank 12.
    pub fn to_data(self) -> Result<TensorData, DynTensorError> {
        struct ToDataHandler<B: Backend> {
            tensor: DynTensor<B>,
        }
        impl<B: Backend> RankHandler for ToDataHandler<B> {
            type Output = TensorData;
            fn call<const R: usize>(self) -> Result<Self::Output, DynTensorError> {
                match self.tensor.kind {
                    KindFlag::Float => {
                        let tensor: Tensor<B, R, Float> = self.tensor.downcast_clone().unwrap();
                        Ok(tensor.to_data())
                    }
                    KindFlag::Int => {
                        let tensor: Tensor<B, R, Int> = self.tensor.downcast_clone().unwrap();
                        Ok(tensor.to_data())
                    }
                    KindFlag::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.tensor.downcast_clone().unwrap();
                        Ok(tensor.to_data())
                    }
                }
            }
        }
        dispatch_rank(self.rank(), ToDataHandler { tensor: self })
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
            .to_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .to_data()
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
            .to_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .to_data()
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
            .to_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .to_data()
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
