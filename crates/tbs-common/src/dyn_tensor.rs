use crate::clone_box::CloneBox;
use crate::index::SlicesError;
use crate::kind::KindFlag;
use crate::{index, tensor_util};
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

macro_rules! cast_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal, $dtype:tt) => {{
        let target_kind: KindFlag = $dtype.into();
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                Ok(match target_kind {
                    KindFlag::Float => $self_type::new(tensor.cast($dtype)),
                    KindFlag::Int => $self_type::new(tensor.int().cast($dtype)),
                    KindFlag::Bool => $self_type::new(tensor.bool()),
                })
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                Ok(match target_kind {
                    KindFlag::Float => $self_type::new(tensor.float().cast($dtype)),
                    KindFlag::Int => $self_type::new(tensor.cast($dtype)),
                    KindFlag::Bool => $self_type::new(tensor.bool()),
                })
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                Ok(match target_kind {
                    KindFlag::Float => $self_type::new(tensor.float().cast($dtype)),
                    KindFlag::Int => $self_type::new(tensor.int().cast($dtype)),
                    KindFlag::Bool => $self,
                })
            }
        }
    }};
}

macro_rules! slice_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal, $( $slices:tt ),*) => {
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor.slice($($slices)*)))
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor.slice($($slices)*)))
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor.slice($($slices)*)))
            }
        }
    };
}

macro_rules! slice_dyn_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal, $slices:tt) => {
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor_util::slice_dyn(tensor, $slices)))
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor_util::slice_dyn(tensor, $slices)))
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor_util::slice_dyn(tensor, $slices)))
            }
        }
    };
}

macro_rules! slice_assign_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal, $slices:tt, $values:tt) => {
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                let values: Tensor<B, $const_rank, Float> = $values.downcast_clone().unwrap();
                Ok($self_type::new(tensor.slice_assign($slices, values)))
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                let values: Tensor<B, $const_rank, Int> = $values.downcast_clone().unwrap();
                Ok($self_type::new(tensor.slice_assign($slices, values)))
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                let values: Tensor<B, $const_rank, Bool> = $values.downcast_clone().unwrap();
                Ok($self_type::new(tensor.slice_assign($slices, values)))
            }
        }
    };
}

macro_rules! slice_assign_dyn_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal, $slices:tt, $values:tt) => {{
        let slices: [Slice; $const_rank] = $slices.try_into().unwrap();
        $self.slice_assign(slices, $values)
    }};
}

macro_rules! flatten_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal) => {
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor.flatten::<1>(0, $self.rank() - 1)))
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor.flatten::<1>(0, $self.rank() - 1)))
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor.flatten::<1>(0, $self.rank() - 1)))
            }
        }
    };
}

macro_rules! to_device_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal, $device:tt) => {
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                let tensor = tensor.to_device($device);
                Ok($self_type::new(tensor))
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                let tensor = tensor.to_device($device);
                Ok($self_type::new(tensor))
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                let tensor = tensor.to_device($device);
                Ok($self_type::new(tensor))
            }
        }
    };
}

macro_rules! from_data_rank_case {
    ($const_rank:literal, $data:tt, $kind:tt, $device:tt) => {
        match $kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = Tensor::from_data($data, $device);
                Ok(DynTensor::new(tensor))
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = Tensor::from_data($data, $device);
                Ok(DynTensor::new(tensor))
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = Tensor::from_data($data, $device);
                Ok(DynTensor::new(tensor))
            }
        }
    };
}

macro_rules! to_data_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal) => {
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                Ok(tensor.to_data())
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                Ok(tensor.to_data())
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                Ok(tensor.to_data())
            }
        }
    };
}

pub enum ValuesArg<B: Backend> {
    Dyn(DynTensor<B>),
    Data(TensorData),
}

impl<B: Backend> Into<ValuesArg<B>> for DynTensor<B> {
    fn into(self) -> ValuesArg<B> {
        ValuesArg::Dyn(self)
    }
}

impl<B: Backend> Into<ValuesArg<B>> for TensorData {
    fn into(self) -> ValuesArg<B> {
        ValuesArg::Data(self)
    }
}

impl<B: Backend, const R: usize, K: BasicOps<B> + 'static> Into<ValuesArg<B>> for Tensor<B, R, K> {
    fn into(self) -> ValuesArg<B> {
        ValuesArg::Dyn(DynTensor::new(self))
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

impl<B: Backend, const R: usize, K: BasicOps<B> + 'static> Into<DynTensor<B>> for Tensor<B, R, K> {
    fn into(self) -> DynTensor<B> {
        DynTensor::new(self)
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
        &self,
        slices: S,
    ) -> Result<Self, DynTensorError>
    where
        S: SliceArg<R>,
    {
        let rank = self.rank();
        let slices = self.shape().into_slices(slices);

        index::check_slices_bounds(&self.shape(), &slices).map_err(DynTensorError::SliceError)?;

        match rank {
            1 => slice_rank_case!(self, Self, 1, slices),
            2 => slice_rank_case!(self, Self, 2, slices),
            3 => slice_rank_case!(self, Self, 3, slices),
            4 => slice_rank_case!(self, Self, 4, slices),
            5 => slice_rank_case!(self, Self, 5, slices),
            6 => slice_rank_case!(self, Self, 6, slices),
            7 => slice_rank_case!(self, Self, 7, slices),
            8 => slice_rank_case!(self, Self, 8, slices),
            9 => slice_rank_case!(self, Self, 9, slices),
            10 => slice_rank_case!(self, Self, 10, slices),
            11 => slice_rank_case!(self, Self, 11, slices),
            12 => slice_rank_case!(self, Self, 12, slices),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("slice rank ({}) is not supported", R),
                rank,
            }),
        }
    }

    /// A dynamic version of [`DynTensor::slice`].
    ///
    /// Generated up to rank 12.
    pub fn slice_dyn(
        &self,
        slices: &[Slice],
    ) -> Result<Self, DynTensorError> {
        let rank = self.rank();

        index::check_slices_bounds(&self.shape(), slices).map_err(DynTensorError::SliceError)?;

        match rank {
            1 => slice_dyn_rank_case!(self, Self, 1, slices),
            2 => slice_dyn_rank_case!(self, Self, 2, slices),
            3 => slice_dyn_rank_case!(self, Self, 3, slices),
            4 => slice_dyn_rank_case!(self, Self, 4, slices),
            5 => slice_dyn_rank_case!(self, Self, 5, slices),
            6 => slice_dyn_rank_case!(self, Self, 6, slices),
            7 => slice_dyn_rank_case!(self, Self, 7, slices),
            8 => slice_dyn_rank_case!(self, Self, 8, slices),
            9 => slice_dyn_rank_case!(self, Self, 9, slices),
            10 => slice_dyn_rank_case!(self, Self, 10, slices),
            11 => slice_dyn_rank_case!(self, Self, 11, slices),
            12 => slice_dyn_rank_case!(self, Self, 12, slices),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("slice rank ({}) is not supported", rank),
                rank,
            }),
        }
    }

    /// Assign values to a slice.
    ///
    /// Generated up to rank 12.
    pub fn slice_assign<const R2: usize, S, V>(
        &self,
        slices: S,
        values: V,
    ) -> Result<Self, DynTensorError>
    where
        S: SliceArg<R2>,
        V: Into<ValuesArg<B>>,
    {
        let rank = self.rank();
        let slices = self.shape().into_slices(slices);
        let values: DynTensor<B> = match values.into() {
            ValuesArg::Dyn(values) => values,
            ValuesArg::Data(values) => DynTensor::from_data(values, &self.device())?,
        };

        index::check_slices_bounds(&self.shape(), &slices).map_err(DynTensorError::SliceError)?;

        if rank != values.rank() {
            return Err(DynTensorError::InvalidArgument {
                msg: format!(
                    "slice of rank ({}) cannot be assigned to tensor of rank ({})",
                    values.rank(),
                    rank
                ),
            });
        }

        let values = values.cast(self.dtype())?.to_device(&self.device())?;

        // TODO: check that slices shape == source.shape

        match rank {
            1 => slice_assign_rank_case!(self, Self, 1, slices, values),
            2 => slice_assign_rank_case!(self, Self, 2, slices, values),
            3 => slice_assign_rank_case!(self, Self, 3, slices, values),
            4 => slice_assign_rank_case!(self, Self, 4, slices, values),
            5 => slice_assign_rank_case!(self, Self, 5, slices, values),
            6 => slice_assign_rank_case!(self, Self, 6, slices, values),
            7 => slice_assign_rank_case!(self, Self, 7, slices, values),
            8 => slice_assign_rank_case!(self, Self, 8, slices, values),
            9 => slice_assign_rank_case!(self, Self, 9, slices, values),
            10 => slice_assign_rank_case!(self, Self, 10, slices, values),
            11 => slice_assign_rank_case!(self, Self, 11, slices, values),
            12 => slice_assign_rank_case!(self, Self, 12, slices, values),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("tensor rank ({}) is not supported", rank),
                rank,
            }),
        }
    }

    /// Dynamic slice rank version of [`DynTensor::slice_assign`].
    ///
    /// Generated up to rank=12.
    pub fn slice_assign_dyn<V>(
        &self,
        slices: &[Slice],
        values: V,
    ) -> Result<Self, DynTensorError>
    where
        V: Into<ValuesArg<B>>,
    {
        let s_rank = slices.len();

        match s_rank {
            1 => slice_assign_dyn_rank_case!(self, Self, 1, slices, values),
            2 => slice_assign_dyn_rank_case!(self, Self, 2, slices, values),
            3 => slice_assign_dyn_rank_case!(self, Self, 3, slices, values),
            4 => slice_assign_dyn_rank_case!(self, Self, 4, slices, values),
            5 => slice_assign_dyn_rank_case!(self, Self, 5, slices, values),
            6 => slice_assign_dyn_rank_case!(self, Self, 6, slices, values),
            7 => slice_assign_dyn_rank_case!(self, Self, 7, slices, values),
            8 => slice_assign_dyn_rank_case!(self, Self, 8, slices, values),
            9 => slice_assign_dyn_rank_case!(self, Self, 9, slices, values),
            10 => slice_assign_dyn_rank_case!(self, Self, 10, slices, values),
            11 => slice_assign_dyn_rank_case!(self, Self, 11, slices, values),
            12 => slice_assign_dyn_rank_case!(self, Self, 12, slices, values),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("slices rank ({}) is not supported", s_rank),
                rank: s_rank,
            }),
        }
    }

    /// Flatten the tensor.
    ///
    /// Generated up to rank 12.
    pub fn flatten(self) -> Result<Self, DynTensorError> {
        let rank = self.rank();

        match rank {
            1 => flatten_rank_case!(self, Self, 1),
            2 => flatten_rank_case!(self, Self, 2),
            3 => flatten_rank_case!(self, Self, 3),
            4 => flatten_rank_case!(self, Self, 4),
            5 => flatten_rank_case!(self, Self, 5),
            6 => flatten_rank_case!(self, Self, 6),
            7 => flatten_rank_case!(self, Self, 7),
            8 => flatten_rank_case!(self, Self, 8),
            9 => flatten_rank_case!(self, Self, 9),
            10 => flatten_rank_case!(self, Self, 10),
            11 => flatten_rank_case!(self, Self, 11),
            12 => flatten_rank_case!(self, Self, 12),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("flatten rank ({}) is not supported", rank),
                rank,
            }),
        }
    }

    /// Cast the tensor.
    ///
    /// Generated up to rank 12.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Result<Self, DynTensorError> {
        let rank = self.rank();

        match rank {
            1 => cast_rank_case!(self, Self, 1, dtype),
            2 => cast_rank_case!(self, Self, 2, dtype),
            3 => cast_rank_case!(self, Self, 3, dtype),
            4 => cast_rank_case!(self, Self, 4, dtype),
            5 => cast_rank_case!(self, Self, 5, dtype),
            6 => cast_rank_case!(self, Self, 6, dtype),
            7 => cast_rank_case!(self, Self, 7, dtype),
            8 => cast_rank_case!(self, Self, 8, dtype),
            9 => cast_rank_case!(self, Self, 9, dtype),
            10 => cast_rank_case!(self, Self, 10, dtype),
            11 => cast_rank_case!(self, Self, 11, dtype),
            12 => cast_rank_case!(self, Self, 12, dtype),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("cast rank ({}) is not supported", rank),
                rank,
            }),
        }
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
        let rank = self.rank();

        match rank {
            1 => to_device_rank_case!(self, Self, 1, device),
            2 => to_device_rank_case!(self, Self, 2, device),
            3 => to_device_rank_case!(self, Self, 3, device),
            4 => to_device_rank_case!(self, Self, 4, device),
            5 => to_device_rank_case!(self, Self, 5, device),
            6 => to_device_rank_case!(self, Self, 6, device),
            7 => to_device_rank_case!(self, Self, 7, device),
            8 => to_device_rank_case!(self, Self, 8, device),
            9 => to_device_rank_case!(self, Self, 9, device),
            10 => to_device_rank_case!(self, Self, 10, device),
            11 => to_device_rank_case!(self, Self, 11, device),
            12 => to_device_rank_case!(self, Self, 12, device),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("to_device rank ({}) is not supported", rank),
                rank,
            }),
        }
    }

    /// Convert a [`TensorData`] to a [`DynTensor`].
    ///
    /// Generated up to rank 12.
    pub fn from_data(
        data: TensorData,
        device: &B::Device,
    ) -> Result<Self, DynTensorError> {
        let rank = data.rank();
        let kind: KindFlag = data.dtype.into();

        match rank {
            1 => from_data_rank_case!(1, data, kind, device),
            2 => from_data_rank_case!(2, data, kind, device),
            3 => from_data_rank_case!(3, data, kind, device),
            4 => from_data_rank_case!(4, data, kind, device),
            5 => from_data_rank_case!(5, data, kind, device),
            6 => from_data_rank_case!(6, data, kind, device),
            7 => from_data_rank_case!(7, data, kind, device),
            8 => from_data_rank_case!(8, data, kind, device),
            9 => from_data_rank_case!(9, data, kind, device),
            10 => from_data_rank_case!(10, data, kind, device),
            11 => from_data_rank_case!(11, data, kind, device),
            12 => from_data_rank_case!(12, data, kind, device),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("from_data rank ({}) is not supported", rank),
                rank,
            }),
        }
    }

    /// Convert the tensor to a [`TensorData`].
    ///
    /// Generated up to rank 12.
    pub fn to_data(self) -> Result<TensorData, DynTensorError> {
        let rank = self.rank();

        match rank {
            1 => to_data_rank_case!(self, Self, 1),
            2 => to_data_rank_case!(self, Self, 2),
            3 => to_data_rank_case!(self, Self, 3),
            4 => to_data_rank_case!(self, Self, 4),
            5 => to_data_rank_case!(self, Self, 5),
            6 => to_data_rank_case!(self, Self, 6),
            7 => to_data_rank_case!(self, Self, 7),
            8 => to_data_rank_case!(self, Self, 8),
            9 => to_data_rank_case!(self, Self, 9),
            10 => to_data_rank_case!(self, Self, 10),
            11 => to_data_rank_case!(self, Self, 11),
            12 => to_data_rank_case!(self, Self, 12),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("to_data rank ({}) is not supported", rank),
                rank,
            }),
        }
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
