use crate::clone_box::CloneBoxDebugSendAny;
use crate::index::SlicesError;
use crate::kind::KindFlag;
use crate::{index, tensor_util};
use burn::Tensor;
use burn::prelude::{Backend, Bool, Float, Int, Shape, SliceArg, TensorData};
use burn::tensor::{BasicOps, DType, Slice};

/// [`DynTensor`] accessor errors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DynTensorError {
    SliceError(SlicesError),
    UnsupportedRank { msg: String, rank: usize },
}

/// A dynamic [`Tensor`] wrapper that can be sliced.
#[derive(Debug, Clone)]
pub struct DynTensor<B: Backend> {
    shape: Shape,
    dtype: DType,
    kind: KindFlag,
    device: B::Device,
    tensor: Box<dyn CloneBoxDebugSendAny>,
    phantom: std::marker::PhantomData<B>,
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
            kind: KindFlag::kind(&tensor).unwrap(),
            device: tensor.device(),
            tensor: Box::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Get the tensor rank.
    pub fn rank(&self) -> usize {
        self.shape.rank()
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

impl<B: Backend> DynTensor<B> {
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
                msg: format!("Selection rank {} is not supported", R),
                rank,
            }),
        }
    }
}

macro_rules! dyn_slice_rank_case {
    ($self:tt, $self_type:tt, $const_rank:literal, $slices:tt) => {
        match $self.kind {
            KindFlag::Float => {
                let tensor: Tensor<B, $const_rank, Float> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor_util::dyn_slice(tensor, $slices)))
            }
            KindFlag::Int => {
                let tensor: Tensor<B, $const_rank, Int> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor_util::dyn_slice(tensor, $slices)))
            }
            KindFlag::Bool => {
                let tensor: Tensor<B, $const_rank, Bool> = $self.downcast_clone().unwrap();
                Ok($self_type::new(tensor_util::dyn_slice(tensor, $slices)))
            }
        }
    };
}

impl<B: Backend> DynTensor<B> {
    /// A dynamic version of [`DynTensor::slice`].
    ///
    /// Generated up to rank 12.
    pub fn dyn_slice(
        &self,
        slices: &[Slice],
    ) -> Result<Self, DynTensorError> {
        let rank = self.rank();

        index::check_slices_bounds(&self.shape(), slices).map_err(DynTensorError::SliceError)?;

        match rank {
            1 => dyn_slice_rank_case!(self, Self, 1, slices),
            2 => dyn_slice_rank_case!(self, Self, 2, slices),
            3 => dyn_slice_rank_case!(self, Self, 3, slices),
            4 => dyn_slice_rank_case!(self, Self, 4, slices),
            5 => dyn_slice_rank_case!(self, Self, 5, slices),
            6 => dyn_slice_rank_case!(self, Self, 6, slices),
            7 => dyn_slice_rank_case!(self, Self, 7, slices),
            8 => dyn_slice_rank_case!(self, Self, 8, slices),
            9 => dyn_slice_rank_case!(self, Self, 9, slices),
            10 => dyn_slice_rank_case!(self, Self, 10, slices),
            11 => dyn_slice_rank_case!(self, Self, 11, slices),
            12 => dyn_slice_rank_case!(self, Self, 12, slices),
            _ => Err(DynTensorError::UnsupportedRank {
                msg: format!("Selection rank ({}) is not supported", rank),
                rank,
            }),
        }
    }
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

impl<B: Backend> DynTensor<B> {
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
                msg: format!("Flatten rank ({}) is not supported", rank),
                rank,
            }),
        }
    }
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

impl<B: Backend> DynTensor<B> {
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
                msg: format!("Flatten rank ({}) is not supported", rank),
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

        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.rank(), 2);

        assert_eq!(stub.dtype(), source.dtype());
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

        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.rank(), 2);

        assert_eq!(stub.dtype(), source.dtype());
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

        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.rank(), 2);

        assert_eq!(stub.dtype(), source.dtype());
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
    fn test_dyn_slice() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        let slice = stub
            .dyn_slice(&vec![Slice::new(0, None, 1), Slice::new(1, None, 1)])
            .unwrap();
        assert_eq!(slice.shape(), [2, 2].into());
        slice
            .downcast_clone::<2, Float>()
            .unwrap()
            .to_data()
            .assert_eq(&source.clone().slice(s![.., 1..]).to_data(), true);
    }
}
