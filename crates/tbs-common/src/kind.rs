use burn::Tensor;
use burn::prelude::{Backend, Bool, Float, Int};
use burn::tensor::{DType, TensorKind};
use serde::{Deserialize, Serialize};
use std::any::Any;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KindError {
    pub msg: String,
}

/// A flag indicating the tensor kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KindFlag {
    Float,
    Int,
    Bool,
}

impl KindFlag {
    /// Returns the kind of the given tensor.
    pub fn kind<B: Backend, const R: usize, K: TensorKind<B> + 'static>(
        tensor: &Tensor<B, R, K>
    ) -> Result<Self, KindError> {
        let any: &dyn Any = tensor;

        if let Some(_) = any.downcast_ref::<Tensor<B, R, Float>>() {
            Ok(Self::Float)
        } else if let Some(_) = any.downcast_ref::<Tensor<B, R, Int>>() {
            Ok(Self::Int)
        } else if let Some(_) = any.downcast_ref::<Tensor<B, R, Bool>>() {
            Ok(Self::Bool)
        } else {
            Err(KindError {
                msg: format!("Unsupported tensor kind: {:?}", K::name()),
            })
        }
    }
}

impl Into<KindFlag> for DType {
    fn into(self) -> KindFlag {
        if self.is_float() {
            KindFlag::Float
        } else if self.is_int() {
            KindFlag::Int
        } else {
            KindFlag::Bool
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind() {
        type B = burn::backend::Wgpu;
        let device = Default::default();

        assert_eq!(
            KindFlag::kind(&Tensor::<B, 2, Float>::ones([2, 3], &device)).unwrap(),
            KindFlag::Float
        );
        assert_eq!(
            KindFlag::kind(&Tensor::<B, 2, Int>::ones([2, 3], &device)).unwrap(),
            KindFlag::Int
        );
        assert_eq!(
            KindFlag::kind(&Tensor::<B, 2, Bool>::ones([2, 3], &device)).unwrap(),
            KindFlag::Bool
        );
    }
}
