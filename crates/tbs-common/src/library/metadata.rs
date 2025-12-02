//! # Tensor Metadata

use burn::prelude::Shape;
use burn::tensor::DType;
use serde::{Deserialize, Serialize};

/// A Serializable `burn::prelude::TensorMetadata`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedShape {
    /// The data type of the tensor.
    pub dtype: DType,

    /// The shape of the tensor.
    pub shape: Shape,
}

impl TypedShape {
    /// Forward the rank of the shape.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Forward the number of elements of the shape.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_shape() {
        let shape = Shape::new([1, 2, 3]);

        let tshape = TypedShape {
            dtype: DType::F32,
            shape: shape.clone(),
        };

        assert_eq!(tshape.dtype, DType::F32);
        assert_eq!(tshape.shape.dims, shape.dims);

        assert_eq!(tshape.num_elements(), shape.num_elements());
        assert_eq!(tshape.rank(), shape.rank());
    }
}
