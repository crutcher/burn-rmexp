//! # Tensor Metadata

use serde::{Serialize, Deserialize};
use burn::tensor::DType;
use burn::prelude::Shape;

/// A Serializable [`burn::prelude::TensorMetadata`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedShape {
    /// The data type of the tensor.
    pub dtype: DType,

    /// The shape of the tensor.
    pub shape: Shape,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_shape() {
        let shape = TypedShape { dtype: DType::F32, shape: Shape::new([1, 2, 3]) };

        assert_eq!(shape.dtype, DType::F32);
        assert_eq!(shape.shape.rank(), 3);
    }

}