//! # Tokio-Block-Stream Common

pub mod clone_box;
pub mod dyn_provider;
pub mod dyn_tensor;
pub mod index;
pub mod kind;
pub mod library;
pub mod tensor_library;
pub mod tensor_util;

/// add two numbers
pub fn add(
    left: u64,
    right: u64,
) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
