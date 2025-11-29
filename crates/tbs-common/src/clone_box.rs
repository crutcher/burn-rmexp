use std::any::Any;
use std::fmt::Debug;

/// A trait for cloning values into a boxed form.
pub trait CloneBoxDebugSendAny: Debug + Send + Any {
    fn clone_box(&self) -> Box<dyn CloneBoxDebugSendAny>;
}

impl dyn CloneBoxDebugSendAny {
    /// Downcasts the boxed value to a specific type.
    ///
    /// See: [`Any::downcast_ref`].
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref::<T>()
    }
}

impl<T: Clone + Debug + Send + Any> CloneBoxDebugSendAny for T {
    fn clone_box(&self) -> Box<dyn CloneBoxDebugSendAny> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CloneBoxDebugSendAny> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use burn::backend::Wgpu;
    use burn::tensor::Distribution;

    fn assert_send<T: Send>() {}

    #[test]
    fn test_clone_box_tensor() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let boxed: Box<dyn CloneBoxDebugSendAny> = Box::new(source.clone());

        assert_send::<Box<dyn CloneBoxDebugSendAny>>();

        let cloned_box = boxed.clone();

        let clone = cloned_box.downcast_ref::<Tensor<B, 2>>().unwrap();

        clone.to_data().assert_eq(&source.to_data(), true);
    }
}
