use crate::NORM_EPS;
use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct RMSNormalization<B: Backend> {
    pub weights: Param<Tensor<B, 1>>,
}

impl<B: Backend> RMSNormalization<B> {
    pub fn forward<const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let rms = (tensor.clone().powf(2.0).mean_dim(D - 1) + NORM_EPS).sqrt();

        tensor.div(rms) * self.weights.val().unsqueeze()
    }
}
