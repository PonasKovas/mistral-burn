use burn::{
    module::Module,
    nn::Linear,
    tensor::{activation::silu, backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
    pub w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, tensor: Tensor<B, 2>) -> Tensor<B, 2> {
        let temp = silu(self.w1.forward(tensor.clone()));

        self.w2.forward(temp * self.w3.forward(tensor))
    }
}
