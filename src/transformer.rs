use crate::{
    attention::Attention, feedforward::FeedForward, rmsnorm::RMSNormalization,
    rope::apply_rotary_emb, HEAD_DIM, N_HEADS, N_KV_HEADS,
};
use burn::{
    module::Module,
    nn::{Embedding, Linear},
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attention: Attention<B>,
    pub feed_forward: FeedForward<B>,
    pub attention_norm: RMSNormalization<B>,
    pub ffn_norm: RMSNormalization<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, tensor: Tensor<B, 2>, freqs_cis: Tensor<B, 3>) -> Tensor<B, 2> {
        let r = self
            .attention
            .forward(self.attention_norm.forward(tensor.clone()), freqs_cis);
        let h = tensor + r;

        let r = self.feed_forward.forward(self.ffn_norm.forward(h.clone()));
        let out = h + r;

        out
    }
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    pub embeddings: Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub norm: RMSNormalization<B>,
    pub output: Linear<B>,
    pub freqs_cis: Tensor<B, 3>,
}

impl<B: Backend> Transformer<B> {
    pub fn forward(&self, input_ids: Tensor<B, 1, Int>, seqlens: Vec<usize>) -> Tensor<B, 2> {
        let mut h = self
            .embeddings
            .forward(input_ids.unsqueeze_dim(0))
            .squeeze(0);

        let mut to_concat = Vec::new();
        for seqlen in seqlens {
            to_concat.push(self.freqs_cis.clone().slice([0..seqlen]));
        }
        let freqs_cis = Tensor::cat(to_concat, 0);

        for layer in &self.layers {
            h = layer.forward(h, freqs_cis.clone());
        }

        h = self.norm.forward(h);

        let out = self.output.forward(h);

        out
    }
}
