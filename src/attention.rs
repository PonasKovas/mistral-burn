use burn::{
    module::Module,
    nn::Linear,
    tensor::{activation::softmax, backend::Backend, Tensor},
};

use crate::{rope::apply_rotary_emb, HEAD_DIM, N_HEADS, N_KV_HEADS};

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wo: Linear<B>,
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, tensor: Tensor<B, 2>, freqs_cis: Tensor<B, 3>) -> Tensor<B, 2> {
        let seqlen_sum = tensor.dims()[0];

        let xq = self
            .wq
            .forward(tensor.clone())
            .reshape([seqlen_sum, N_HEADS, HEAD_DIM]);
        let xk = self
            .wk
            .forward(tensor.clone())
            .reshape([seqlen_sum, N_KV_HEADS, HEAD_DIM]);
        let xv = self
            .wv
            .forward(tensor)
            .reshape([seqlen_sum, N_KV_HEADS, HEAD_DIM]);

        let xq = apply_rotary_emb(xq, freqs_cis.clone());
        let xk = apply_rotary_emb(xk, freqs_cis);

        // repeat_interleave workaround
        let keys: Tensor<B, 3> = Tensor::stack::<4>(
            [(); N_HEADS / N_KV_HEADS]
                .into_iter()
                .map(|_| xk.clone())
                .collect::<Vec<_>>(),
            2,
        )
        .flatten(1, 2);
        let values: Tensor<B, 3> = Tensor::stack::<4>(
            [(); N_HEADS / N_KV_HEADS]
                .into_iter()
                .map(|_| xv.clone())
                .collect::<Vec<_>>(),
            2,
        )
        .flatten(1, 2);

        // attention
        let scale = (xq.dims()[2] as f64).powf(-0.5);
        let query = xq * scale;
        let attention = query.matmul(keys.swap_dims(1, 2));
        let attention = softmax(attention, 2);
        // dropout ommited here for now
        let attention = attention.matmul(values);

        let out = self.wo.forward(attention.reshape([seqlen_sum as i32, -1]));

        out
    }
}
