use crate::{HEAD_DIM, N_KV_HEADS};
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use num_traits::cast::ToPrimitive;

pub struct RotatingBufferCache<B: Backend> {
    pub cache_k: Tensor<B, 5>,
    pub cache_v: Tensor<B, 5>,
    // N_LAYERS,
    // MAX_BATCHES,
    // sliding_window,
    // N_KV_HEADS,
    // HEAD_DIM
    pub kv_seqlens: Vec<usize>, // the sizes (seqlens) of each batch in the cache
}

#[derive(Clone)]
pub struct CacheView<B: Backend> {
    pub cache_k: Tensor<B, 4>,
    pub cache_v: Tensor<B, 4>,
    pub kv_seqlens: Vec<usize>, // the sizes (seqlens) of each batch in the cache
    pub seqlens: Vec<usize>,    // the sizes (seqlens) of each batch to be added to the cache
    pub cache_positions: Vec<usize>,
    pub prefill: bool,
    pub mask: Tensor<B, 3>,
}

impl<B: Backend> RotatingBufferCache<B> {
    pub fn get_view(
        &self,
        layer_id: usize,
        metadata: RotatingCacheInputMetadata<B>,
    ) -> CacheView<B> {
        let d = self.cache_k.dims();
        CacheView {
            cache_k: self
                .cache_k
                .clone()
                .slice([layer_id..layer_id, 0..d[1], 0..d[2], 0..d[3], 0..d[4]])
                .squeeze(0),
            cache_v: self
                .cache_v
                .clone()
                .slice([layer_id..layer_id, 0..d[1], 0..d[2], 0..d[3], 0..d[4]])
                .squeeze(0),
            kv_seqlens: self.kv_seqlens.clone(),
            metadata,
        }
    }
}

impl<B: Backend> CacheView<B> {
    pub fn update(&mut self, xk: Tensor<B, 3>, xv: Tensor<B, 3>) {
        // n_kv_heads, head_dim = self.cache_k.shape[-2:]
        // flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        // flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        // flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        // flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])
        let flat_cache_k = self
            .clone()
            .cache_k
            .reshape([-1, N_KV_HEADS as i32, HEAD_DIM as i32]);
        let flat_cache_v = self
            .clone()
            .cache_v
            .reshape([-1, N_KV_HEADS as i32, HEAD_DIM as i32]);

        for pos in self.metadata.cache_positions.clone().iter_dim(0) {
            let pos = pos.into_scalar().to_usize().unwrap();
        }
    }
}
