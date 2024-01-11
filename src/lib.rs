use sentencepiece::SentencePieceProcessor;
use std::path::Path;
use thiserror::Error;

pub mod attention;
// pub mod cache;
pub mod feedforward;
pub mod rmsnorm;
pub mod rope;
pub mod transformer;

// word embedding dimensions size
pub const DIM: usize = 4096;
// word embedding size in the attention layer
pub const HEAD_DIM: usize = 128;

// number of sequential transformer blocks
pub const N_LAYERS: usize = 32;
// number of parallel attention heads for Query
pub const N_HEADS: usize = 32;
// number of parallel attention heads for Key and Value
pub const N_KV_HEADS: usize = 8;

// internal of the feedforward layer
pub const HIDDEN_DIM: usize = 14336;

pub const NORM_EPS: f64 = 1e-05;
pub const SLIDING_WINDOW: usize = 4096;
pub const VOCAB_SIZE: usize = 32000;

pub struct Mistral {
    tokenizer: SentencePieceProcessor,
    model: (),
}

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("error loading the tokenizer")]
    LoadTokenizerError(#[from] sentencepiece::SentencePieceError),
}

impl Mistral {
    pub fn load(
        tokenizer_path: impl AsRef<Path>,
        mistral_path: impl AsRef<Path>,
    ) -> Result<Self, LoadError> {
        let tokenizer = SentencePieceProcessor::open(tokenizer_path)?;

        todo!()
    }
}
