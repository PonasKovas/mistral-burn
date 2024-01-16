use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    iter,
    path::Path,
};

use anyhow::{anyhow, bail};
use burn::{
    backend::{NdArray, Wgpu},
    module::{Module, Param},
    nn::{EmbeddingConfig, EmbeddingRecord, Linear},
    tensor::{activation::softmax, backend::Backend, Data, Float, Int, Tensor},
};
use mistral_burn::{
    attention::Attention,
    feedforward::FeedForward,
    rmsnorm::RMSNormalization,
    rope::precompute_freqs_cis,
    transformer::{Transformer, TransformerBlock},
    DIM, HEAD_DIM, HIDDEN_DIM, N_HEADS, N_KV_HEADS, N_LAYERS, VOCAB_SIZE,
};
use sentencepiece::SentencePieceProcessor;

struct ModelFile {
    file: BufReader<File>,
}

impl ModelFile {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            file: BufReader::new(File::open(path)?),
        })
    }
    pub fn get_weights(&mut self, query: &str) -> anyhow::Result<Vec<f32>> {
        self.file.seek(SeekFrom::Start(0))?; // reset the cursor

        let buf = &mut [0u8; 4096];

        loop {
            match rmp::decode::read_array_len(&mut self.file) {
                Ok(2) => {} // Ok
                Ok(other) => bail!("Invalid format, expected tuple2, found tuple{}", other),
                Err(e) => match e {
                    rmp::decode::ValueReadError::InvalidMarkerRead(_) => bail!("Param not found"), // EOF
                    e => bail!(e),
                },
            }

            let param_name = rmp::decode::read_str(&mut self.file, &mut buf[..])
                .map_err(|_| anyhow!("Error reading param name"))?;

            let array_len = rmp::decode::read_bin_len(&mut self.file)? as usize;

            if param_name != query {
                // skip
                self.file.seek(SeekFrom::Current(array_len as i64))?;
                continue;
            }

            let floats_len = array_len / 4; // because f32 is 4 bytes

            let weights = if is_little_endian() {
                // the fast and cool way which only works if the machine is little endian
                let mut weights = vec![f32::NAN; floats_len];
                self.file.read_exact(unsafe {
                    std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, array_len)
                })?;

                weights
            } else {
                // the bitch nerd way
                let mut weights = Vec::with_capacity(floats_len);
                for _ in 0..floats_len {
                    let mut buffer = [0u8; 4];
                    self.file.read_exact(&mut buffer[..])?;

                    weights.push(f32::from_le_bytes(buffer));
                }

                weights
            };

            return Ok(weights);
        }
    }
}

fn main() -> anyhow::Result<()> {
    type B = NdArray;
    // type B = Wgpu;

    let tokenizer = SentencePieceProcessor::open(
        "/home/mykolas/Documents/Rust/mistral-burn/models/tokenizer.model",
    )?;

    let model = load_model::<B>()?;

    let bos = tokenizer.bos_id().expect("no BOS id") as i32;

    let prompt = "test";
    let prompt_encoded: Vec<_> = iter::once(bos)
        .chain(tokenizer.encode(prompt)?.into_iter().map(|e| e.id as i32))
        .collect();
    let prompt_size = prompt_encoded.len();
    let prompt_tensor: Tensor<B, 1, Int> =
        Tensor::from_ints(&prompt_encoded[..], &Default::default());
    // TESTING SITE
    let mut h = model
        .embeddings
        .forward(prompt_tensor.unsqueeze_dim(0))
        .squeeze::<2>(0);

    let mut to_concat = Vec::new();
    for seqlen in [2] {
        to_concat.push(model.freqs_cis.clone().slice([0..seqlen]));
    }
    let freqs_cis = Tensor::cat(to_concat, 0);

    h = model.layers[0]
        .attention
        .forward(model.layers[0].attention_norm.forward(h), freqs_cis);

    println!("after attention: {:?}", h);

    todo!();

    let logits = model.forward(prompt_tensor, vec![prompt_size]);

    let logits = softmax(logits.slice([0..1]).squeeze::<1>(0), 0);

    let mut probabilities: Vec<_> = logits.into_data().value.into_iter().enumerate().collect();
    probabilities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for i in 0..5 {
        let token_id = probabilities[i].0 as u32;
        println!(
            "{}: {}",
            i + 1,
            tokenizer.decode_piece_ids(&[token_id]).unwrap()
        );
    }

    Ok(())
}

fn load_model<B: Backend>() -> anyhow::Result<Transformer<B>> {
    let mut model_file =
        ModelFile::open("/home/mykolas/Documents/Rust/mistral-burn/models/mistral-f32.weights")?;

    let embeddings = EmbeddingConfig::new(VOCAB_SIZE, DIM).init_with(EmbeddingRecord {
        weight: Tensor::<B, 1>::from_floats(
            &(model_file.get_weights("tok_embeddings.weight")?)[..],
            &Default::default(),
        )
        .reshape([VOCAB_SIZE, DIM])
        .into(),
    });

    let norm = RMSNormalization {
        weights: Tensor::<B, 1>::from_floats(
            &(model_file.get_weights("norm.weight")?)[..],
            &Default::default(),
        )
        .into(),
    };

    let output = Linear {
        weight: Tensor::<B, 1>::from_floats(
            &(model_file.get_weights("output.weight")?)[..],
            &Default::default(),
        )
        .reshape([VOCAB_SIZE, DIM])
        .transpose()
        .into(),
        bias: None,
    };

    let freqs_cis = precompute_freqs_cis::<B>(HEAD_DIM, 128_000);

    let mut layers = Vec::new();

    for i in 0..1 {
        //N_LAYERS {
        let feed_forward = FeedForward {
            w1: Linear {
                weight: Tensor::<B, 1>::from_floats(
                    &(model_file.get_weights(&format!("layers.{i}.feed_forward.w1.weight"))?)[..],
                    &Default::default(),
                )
                .reshape([HIDDEN_DIM, DIM])
                .transpose()
                .into(),
                bias: None,
            },
            w2: Linear {
                weight: Tensor::<B, 1>::from_floats(
                    &(model_file.get_weights(&format!("layers.{i}.feed_forward.w2.weight"))?)[..],
                    &Default::default(),
                )
                .reshape([DIM, HIDDEN_DIM])
                .transpose()
                .into(),
                bias: None,
            },
            w3: Linear {
                weight: Tensor::<B, 1>::from_floats(
                    &(model_file.get_weights(&format!("layers.{i}.feed_forward.w3.weight"))?)[..],
                    &Default::default(),
                )
                .reshape([HIDDEN_DIM, DIM])
                .transpose()
                .into(),
                bias: None,
            },
        };

        let attention_norm = RMSNormalization {
            weights: Tensor::<B, 1>::from_floats(
                &(model_file.get_weights(&format!("layers.{i}.attention_norm.weight"))?)[..],
                &Default::default(),
            )
            .into(),
        };

        let ffn_norm = RMSNormalization {
            weights: Tensor::<B, 1>::from_floats(
                &(model_file.get_weights(&format!("layers.{i}.ffn_norm.weight"))?)[..],
                &Default::default(),
            )
            .into(),
        };

        let attention = Attention {
            wq: Linear {
                weight: Tensor::<B, 1>::from_floats(
                    &(model_file.get_weights(&format!("layers.{i}.attention.wq.weight"))?)[..],
                    &Default::default(),
                )
                .reshape([N_HEADS * HEAD_DIM, DIM])
                .transpose()
                .into(),
                bias: None,
            },
            wk: Linear {
                weight: Tensor::<B, 1>::from_floats(
                    &(model_file.get_weights(&format!("layers.{i}.attention.wk.weight"))?)[..],
                    &Default::default(),
                )
                .reshape([N_KV_HEADS * HEAD_DIM, DIM])
                .transpose()
                .into(),
                bias: None,
            },
            wv: Linear {
                weight: Tensor::<B, 1>::from_floats(
                    &(model_file.get_weights(&format!("layers.{i}.attention.wv.weight"))?)[..],
                    &Default::default(),
                )
                .reshape([N_KV_HEADS * HEAD_DIM, DIM])
                .transpose()
                .into(),
                bias: None,
            },
            wo: Linear {
                weight: Tensor::<B, 1>::from_floats(
                    &(model_file.get_weights(&format!("layers.{i}.attention.wo.weight"))?)[..],
                    &Default::default(),
                )
                .reshape([DIM, N_HEADS * HEAD_DIM])
                .transpose()
                .into(),
                bias: None,
            },
        };

        let layer = TransformerBlock {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        };

        layers.push(layer);
    }

    let model = Transformer {
        embeddings,
        layers,
        norm,
        output,
        freqs_cis,
    };

    Ok(model)
}

fn is_little_endian() -> bool {
    const TEST: u16 = 0xFF00;
    let bytes = TEST.to_ne_bytes();

    u16::from_le_bytes(bytes) == TEST
}
