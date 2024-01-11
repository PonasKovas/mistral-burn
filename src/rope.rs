use burn::tensor::{backend::Backend, Tensor};

// Workaround for torch.outer
fn outer_product<B: Backend>(v1: Tensor<B, 1>, v2: Tensor<B, 1>) -> Tensor<B, 2> {
    let v1: Tensor<B, 2> = v1.unsqueeze().transpose();
    let v2: Tensor<B, 2> = v2.unsqueeze();

    v1.matmul(v2)
}

pub fn precompute_freqs_cis<B: Backend>(dim: usize, end: usize) -> Tensor<B, 3> {
    const THETA: f64 = 10000.0;

    let freqs: Tensor<B, 1> =
        Tensor::arange_step(0..(dim - 1), 2, &Default::default()).float() / (dim as f64);
    let freqs = (freqs * THETA.ln() * -1.).exp(); // workaround of THETA to the power of the freqs

    let t = Tensor::arange(0..end, &Default::default()).float();
    let freqs = outer_product(t, freqs);

    // torch.polar workaround by increasing the rank and having one dimension for cosines and one for sines
    let cosines = freqs.clone().cos();
    let sines = freqs.sin();
    let result: Tensor<B, 3> = Tensor::stack::<3>(vec![cosines, sines], 2);

    result
}

pub fn apply_rotary_emb<B: Backend>(tensor: Tensor<B, 3>, freqs_cis: Tensor<B, 3>) -> Tensor<B, 3> {
    let og_shape = tensor.dims();
    let tensor = tensor.reshape([og_shape[0], og_shape[1], og_shape[2] / 2, 2]);

    // emulate complex number multiplication
    // (a+bi) * (c+di) = (ac - bd) + (ad + bc)i
    let t_dims = tensor.dims();
    let t_real: Tensor<B, 3> = tensor
        .clone()
        .slice([0..t_dims[0], 0..t_dims[1], 0..t_dims[2], 0..1])
        .squeeze(3);
    let t_imaginary: Tensor<B, 3> = tensor
        .slice([0..t_dims[0], 0..t_dims[1], 0..t_dims[2], 1..2])
        .squeeze(3);

    let freqs_cis: Tensor<B, 4> = freqs_cis.unsqueeze_dim(1);
    let f_dims = freqs_cis.dims();
    let f_real: Tensor<B, 3> = freqs_cis
        .clone()
        .slice([0..f_dims[0], 0..f_dims[1], 0..f_dims[2], 0..1])
        .squeeze(3);
    let f_imaginary: Tensor<B, 3> = freqs_cis
        .slice([0..f_dims[0], 0..f_dims[1], 0..f_dims[2], 1..2])
        .squeeze(3);

    let result_real = t_real.clone() * f_real.clone() - t_imaginary.clone() * f_imaginary.clone();
    let result_imaginary = t_real * f_imaginary + t_imaginary * f_real;

    let result = Tensor::stack::<4>(vec![result_real, result_imaginary], 3);

    result.flatten(2, 3)
}
