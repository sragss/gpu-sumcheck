use poly_bind_bench::gpu::GPUPoly;
use poly_bind_bench::{rand_fr, rand_vec, DensePolynomial};
    use std::time::Instant;


fn main() {
    poly_bind_bench::sumcheck::bench::main();
    // let size: usize = 1 << 27;
    // let rand = rand_vec::<ark_bn254::Fr>(size);
    // let mut poly_single = DensePolynomial::new(rand.clone());
    // let mut poly_thread = poly_single.clone();

    // let bind_point: ark_bn254::Fr = rand_fr();


    // let start_single = Instant::now();
    // poly_single.bound_poly_var_top(&bind_point);
    // let duration_single = start_single.elapsed();
    // println!("Time taken for single-threaded: {:?}", duration_single);

    // let start_thread = Instant::now();
    // poly_thread.bound_poly_var_top_par(&bind_point);
    // let duration_thread = start_thread.elapsed();
    // println!("Time taken for multi-threaded: {:?}", duration_thread);

    // icicle();

    // let mut gpu_poly = GPUPoly::new_ark(rand);
    // let mut gpu_poly_2 = gpu_poly.clone();
    // drop(gpu_poly);
    // let start_gpu = Instant::now();
    // gpu_poly.bind_poly_var_top(ScalarCfg::generate_random(1)[0]);
    // let duration_gpu = start_gpu.elapsed();
    // println!("Time taken for GPU: {:?}", duration_gpu);

    // println!("GPU 2");
    // let start_gpu_2 = Instant::now();
    // gpu_poly_2.bind_poly_var_top(ScalarCfg::generate_random(1)[0]);
    // let duration_gpu_2 = start_gpu_2.elapsed();
    // println!("Time taken for GPU 2: {:?}", duration_gpu_2);
}

// use icicle::HostOrDeviceSlice;
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::vec_ops::{add_scalars, VecOpsConfig};
use icicle_core::traits::GenerateRandom;
use icicle_core::ntt::FieldImpl;
fn icicle() {

    let test_size = 1 << 22;

    HostSlice::from_slice(&vec![ScalarField::zero(); test_size]);
    let l = ScalarCfg::generate_random(test_size);
    let r = ScalarCfg::generate_random(test_size);
    let a  = HostSlice::from_slice(&l);
    let b  = HostSlice::from_slice(&r);

    let mut res =vec![ScalarField::zero(); test_size];
    let result = HostSlice::from_mut_slice(&mut res);

    let cfg = VecOpsConfig::default();
    let start_add_scalars = Instant::now();
    add_scalars(a, b, result, &cfg).unwrap();
    let duration_add_scalars = start_add_scalars.elapsed();
    println!("Time taken for add_scalars: {:?}", duration_add_scalars);
}


