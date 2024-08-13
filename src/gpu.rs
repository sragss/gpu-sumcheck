use ark_ff::PrimeField;
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_bn254::ntt;
use icicle_core::traits::ArkConvertible;
use icicle_core::{traits::GenerateRandom, vec_ops::VecOpsConfig};
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_core::vec_ops::{mul_scalars, sub_scalars};
use icicle_core::ntt::FieldImpl;
use std::hint::black_box;
use std::time::Instant;
use icicle_bn254::polynomials::DensePolynomial as PolynomialBn254;
use icicle_core::polynomials::UnivariatePolynomial;


#[derive(Clone)]
pub struct GPUPoly {
    pub z: Vec<ScalarField>,
    len: usize
}

impl GPUPoly {
    pub fn new(size: usize) -> Self {
        let z = ScalarCfg::generate_random(size);
        Self { z, len: size }
    }

    pub fn new_ark(ark: Vec<ark_bn254::Fr>) -> Self {
        let len = ark.len();
        let z = ark.into_iter().map(|item| ScalarField::from_ark(item)).collect();
        Self { z, len }
    }



    // low += r * (high - low)
    pub fn bind_poly_var_top(&mut self, r: ScalarField) {
        let new_len = self.len / 2;

        let cfg = VecOpsConfig::default();
        let (low, high) = self.z.split_at_mut(new_len);
        let low_slice = HostSlice::from_slice(low);
        let high_slice = HostSlice::from_slice(high);
        let mut diff_local = vec![ScalarField::zero(); new_len];
        let diff = HostSlice::from_mut_slice(&mut diff_local);
        // let mut diff = DeviceVec::cuda_malloc(new_len).unwrap(); // TODO(sragss): Ideally this stays on the device
        let start_sub = Instant::now();
        sub_scalars(low_slice, high_slice, diff, &cfg).unwrap();
        let duration_sub = start_sub.elapsed();
        println!("Time taken for sub_scalars: {:?}", duration_sub);

        let r = vec![r; new_len];
        let r_slice = HostSlice::from_slice(&r);
        let mut new_low = vec![ScalarField::zero(); new_len];
        let mut new_low_slice = HostSlice::from_mut_slice(&mut new_low);
        
        let start_mul = Instant::now();
        mul_scalars(r_slice, low_slice, new_low_slice, &cfg).unwrap();
        let duration_mul = start_mul.elapsed();
        println!("Time taken for mul_scalars: {:?}", duration_mul);

        // TODO(sragss): Need to copy back to self.z[..new_len]

        self.len = new_len;
    }

    // low += r * (high - low)
    pub fn bind_poly_var_top2(&mut self, r: ScalarField) {
        let new_len = self.len / 2;

        // let cfg = VecOpsConfig::default();
        let (low, high) = self.z.split_at(new_len);
        let low_slice = HostSlice::from_slice(low);
        let high_slice = HostSlice::from_slice(high);
        assert!(PolynomialBn254::init_cuda_backend());
        let low_poly = PolynomialBn254::from_coeffs(low_slice, new_len);
        let high_poly = PolynomialBn254::from_coeffs(high_slice, new_len);
        let start_diff = Instant::now();
        let diff_poly: PolynomialBn254 = &high_poly - &low_poly;
        let duration_diff = start_diff.elapsed();
        println!("Sub: {:?}", duration_diff);

        let start_mul = Instant::now();
        let mul_poly = &r * &diff_poly;
        let duration_mul = start_mul.elapsed();
        println!("Mul: {:?}", duration_mul);

        let new_low = HostSlice::from_mut_slice(&mut self.z[..new_len]);
        let start_copy = Instant::now();
        mul_poly.copy_coeffs(0, new_low);
        let duration_copy = start_copy.elapsed();
        println!("copy_coeffs: {:?}", duration_copy);
        self.z.truncate(new_len);

        self.len = new_len;
    }
}
