use crate::poly::plain::DensePolynomial;
use ark_bn254::Fr;
use icicle_bn254::curve::ScalarField as GPUScalar;
use icicle_bn254::polynomials::DensePolynomial as IngoPoly;
use icicle_core::vec_ops::{bind_scalars, VecOpsConfig};
use icicle_core::{ntt::FieldImpl, vec_ops::sub_scalars};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::ArkConvertible;
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use rayon::prelude::*;

pub struct GPUPoly {
    pub poly: IngoPoly,
    pub len: usize,
}

impl GPUPoly {
    pub fn new(z: Vec<Fr>) -> Self {
        IngoPoly::init_cuda_backend();
        // TODO(sragss): Can do this on GPU.
        let z: Vec<_> = z
            .into_par_iter()
            .map(|item| GPUScalar::from_ark(item))
            .collect();
        let slice = HostSlice::from_slice(&z);
        let poly = IngoPoly::from_coeffs(slice, z.len());
        Self { poly, len: z.len() }
    }

    pub fn bound_poly_var_bot(&mut self, r: &Fr) {
        self.len /= 2;

        let even = self.poly.even();
        let odd = self.poly.odd();
        let mut m_poly = &odd - &even;
        let r_gpu = GPUScalar::from_ark(r.to_owned());
        m_poly = &r_gpu * &m_poly;
        let result = &even + &m_poly;
        self.poly = result;
    }

    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_top(&mut self, r: &Fr) {
        self.len /= 2;


        let coeffs = self.poly.coeffs_mut_slice();

        bind_scalars(coeffs, GPUScalar::from_ark(r.to_owned()), self.len).unwrap();
    }

    pub fn to_ark(self) -> DensePolynomial<Fr> {
        let mut host_vals: Vec<GPUScalar> = vec![GPUScalar::zero(); self.len];
        let host_slice = HostSlice::from_mut_slice(&mut host_vals);
        self.poly.copy_coeffs(0, host_slice);
        DensePolynomial::new(host_vals.into_iter().map(|item| item.to_ark()).collect())
    }
}
