use crate::poly::gpu::GPUPoly;
use crate::sumcheck::CubicSumcheck;
use ark_bn254::Fr;
use icicle_bn254::polynomials::DensePolynomial as IngoPoly;
use icicle_bn254::curve::ScalarField as GPUScalar;
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::ArkConvertible;
use icicle_core::vec_ops::{bind_triple_scalars, eval_cubic_scalars, mul_scalars, sub_scalars, sum_scalars};
use icicle_core::vec_ops::VecOpsConfig;
use icicle_cuda_runtime::memory::DeviceVec;
use icicle_core::ntt::FieldImpl;

pub struct GPUSumcheck {
    eq: GPUPoly,
    a: GPUPoly,
    b: GPUPoly,
}

// #[tracing::instrument(skip_all)]
fn split(poly: &IngoPoly, len: usize) -> (IngoPoly, IngoPoly) {
    let n = len / 2;
    let low = poly.slice(0, 1, n as u64);
    let high = poly.slice(n as u64, 1, n as u64);
    (low, high)
}

// #[tracing::instrument(skip_all)]
fn sum_poly(poly: &IngoPoly, len: usize) -> Fr {
    if len == 1 {
        let gpu_scalar = poly.get_coeff(0);
        gpu_scalar.to_ark()
    } else {
        let (low, high) = split(poly, len);
        let len = len / 2;
        let half = &low + &high;
        return sum_poly(&half, len);
    }
}

impl CubicSumcheck for GPUSumcheck {
    fn new(eq: Vec<Fr>, a: Vec<Fr>, b: Vec<Fr>) -> Self {
        let eq = GPUPoly::new(eq);
        let a = GPUPoly::new(a);
        let b = GPUPoly::new(b);

        Self { eq, a, b }
    }


    #[tracing::instrument(skip_all)]
    fn eval_cubic_top(&mut self) -> (Fr, Fr, Fr, Fr) {
        assert_eq!(self.eq.len, self.a.len);
        assert_eq!(self.a.len, self.b.len);
        let n = self.eq.len / 2;

        let mut result = vec![GPUScalar::zero(); 4];
        
        let eq_coeffs = self.eq.poly.coeffs_mut_slice();
        let a_coeffs = self.a.poly.coeffs_mut_slice();
        let b_coeffs = self.b.poly.coeffs_mut_slice();
        
        eval_cubic_scalars(eq_coeffs, a_coeffs, b_coeffs, n, &mut result).unwrap();

        (result[0].to_ark(), result[1].to_ark(), result[2].to_ark(), result[3].to_ark())
    }

    #[tracing::instrument(skip_all)]
    fn bind_top(&mut self, r: &Fr) {
        // self.eq.bound_poly_var_top(&r);
        // self.a.bound_poly_var_top(&r);
        // self.b.bound_poly_var_top(&r);

        // Note(sragss): Below is identical speeds.
        let len = self.eq.len / 2;
        self.eq.len = len;
        self.a.len = len;
        self.b.len = len;

        bind_triple_scalars(
            &mut self.eq.poly.coeffs_mut_slice()[..], 
            &mut self.a.poly.coeffs_mut_slice()[..], 
            &mut self.b.poly.coeffs_mut_slice()[..], 
            GPUScalar::from_ark(r.to_owned()), 
            len
        ).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn sum() {
        let evals = vec![Fr::from(12), Fr::from(13), Fr::from(14), Fr::from(15)];
        let poly = GPUPoly::new(evals);
        let result = sum_poly(&poly.poly, 4);

        assert_eq!(result, Fr::from(54));
    }
}
