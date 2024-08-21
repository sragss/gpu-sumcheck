use crate::poly::gpu::GPUPoly;
use crate::sumcheck::CubicSumcheck;
use ark_bn254::Fr;
use icicle_bn254::polynomials::DensePolynomial as IngoPoly;
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::traits::ArkConvertible;
use icicle_core::vec_ops::mul_scalars;
use icicle_core::vec_ops::VecOpsConfig;
use icicle_cuda_runtime::memory::DeviceVec;

pub struct GPUSumcheck {
    eq: GPUPoly,
    a: GPUPoly,
    b: GPUPoly,
}

fn split(poly: &IngoPoly, len: usize) -> (IngoPoly, IngoPoly) {
    let n = len / 2;
    let low = poly.slice(0, 1, n as u64);
    let high = poly.slice(n as u64, 1, n as u64);
    (low, high)
}

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

    // TODO(sragss): This is likely going to be slow as shit depending on how .even and .odd are implemented.
    // low + r * (high - low)
    fn eval_cubic_top(&mut self) -> (Fr, Fr, Fr, Fr) {
        assert_eq!(self.eq.len, self.a.len);
        assert_eq!(self.a.len, self.b.len);
        let n = self.eq.len / 2;

        let (mut eq_low, mut eq_high) = split(&self.eq.poly, self.eq.len);
        let (mut a_low, mut a_high) = split(&self.a.poly, self.a.len);
        let (mut b_low, mut b_high) = split(&self.b.poly, self.b.len);

        let cfg = VecOpsConfig::default();

        let mut buff = DeviceVec::cuda_malloc(n).unwrap();
        let mut buff_2 = DeviceVec::cuda_malloc(n).unwrap();
        let mut prod_3_sum = |a: &mut IngoPoly, b: &mut IngoPoly, c: &mut IngoPoly| -> Fr {
            mul_scalars(
                a.coeffs_mut_slice(),
                b.coeffs_mut_slice(),
                &mut buff[..],
                &cfg,
            )
            .unwrap();
            mul_scalars(&buff[..], c.coeffs_mut_slice(), &mut buff_2[..], &cfg).unwrap();
            let poly = IngoPoly::from_coeffs(&buff_2[..], n);
            sum_poly(&poly, n)
        };

        let eval_0 = prod_3_sum(&mut eq_low, &mut a_low, &mut b_low);
        let eval_1 = prod_3_sum(&mut eq_high, &mut a_high, &mut b_high);

        let eq_m = &eq_high - &eq_low;
        let a_m = &a_high - &a_low;
        let b_m = &b_high - &b_low;

        let mut eq_2 = &eq_high + &eq_m;
        let mut a_2 = &a_high + &a_m;
        let mut b_2 = &b_high + &b_m;

        let eval_2 = prod_3_sum(&mut eq_2, &mut a_2, &mut b_2);

        let mut eq_3 = &eq_2 + &eq_m;
        let mut a_3 = &a_2 + &a_m;
        let mut b_3 = &b_2 + &b_m;

        let eval_3 = prod_3_sum(&mut eq_3, &mut a_3, &mut b_3);

        (eval_0, eval_1, eval_2, eval_3)
    }

    fn bind_top(&mut self, r: &Fr) {
        self.eq.bound_poly_var_top(&r);
        self.a.bound_poly_var_top(&r);
        self.b.bound_poly_var_top(&r);
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
