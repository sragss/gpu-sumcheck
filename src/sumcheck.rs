use icicle_bn254::polynomials::DensePolynomial as IngoPoly;
use icicle_bn254::curve::ScalarField as GPUScalar;
use ark_bn254::Fr;
use icicle_core::{traits::ArkConvertible, vec_ops::{add_scalars, sub_scalars, VecOpsConfig}};
use icicle_cuda_runtime::memory::{DeviceSlice, DeviceVec, HostSlice};
use rayon::prelude::*;
use ark_std::{Zero, One};
use icicle_core::polynomials::UnivariatePolynomial;
use icicle_core::ntt::FieldImpl;

use crate::DensePolynomial;

struct GPUPoly {
    poly: IngoPoly,
    len: usize,
}

impl GPUPoly {
    fn new(z: Vec<Fr>) -> Self {
        IngoPoly::init_cuda_backend();
        let z: Vec<_> = z.into_iter().map(|item| GPUScalar::from_ark(item)).collect();
        let slice = HostSlice::from_slice(&z);
        let poly = IngoPoly::from_coeffs(slice, z.len());
        Self { poly, len: z.len() }
    }

    fn bound_poly_var_bot(&mut self, r: &Fr) {
        self.len /= 2;

        let even = self.poly.even();
        let odd = self.poly.odd();
        let mut m_poly = &odd - &even;
        let r_gpu = GPUScalar::from_ark(r.to_owned());
        m_poly = &r_gpu * &m_poly;
        let result = &even + &m_poly;
        self.poly = result;
    }

    fn bound_poly_var_top(&mut self, r: &Fr) {
        self.len /= 2;

        let low = self.poly.slice(0, 1, self.len as u64);
        let high = self.poly.slice(self.len as u64, 1, self.len as u64);
        let mut m_poly = &high- &low;
        let r_gpu = GPUScalar::from_ark(r.to_owned());
        m_poly = &r_gpu * &m_poly;
        let result = &low + &m_poly;
        self.poly = result;
    }

    fn to_ark(self) -> DensePolynomial<Fr> {
        let mut host_vals: Vec<GPUScalar> = vec![GPUScalar::zero(); self.len];
        let host_slice = HostSlice::from_mut_slice(&mut host_vals);
        self.poly.copy_coeffs(0, host_slice);
        DensePolynomial::new(host_vals.into_iter().map(|item| item.to_ark()).collect())
    }
}




struct CubicSumcheckProof {
    round_polys: Vec<(Fr, Fr, Fr, Fr)>,
    rs: Vec<Fr>
}

impl CubicSumcheckProof {
    fn fiat_shamir(round_poly: (Fr, Fr, Fr, Fr)) -> Fr {
        (round_poly.0 * round_poly.1 * round_poly.2 * round_poly.3 + Fr::from(13)) * Fr::from(29)
    }

    /// Evaluates the univariate polynomial as specified by its evaluations over [0, ... 3]
    /// at a new point 'r' using Lagrange interpolation.
    /// 
    /// evals: f(0), f(1), f(2), f(3)
    /// r: f(r)
    fn eval_uni(evals: (Fr, Fr, Fr, Fr), r: &Fr) -> Fr {
        let (f0, f1, f2, f3) = evals;

        let l0 = (*r - Fr::from(1)) * (*r - Fr::from(2)) * (*r - Fr::from(3)) / ((Fr::from(0) - Fr::from(1)) * (Fr::from(0) - Fr::from(2)) * (Fr::from(0) - Fr::from(3)));
        let l1 = (*r - Fr::from(0)) * (*r - Fr::from(2)) * (*r - Fr::from(3)) / ((Fr::from(1) - Fr::from(0)) * (Fr::from(1) - Fr::from(2)) * (Fr::from(1) - Fr::from(3)));
        let l2 = (*r - Fr::from(0)) * (*r - Fr::from(1)) * (*r - Fr::from(3)) / ((Fr::from(2) - Fr::from(0)) * (Fr::from(2) - Fr::from(1)) * (Fr::from(2) - Fr::from(3)));
        let l3 = (*r - Fr::from(0)) * (*r - Fr::from(1)) * (*r - Fr::from(2)) / ((Fr::from(3) - Fr::from(0)) * (Fr::from(3) - Fr::from(1)) * (Fr::from(3) - Fr::from(2)));

        f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3
    }

    /// returns a claim
    fn verify(&self, claim: &Fr) -> Fr {
        let num_rounds = self.round_polys.len();
        assert_eq!(self.rs.len(), num_rounds);

        let mut prev_claim: Fr = claim.to_owned();

        let mut v_rs = Vec::with_capacity(num_rounds);

        for i in 0..num_rounds {
            let round_poly = self.round_polys[i];
            assert_eq!(round_poly.0 + round_poly.1, prev_claim, "Round {i}");
            let r = Self::fiat_shamir(round_poly);
            println!("v_r {r:?}");
            v_rs.push(r);
            prev_claim = Self::eval_uni(round_poly, &r);
        }

        assert_eq!(v_rs, self.rs);

        prev_claim
    }
}


trait CubicSumcheck {
    fn new(eq: Vec<Fr>, a: Vec<Fr>, b: Vec<Fr>) -> Self;
    fn eval_cubic_top(&mut self) -> (Fr, Fr, Fr, Fr);
    fn bind_top(&mut self, r: &Fr);


    fn sumcheck_top(&mut self, num_rounds: usize) -> CubicSumcheckProof {
        let mut round_polys = Vec::with_capacity(num_rounds);
        let mut rs = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds {
            let evals = self.eval_cubic_top();
            round_polys.push(evals);
            println!("round evals: {evals:?}");
            let r = CubicSumcheckProof::fiat_shamir(evals);
            rs.push(r);
            println!("p_r {r:?}");
            self.bind_top(&r);
        }

        CubicSumcheckProof {
            round_polys,
            rs
        }
    }
}

struct PlainSumcheck {
    eq: DensePolynomial<Fr>,
    a: DensePolynomial<Fr>,
    b: DensePolynomial<Fr>
}

impl CubicSumcheck for PlainSumcheck {
    fn new(eq: Vec<Fr>, a: Vec<Fr>, b: Vec<Fr>) -> Self {
        let eq = DensePolynomial::new(eq);
        let a = DensePolynomial::new(a);
        let b = DensePolynomial::new(b);

        Self { eq, a, b }
    }

    fn eval_cubic_top(&mut self) -> (Fr, Fr, Fr, Fr) {
        let len = self.eq.Z.len();
        assert_eq!(self.a.Z.len(), len);
        assert_eq!(self.b.Z.len(), len);
        let n = len / 2;

        // low + r * (high - low)
        let (eval_0, eval_1, eval_2, eval_3) = (0..n).into_par_iter().map(|i| {
            let low = 2*i;
            let high = 2*i + 1;

            let eval_0: Fr = self.eq[low] * self.a[low] * self.b[low];
            let eval_1: Fr = self.eq[high] * self.a[high] * self.b[high];
            
            let eq_m: Fr = self.eq[high] - self.eq[low];
            let a_m: Fr = self.a[high] - self.a[low];
            let b_m: Fr = self.b[high] - self.b[low];

            let eq_2 = self.eq[high] + eq_m;
            let a_2 = self.a[high] + a_m;
            let b_2 = self.b[high] + b_m;
            let eval_2 = eq_2 * a_2 * b_2;
            
            let eq_3 = eq_2 + eq_m;
            let a_3 = a_2 + a_m;
            let b_3 = b_2 + b_m;
            let eval_3 = eq_3 * a_3 * b_3;

            println!("index {i}: eval_0: {eval_0:?}");

            (eval_0, eval_1, eval_2, eval_3)
        }).reduce(|| (Fr::zero(), Fr::zero(), Fr::zero(), Fr::zero()), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));

        (eval_0, eval_1, eval_2, eval_3)
    }

    fn bind_top(&mut self, r: &Fr) {
        self.eq.bound_poly_var_bot(r);
        self.a.bound_poly_var_bot(r);
        self.b.bound_poly_var_bot(r);
    }
}

struct GPUSumcheck {
    eq: GPUPoly,
    a: GPUPoly,
    b: GPUPoly
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
        IngoPoly::from_coeffs(self.a.poly.coeffs_mut_slice(), 1 << 12);

        unimplemented!();
    }

    fn bind_top(&mut self, r: &Fr) {
        unimplemented!();
    }

}


pub mod bench {
    use super::*;

    pub fn main() {
        let size = 1 << 24;
        let mut evals = Vec::with_capacity(size);

        for i in 0..size {
            evals.push(Fr::from(i as u64));
        }

        use std::time::Instant;

        let mut plain = DensePolynomial::new(evals.clone());
        let r = Fr::from(20);

        let start_plain = Instant::now();
        plain.bound_poly_var_top_par(&r);
        let duration_plain = start_plain.elapsed();

        let mut gpu = GPUPoly::new(evals);
        
        let start_gpu = Instant::now();
        gpu.bound_poly_var_top(&r);
        let duration_gpu = start_gpu.elapsed();

        println!("Time taken for plain bound_poly_var_top: {:?}", duration_plain);
        println!("Time taken for GPU bound_poly_var_top: {:?}", duration_gpu);

    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain() {
        let eq = vec![Fr::from(12), Fr::from(13), Fr::from(14), Fr::from(15)];
        let a = eq.clone();
        let b = eq.clone();

        let claim: Fr = (0..eq.len()).into_iter().map(|i| eq[i] * a[i] * b[i]).sum();

        let eq = DensePolynomial::new(eq);
        let a = DensePolynomial::new(a);
        let b = DensePolynomial::new(b);

        let mut plain = PlainSumcheck {eq, a, b};
        let proof = plain.sumcheck_top(2);
        proof.verify(&claim);
    }

    #[test]
    fn gpu_bind_bot() {
        let evals = vec![Fr::from(11), Fr::from(12), Fr::from(13), Fr::from(14), Fr::from(15), Fr::from(16), Fr::from(17), Fr::from(18)];
        let mut poly = DensePolynomial::new(evals.clone());
        let mut gpu_poly = GPUPoly::new(evals);

        let r = Fr::from(20);

        poly.bound_poly_var_bot(&r);
        gpu_poly.bound_poly_var_bot(&r);
        assert_eq!(poly, gpu_poly.to_ark());
    }

    #[test]
    fn gpu_bind_top() {
        let evals = vec![Fr::from(11), Fr::from(12), Fr::from(13), Fr::from(14), Fr::from(15), Fr::from(16), Fr::from(17), Fr::from(18)];
        let mut poly = DensePolynomial::new(evals.clone());
        let mut gpu_poly = GPUPoly::new(evals);

        let r = Fr::from(20);

        poly.bound_poly_var_top(&r);
        gpu_poly.bound_poly_var_top(&r);
        assert_eq!(poly, gpu_poly.to_ark());
    }
}