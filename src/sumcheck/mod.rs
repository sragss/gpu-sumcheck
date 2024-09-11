use ark_bn254::Fr;
use rayon::prelude::*;

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod plain;
pub mod simd;

#[derive(Clone, Debug, PartialEq, Eq)]
struct CubicSumcheckProof {
    round_polys: Vec<(Fr, Fr, Fr, Fr)>,
    rs: Vec<Fr>,
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

        let l0 = (*r - Fr::from(1)) * (*r - Fr::from(2)) * (*r - Fr::from(3))
            / ((Fr::from(0) - Fr::from(1))
                * (Fr::from(0) - Fr::from(2))
                * (Fr::from(0) - Fr::from(3)));
        let l1 = (*r - Fr::from(0)) * (*r - Fr::from(2)) * (*r - Fr::from(3))
            / ((Fr::from(1) - Fr::from(0))
                * (Fr::from(1) - Fr::from(2))
                * (Fr::from(1) - Fr::from(3)));
        let l2 = (*r - Fr::from(0)) * (*r - Fr::from(1)) * (*r - Fr::from(3))
            / ((Fr::from(2) - Fr::from(0))
                * (Fr::from(2) - Fr::from(1))
                * (Fr::from(2) - Fr::from(3)));
        let l3 = (*r - Fr::from(0)) * (*r - Fr::from(1)) * (*r - Fr::from(2))
            / ((Fr::from(3) - Fr::from(0))
                * (Fr::from(3) - Fr::from(1))
                * (Fr::from(3) - Fr::from(2)));

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

    #[tracing::instrument(skip_all)]
    fn sumcheck_top(&mut self, num_rounds: usize) -> CubicSumcheckProof {
        let mut round_polys = Vec::with_capacity(num_rounds);
        let mut rs = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let start_time = std::time::Instant::now();

            let evals = self.eval_cubic_top();

            round_polys.push(evals);
            let r = CubicSumcheckProof::fiat_shamir(evals);
            rs.push(r);

            self.bind_top(&r);

            let duration = start_time.elapsed();
            println!("Round {}: {:?}", round, duration);
        }

        CubicSumcheckProof { round_polys, rs }
    }
}

pub mod bench {
    use bench::simd::SIMDSumcheck;

    use super::*;
    #[cfg(feature = "gpu")]
    use crate::sumcheck::gpu::GPUSumcheck;
    use crate::sumcheck::plain::PlainSumcheck;
    use std::time::Instant;

    pub fn main() {
        let log_size = 28;
        let size = 1 << log_size;
        let mut evals = Vec::with_capacity(size);

        for i in 0..size {
            evals.push(Fr::from(i as u64));
        }

        let claim: Fr = evals.par_iter().map(|eval| eval * eval * eval).sum();

        let mut plain = PlainSumcheck::new(evals.clone(), evals.clone(), evals.clone());
        let start_plain = Instant::now();
        let plain_proof = plain.sumcheck_top(log_size);
        let duration_plain = start_plain.elapsed();
        println!("PlainSumcheck: {:?}\n\n", duration_plain);

        // SIMD
        let mut simd = SIMDSumcheck::new(evals.clone(), evals.clone(), evals.clone());
        let start_simd = Instant::now();
        tracing_texray::examine(tracing::info_span!("simd_sumcheck")).in_scope(|| {
            let simd_proof = simd.sumcheck_top(log_size);

            assert_eq!(plain_proof, simd_proof);
        });
        let duration_simd = start_simd.elapsed();
        println!("SIMDSumcheck: {:?}", duration_simd);

        #[cfg(feature = "gpu")]
        {
            let mut gpu = GPUSumcheck::new(evals.clone(), evals.clone(), evals.clone());
            let start_gpu = Instant::now();
            tracing_texray::examine(tracing::info_span!("gpu_sumcheck")).in_scope(|| {
                let gpu_proof = gpu.sumcheck_top(log_size);

                assert_eq!(plain_proof, gpu_proof);
            });
            let duration_gpu = start_gpu.elapsed();
            println!("GPUSumcheck: {:?}", duration_gpu);
        }

        plain_proof.verify(&claim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "gpu")]
    use crate::poly::gpu::GPUPoly;
    use crate::poly::plain::DensePolynomial;
    #[cfg(feature = "gpu")]
    use crate::sumcheck::gpu::GPUSumcheck;
    use crate::sumcheck::plain::PlainSumcheck;

    #[test]
    fn plain_sumcheck() {
        let eq = vec![Fr::from(12), Fr::from(13), Fr::from(14), Fr::from(15)];
        let a = eq.clone();
        let b = eq.clone();

        let claim: Fr = (0..eq.len()).into_iter().map(|i| eq[i] * a[i] * b[i]).sum();

        let mut plain = PlainSumcheck::new(eq, a, b);
        let proof = plain.sumcheck_top(2);
        proof.verify(&claim);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_sumcheck() {
        let eq = vec![Fr::from(12), Fr::from(13), Fr::from(14), Fr::from(15)];
        let a = eq.clone();
        let b = eq.clone();

        let claim: Fr = (0..eq.len()).into_iter().map(|i| eq[i] * a[i] * b[i]).sum();

        let mut sumcheck = GPUSumcheck::new(eq, a, b);
        let proof = sumcheck.sumcheck_top(2);
        proof.verify(&claim);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_bind_bot() {
        let evals = vec![
            Fr::from(11),
            Fr::from(12),
            Fr::from(13),
            Fr::from(14),
            Fr::from(15),
            Fr::from(16),
            Fr::from(17),
            Fr::from(18),
        ];
        let mut poly = DensePolynomial::new(evals.clone());
        let mut gpu_poly = GPUPoly::new(evals);

        let r = Fr::from(20);

        poly.bound_poly_var_bot(&r);
        gpu_poly.bound_poly_var_bot(&r);
        assert_eq!(poly, gpu_poly.to_ark());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_bind_top() {
        let evals = vec![
            Fr::from(11),
            Fr::from(12),
            Fr::from(13),
            Fr::from(14),
            Fr::from(15),
            Fr::from(16),
            Fr::from(17),
            Fr::from(18),
        ];
        let mut poly = DensePolynomial::new(evals.clone());
        let mut gpu_poly = GPUPoly::new(evals);

        let r = Fr::from(20);

        poly.bound_poly_var_top(&r);
        gpu_poly.bound_poly_var_top(&r);
        assert_eq!(poly, gpu_poly.to_ark());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn eval_cubic_top_parity() {
        let evals = vec![
            Fr::from(11),
            Fr::from(12),
            Fr::from(13),
            Fr::from(14),
            Fr::from(15),
            Fr::from(16),
            Fr::from(17),
            Fr::from(18),
        ];
        let mut plain_sumcheck = PlainSumcheck::new(evals.clone(), evals.clone(), evals.clone());
        let mut gpu_sumcheck = GPUSumcheck::new(evals.clone(), evals.clone(), evals.clone());

        let plain_res = plain_sumcheck.eval_cubic_top();
        let gpu_res = gpu_sumcheck.eval_cubic_top();

        assert_eq!(plain_res, gpu_res);
    }
}
