use ark_ff::PrimeField;
use ark_std::test_rng;
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DensePolynomial<F: PrimeField> {
    pub Z: Vec<F>
}

impl<F: PrimeField> DensePolynomial<F> {
    pub fn new(z: Vec<F>) -> Self {
        assert!(
            is_power_of_two(z.len()),
            "Dense multi-linear polynomials must be made from a power of 2 (not {})",
            z.len()
        );

        DensePolynomial {
            Z: z
        }
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.Z.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.Z.truncate(n);
    }

    pub fn bound_poly_var_top_par(&mut self, r: &F) {
        let n = self.Z.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.Z.truncate(n);
    }

    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.Z.len() / 2;
        for i in 0..n {
            self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
        }
        self.Z.truncate(n);
    }

}

impl<F: PrimeField> std::ops::Index<usize> for DensePolynomial<F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, _index: usize) -> &F {
        &(self.Z[_index])
    }
}

pub fn rand_vec<F: PrimeField>(n: usize) -> Vec<F> {
    assert!(is_power_of_two(n));

    let mut result: Vec<F> = Vec::with_capacity(n);
    let mut rng = test_rng();
    for _ in 0..n {
        result.push(F::rand(&mut rng));
    }

    result
}

pub fn rand_fr<F: PrimeField>() -> F {
    let mut rng = test_rng();
    F::rand(&mut rng)
}

/// Checks if `num` is a power of 2.
pub fn is_power_of_two(num: usize) -> bool {
    num != 0 && (num & (num - 1)) == 0
}