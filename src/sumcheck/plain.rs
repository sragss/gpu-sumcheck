use crate::poly::plain::DensePolynomial;
use ark_bn254::Fr;
use crate::sumcheck::CubicSumcheck;
use rayon::prelude::*;
use ark_std::Zero;


pub struct PlainSumcheck {
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

            // println!("index {i}: eval_0: {eval_0:?}");

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