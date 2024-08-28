use crate::poly::plain::DensePolynomial;
use crate::sumcheck::CubicSumcheck;
use ark_bn254::Fr;
use ark_std::Zero;
use rayon::prelude::*;

pub struct SIMDSumcheck {
    eq: DensePolynomial<Fr>,
    a: DensePolynomial<Fr>,
    b: DensePolynomial<Fr>,
}

impl CubicSumcheck for SIMDSumcheck {
    fn new(eq: Vec<Fr>, a: Vec<Fr>, b: Vec<Fr>) -> Self {
        let eq = DensePolynomial::new(eq);
        let a = DensePolynomial::new(a);
        let b = DensePolynomial::new(b);

        Self { eq, a, b }
    }

    #[tracing::instrument(skip_all)]
    fn eval_cubic_top(&mut self) -> (Fr, Fr, Fr, Fr) {
        let len = self.eq.Z.len();
        assert_eq!(self.a.Z.len(), len);
        assert_eq!(self.b.Z.len(), len);
        let n = len / 2;

        use vectorized_fields::*;

        let rayon_threads = rayon::current_num_threads();
        let chunk_size = (n / rayon_threads / 16) + 1; // Non-zero + better work-stealing
        let (eq_low, eq_high) = self.eq.Z.split_at(n);
        let (a_low, a_high) = self.a.Z.split_at(n);
        let (b_low, b_high) = self.b.Z.split_at(n);

        let (eval_0, eval_1, eval_2, eval_3) = eq_low
            .par_chunks(chunk_size)
            .zip(eq_high.par_chunks(chunk_size))
            .zip(a_low.par_chunks(chunk_size))
            .zip(a_high.par_chunks(chunk_size))
            .zip(b_low.par_chunks(chunk_size))
            .zip(b_high.par_chunks(chunk_size))
            .map(|(((((eq_low, eq_high), a_low), a_high), b_low), b_high)| {
                let chunk_size = eq_low.len();
                let mut buff = unsafe_alloc_vec(chunk_size);
                mul_vec_bn254(eq_low, a_low, &mut buff);
                let eval_0 = inner_product_bn254(&buff, b_low);
                mul_vec_bn254(eq_high, a_high, &mut buff);
                let eval_1 = inner_product_bn254(&buff, b_high);

                let mut eq_m = unsafe_alloc_vec(chunk_size);
                let mut a_m = unsafe_alloc_vec(chunk_size);
                let mut b_m = unsafe_alloc_vec(chunk_size);
                sub_vec_bn254(eq_high, eq_low, &mut eq_m);
                sub_vec_bn254(a_high, a_low, &mut a_m);
                sub_vec_bn254(b_high, b_low, &mut b_m);
                
                // 2
                let mut eq_2 = unsafe_alloc_vec(chunk_size);
                let mut a_2 = unsafe_alloc_vec(chunk_size);
                let mut b_2 = unsafe_alloc_vec(chunk_size);
                add_vec_bn254(&eq_high, &eq_m, &mut eq_2);
                add_vec_bn254(&a_high, &a_m, &mut a_2);
                add_vec_bn254(&b_high, &b_m, &mut b_2);
                mul_vec_bn254(&eq_2, &a_2, &mut buff);
                let eval_2 = inner_product_bn254(&buff, &b_2);

                // 3
                let mut eq_3 = unsafe_alloc_vec(chunk_size);
                let mut a_3 = unsafe_alloc_vec(chunk_size);
                let mut b_3 = unsafe_alloc_vec(chunk_size);
                add_vec_bn254(&eq_2, &eq_m, &mut eq_3);
                add_vec_bn254(&a_2, &a_m, &mut a_3);
                add_vec_bn254(&b_2, &b_m, &mut b_3);
                mul_vec_bn254(&eq_3, &a_3, &mut buff);
                let eval_3 = inner_product_bn254(&buff, &b_3);

                (eval_0, eval_1, eval_2, eval_3)
            })
            .reduce(
                || (Fr::zero(), Fr::zero(), Fr::zero(), Fr::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
            );

        (eval_0, eval_1, eval_2, eval_3)
    }

    #[tracing::instrument(skip_all)]
    fn bind_top(&mut self, r: &Fr) {
        self.eq.bound_poly_var_top(r);
        self.a.bound_poly_var_top(r);
        self.b.bound_poly_var_top(r);
    }
}

pub fn unsafe_alloc_vec(size: usize) -> Vec<Fr> {
    let mut vec = Vec::with_capacity(size);
    unsafe {
        vec.set_len(size);
    }
    vec
}

#[tracing::instrument(skip_all)]
pub fn unsafe_allocate_zero_vec(size: usize) -> Vec<Fr> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value
    // Bulk allocate zeros, unsafely
    let result: Vec<Fr>;
    unsafe {
        let layout = std::alloc::Layout::array::<Fr>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut Fr;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}