use crate::poly::plain::DensePolynomial;
use crate::sumcheck::CubicSumcheck;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;
use rayon::prelude::*;

pub struct SIMDPolynomial {
    pub Z: Vec<Fr>,
}

impl SIMDPolynomial {
    pub fn bound_poly_var_top_par(&mut self, r: &Fr) {
        let n = self.Z.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        use vectorized_fields::*;

        let rayon_threads = rayon::current_num_threads();
        let chunk_size = (n / rayon_threads / 16) + 2; // Non-zero + better work-stealing
        let chunk_size = std::cmp::min(chunk_size, 512);

        let r = vec![r.clone(); chunk_size];
        left.par_chunks_mut(chunk_size)
            .zip(right.par_chunks_mut(chunk_size))
            .for_each(|(left_chunk, right_chunk)| {
                let chunk_size = left_chunk.len();

                sub_vec_inplace_bn254(right_chunk, left_chunk);
                mul_vec_inplace_bn254(right_chunk, &r[..chunk_size]);
                add_vec_inplace_bn254(left_chunk, right_chunk);
            });

        self.Z.truncate(n);
    }
}

pub struct SIMDSumcheck {
    eq: SIMDPolynomial,
    a: SIMDPolynomial,
    b: SIMDPolynomial,
}

impl CubicSumcheck for SIMDSumcheck {
    fn new(eq: Vec<Fr>, a: Vec<Fr>, b: Vec<Fr>) -> Self {
        let eq = SIMDPolynomial { Z: eq };
        let a = SIMDPolynomial { Z: a };
        let b = SIMDPolynomial { Z: b };

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
        let chunk_size = (n / rayon_threads / 32) + 2; // Non-zero + better work-stealing
        let chunk_size = std::cmp::min(chunk_size, 512);
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
                add_vec_inplace_bn254(&mut eq_2, &eq_m);
                add_vec_inplace_bn254(&mut a_2, &a_m);
                add_vec_inplace_bn254(&mut b_2, &b_m);
                mul_vec_inplace_bn254(&mut eq_2, &a_2);
                let eval_3 = inner_product_bn254(&eq_2, &b_2);

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
        rayon::join(
            || self.eq.bound_poly_var_top_par(r),
            || {
                rayon::join(
                    || self.a.bound_poly_var_top_par(r),
                    || self.b.bound_poly_var_top_par(r),
                )
            },
        );
    }
}

#[tracing::instrument(skip_all)]
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
