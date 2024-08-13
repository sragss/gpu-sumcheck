use ark_bn254::Fr;
use crate::poly::gpu::GPUPoly;
use crate::sumcheck::CubicSumcheck;
use icicle_bn254::polynomials::DensePolynomial as IngoPoly;
use icicle_core::polynomials::UnivariatePolynomial;


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