fn main() {
    tracing_texray::init();

    poly_bind_bench::sumcheck::bench::main();
}