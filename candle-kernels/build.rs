fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=src/fft.cu");

    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/ptx.rs").unwrap();
}
