//! With the `cuda` feature, the CUDA kernels compiled by nvcc to PTX, and the
//! PTX assembled by ptxas for the GPU found by `nvidia-smi` or the one
//! `CUDA_COMPUTE_CAP` names (`OUT_DIR/kernels.ptx`, `OUT_DIR/kernels.cubin`).
//!
//! The engine loads the machine code when the GPU is the one built for: a
//! driver older than the toolkit loads it, when it can't compile the
//! toolkit's PTX. Other GPUs compile the PTX.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cuda")]
    if let Err(e) = cuda() {
        panic!("cuda kernels: {e}");
    }
}

#[cfg(feature = "cuda")]
fn cuda() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use std::process::Command;

    let arch = cudaforge::detect_compute_cap()?;
    cudaforge::KernelBuilder::new()
        .compute_cap_arch(&arch.to_string())
        .source_files(["src/cuda/kernels.cu"])
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-lineinfo")
        .build_ptx()?;

    let out = PathBuf::from(std::env::var("OUT_DIR")?);
    let nvcc = cudaforge::CudaToolkit::detect()?.nvcc_path;
    let ptxas = nvcc.with_file_name(if cfg!(windows) { "ptxas.exe" } else { "ptxas" });
    let sm = arch.to_nvcc_arch();
    let status = Command::new(&ptxas)
        .arg(format!("-arch={sm}"))
        .arg("-O3")
        .arg(out.join("kernels.ptx"))
        .arg("-o")
        .arg(out.join("kernels.cubin"))
        .status()
        .map_err(|e| format!("{}: {e}", ptxas.display()))?;
    if !status.success() {
        return Err(format!("{} failed: {status}", ptxas.display()).into());
    }
    println!("cargo:rustc-env=BIV_CUDA_ARCH={sm}");
    Ok(())
}
