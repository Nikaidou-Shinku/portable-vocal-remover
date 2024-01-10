use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(version, author)]
#[command(about = "A Rust port of UVR", long_about = None)]
pub struct Cli {
  #[arg(short, long, help = "Input audio file path")]
  #[arg(value_name = "INPUT")]
  pub input_path: PathBuf,

  #[arg(short, long, help = "Directory to save output audio")]
  #[arg(value_name = "OUTPUT", default_value = ".")]
  pub output_path: PathBuf,

  #[arg(short, long, help = "Use DirectML backend for inference")]
  pub directml_backend: bool,

  #[arg(short, long, help = "Use CUDA backend for inference")]
  pub cuda_backend: bool,

  #[arg(short, long, help = "Use TensorRT backend for inference")]
  pub tensorrt_backend: bool,
}
