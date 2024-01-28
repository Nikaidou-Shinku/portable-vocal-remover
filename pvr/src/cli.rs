use std::path::PathBuf;

use clap::Parser;

#[cfg(target_os = "windows")]
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

  #[arg(
    short,
    long,
    help = "The model used, leave blank to see all available models"
  )]
  #[arg(value_name = "PRESET")]
  pub preset: Option<usize>,

  #[arg(short, long, help = "Use DirectML backend for inference")]
  pub directml_backend: bool,

  #[arg(short, long, help = "Use CUDA backend for inference")]
  pub cuda_backend: bool,

  #[arg(short, long, help = "Use TensorRT backend for inference")]
  pub tensorrt_backend: bool,

  #[arg(short, long, help = "File format used to save results (wav/flac)")]
  #[arg(value_name = "FORMAT", default_value = "flac")]
  pub format: String,
}

#[cfg(not(target_os = "windows"))]
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

  #[arg(
    short,
    long,
    help = "The model used, leave blank to see all available models"
  )]
  #[arg(value_name = "PRESET")]
  pub preset: Option<usize>,

  #[arg(short, long, help = "Use CUDA backend for inference")]
  pub cuda_backend: bool,

  #[arg(short, long, help = "File format used to save results (wav/flac)")]
  #[arg(value_name = "FORMAT", default_value = "flac")]
  pub format: String,
}
