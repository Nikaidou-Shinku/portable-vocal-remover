use std::{env, path::PathBuf};

use ort::{GraphOptimizationLevel, Session};

use super::stft::Stft;
use super::MdxSeperator;

// TODO: auto check
// enum MdxType {
//   Vocals,
//   Instrumental,
//   Reverb,
// }

pub struct MdxConfig {
  pub name: &'static str,
  filename: &'static str,
  n_fft: usize,
  dim_t: u8,
  dim_f: usize,
  compensate: f64,
}

impl MdxConfig {
  pub const fn new(
    name: &'static str,
    filename: &'static str,
    n_fft: usize,
    dim_t: u8,
    dim_f: usize,
    compensate: f64,
  ) -> Self {
    Self {
      name,
      filename,
      n_fft,
      dim_t,
      dim_f,
      compensate,
    }
  }

  pub fn exists(&self) -> bool {
    let model_path = env::var("PVR_MODELS")
      .map(|s| PathBuf::from(s))
      .unwrap_or_else(|_| {
        env::current_exe()
          .expect("Failed to get exe path")
          .parent()
          .expect("Failed to get the parent path of exe")
          .join("models")
      })
      .join(self.filename);

    model_path.exists()
  }

  pub fn build(&self) -> MdxSeperator {
    tracing::info!(name = self.name, "Building model...");

    let model_path = env::var("PVR_MODELS")
      .map(|s| PathBuf::from(s))
      .unwrap_or_else(|_| {
        env::current_exe()
          .expect("Failed to get exe path")
          .parent()
          .expect("Failed to get the parent path of exe")
          .join("models")
      })
      .join(self.filename);

    let model = Session::builder()
      .expect("Failed to get ort session builder")
      .with_optimization_level(GraphOptimizationLevel::Level3)
      .expect("Failed to optimize ort session")
      .with_model_from_file(model_path)
      .expect("Failed to load onnx model");

    let stft = Stft::new(self.n_fft, 1024, self.dim_f);

    MdxSeperator {
      n_fft: self.n_fft,
      segment_size: 1 << self.dim_t, // TODO: support other segment size
      stft,
      model,
      compensate: self.compensate,
    }
  }
}
