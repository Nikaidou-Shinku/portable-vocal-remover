use pvr_core::config::Backend;
use smallvec::SmallVec;

use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use crate::cli::Cli;

pub fn setup_tracing() {
  let subscriber = FmtSubscriber::builder()
    .with_max_level(Level::INFO)
    .with_target(false)
    .finish();

  tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");
}

pub fn setup_ort(args: &Cli) {
  let mut backends: SmallVec<[_; 3]> = SmallVec::new();

  #[cfg(target_os = "windows")]
  if args.directml_backend {
    backends.push(Backend::DirectML);
  }

  if args.cuda_backend {
    backends.push(Backend::CUDA);
  }

  #[cfg(target_os = "windows")]
  if args.tensorrt_backend {
    backends.push(Backend::TensorRT);
  }

  if backends.is_empty() {
    tracing::warn!("No backend is specified, use CPU for inference...");
    backends.push(Backend::CPU);
  }

  pvr_core::config::setup_backends(backends).expect("Init ort execution providers failed");
}
