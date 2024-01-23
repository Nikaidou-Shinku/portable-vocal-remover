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
    backends.push(ort::DirectMLExecutionProvider::default().build());
  }

  if args.cuda_backend {
    backends.push(ort::CUDAExecutionProvider::default().build());
  }

  #[cfg(target_os = "windows")]
  if args.tensorrt_backend {
    backends.push(ort::TensorRTExecutionProvider::default().build());
  }

  if backends.is_empty() {
    tracing::warn!("No backend is specified, use CPU for inference...");
  }

  ort::init()
    .with_execution_providers(backends)
    .commit()
    .expect("Init ort execution providers failed");
}
