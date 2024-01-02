use ort::{CUDAExecutionProvider, TensorRTExecutionProvider};

use tracing::Level;
use tracing_subscriber::FmtSubscriber;

pub fn setup_tracing() {
  let subscriber = FmtSubscriber::builder()
    .with_max_level(Level::INFO)
    .with_target(false)
    .finish();

  tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");
}

pub fn setup_ort() {
  ort::init()
    .with_execution_providers([
      TensorRTExecutionProvider::default().build(),
      CUDAExecutionProvider::default().build(),
    ])
    .commit()
    .expect("Init ort execution providers failed");
}
