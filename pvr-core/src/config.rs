use anyhow::Result;
use ort::{
  CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider,
  ExecutionProviderDispatch, TensorRTExecutionProvider,
};

pub enum Backend {
  DirectML,
  CUDA,
  TensorRT,
  CPU,
}

impl Backend {
  fn to_ep(&self) -> ExecutionProviderDispatch {
    match self {
      Self::DirectML => DirectMLExecutionProvider::default().build(),
      Self::CUDA => CUDAExecutionProvider::default().build(),
      Self::TensorRT => TensorRTExecutionProvider::default().build(),
      Self::CPU => CPUExecutionProvider::default().build(),
    }
  }
}

pub fn setup_backends(backends: impl AsRef<[Backend]>) -> Result<()> {
  let backends: Vec<_> = backends.as_ref().into_iter().map(|b| b.to_ep()).collect();
  ort::init().with_execution_providers(backends).commit()?;
  Ok(())
}
