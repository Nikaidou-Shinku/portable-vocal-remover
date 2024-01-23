use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  #[error("the path `{0}` can not be converted into C-style string")]
  PathEncoding(PathBuf),
  #[error("General failure to set up encoder, code: {0}")]
  EncoderInitError(u32),
  #[error("Error occurred while encoding")]
  EncoderError,
}

pub type Result<T> = ::std::result::Result<T, Error>;
