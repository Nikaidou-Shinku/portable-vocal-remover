use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
  #[error("the path `{0}` can not be converted into C-style string")]
  PathEncoding(PathBuf),
}

pub type Result<T> = ::std::result::Result<T, Error>;
