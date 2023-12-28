pub trait OptionExt {
  type Inner;
  type Container<T>;

  fn try_map<U, F, E>(self, f: F) -> Result<Self::Container<U>, E>
  where
    F: FnOnce(Self::Inner) -> Result<U, E>;
}

impl<T> OptionExt for Option<T> {
  type Inner = T;
  type Container<U> = Option<U>;

  fn try_map<U, F, E>(self, f: F) -> Result<Option<U>, E>
  where
    F: FnOnce(T) -> Result<U, E>,
  {
    match self {
      Some(x) => f(x).map(Some),
      None => Ok(None),
    }
  }
}
