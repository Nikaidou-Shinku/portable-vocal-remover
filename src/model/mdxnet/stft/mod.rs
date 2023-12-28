mod hann;

use candle_core::{Module, Tensor};
use rustfft::FftPlanner;

fn stft(input: &Tensor, n_fft: usize, hop_length: usize) -> Result<Tensor, candle_core::Error> {
  let mut planner = FftPlanner::<f32>::new();
  let res = planner.plan_fft_forward(n_fft);

  todo!()
}

struct Stft {
  n_fft: usize,
  hop_length: usize,
}

impl Module for Stft {
  fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
    let Some((&[c, t], batch_dims)) = xs.dims().split_last_chunk::<2>() else {
      return Err(candle_core::Error::Msg("shape error".to_owned()).bt());
    };

    let mut x = xs.reshape(((), t))?;
    x = stft(&x, self.n_fft, self.hop_length)?;
    todo!()
  }
}
