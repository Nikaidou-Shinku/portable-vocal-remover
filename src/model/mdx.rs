use std::ops::{AddAssign, MulAssign};

use ndarray::{concatenate, s, Array1, Array2, Array3, Array4, Axis};
use ort::{GraphOptimizationLevel, Session};

use super::stft::Stft;

fn hann_window(window_length: usize) -> Array1<f64> {
  if window_length == 0 {
    return Array1::zeros(0);
  }

  if window_length == 1 {
    return Array1::ones(1);
  }

  let half_length = (window_length + 1) / 2;
  let scaling = (std::f64::consts::PI * 2.0) / (window_length - 1) as f64;

  let mut res = Array1::zeros(window_length);

  for i in 0..half_length {
    let cur = 0.5 - 0.5 * (scaling * i as f64).cos();
    res[i] = cur;
    res[window_length - i - 1] = cur;
  }

  res
}

pub struct MdxSeperator {
  stft: Stft,
  model: Session,
}

// n_fft = 7680
// hop = 1024
// mdx_segment_size = 256
impl MdxSeperator {
  pub fn new() -> Self {
    let model = Session::builder()
      .unwrap()
      .with_optimization_level(GraphOptimizationLevel::Level3)
      .unwrap()
      .with_intra_threads(4)
      .unwrap()
      .with_model_from_file("models/UVR_MDXNET_Main.onnx")
      .unwrap();

    let stft = Stft::new(7680, 1024, 3072);

    Self { stft, model }
  }

  // fn seperate(&self, path: impl AsRef<Path>) {
  //   todo!()
  // }

  pub fn demix(&self, mix: Array2<f64> /*, is_match_mix: bool*/) -> Array2<f64> {
    let (_, length) = mix.dim();

    let trim = 7680 / 2;
    let chunk_size = 1024 * (256 - 1);
    let gen_size = chunk_size - 2 * trim;
    let pad = gen_size + trim - (length % gen_size);

    let mixture = concatenate(
      Axis(1),
      &[
        Array2::zeros((2, trim)).view(),
        mix.view(),
        Array2::zeros((2, pad)).view(),
      ],
    )
    .unwrap();

    let step = chunk_size - 7680;

    let new_len = trim + length + pad;

    let mut result: Array3<f64> = Array3::zeros((1, 2, new_len));
    let mut divider: Array3<f64> = Array3::zeros((1, 2, new_len));

    for i in (0..new_len).step_by(step) {
      let start = i;
      let end = (i + chunk_size).min(new_len);

      tracing::info!(
        "{:.2}% Processing... ({end}/{new_len})",
        end as f64 * 100.0 / new_len as f64
      );

      let window = {
        let window = hann_window(end - start);
        let mut res = Array3::zeros((1, 2, end - start));
        res.slice_mut(s![.., 0, ..]).assign(&window);
        res.slice_mut(s![.., 1, ..]).assign(&window);
        res
      };

      let mut mix_part = mixture.slice(s![.., start..end]).to_owned();

      if end != i + chunk_size {
        let pad_size = i + chunk_size - end;
        mix_part
          .append(Axis(1), Array2::zeros((2, pad_size)).view())
          .unwrap();
      }

      let mut tar_waves = self.run_model(mix_part.insert_axis(Axis(0)).to_owned());

      tar_waves
        .slice_mut(s![.., .., ..(end - start)])
        .mul_assign(&window);

      divider
        .slice_mut(s![.., .., start..end])
        .add_assign(&window);

      result
        .slice_mut(s![.., .., start..end])
        .add_assign(&tar_waves.slice(s![.., .., ..(end - start)]));
    }

    let tar_waves = result / divider;
    let tar_waves = tar_waves.remove_axis(Axis(0));

    let right = (new_len - trim).min(trim + length);
    let tar_waves = tar_waves.slice(s![.., trim..right]);

    let compensate = 1.043;

    tar_waves.mapv(|x| x * compensate)
  }

  fn run_model(&self, mix: Array3<f64>) -> Array3<f64> {
    let mut spek = self.stft.apply(mix);
    spek.slice_mut(s![.., .., ..3, ..]).fill(0.0);

    let spec_pred = self
      .model
      .run(ort::inputs![spek.mapv(|x| x as f32)].unwrap())
      .unwrap();
    let spec_pred: Array4<f32> = spec_pred["output"]
      .extract_tensor::<f32>()
      .unwrap()
      .view()
      .to_owned()
      .into_dimensionality()
      .unwrap();

    self.stft.inverse(spec_pred.mapv(|x| x.into()))
  }
}
