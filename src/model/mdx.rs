use std::path::Path;

use ndarray::{concatenate, s, Array2, Array3, Array4, Axis};
use ort::{GraphOptimizationLevel, Session};

use super::stft::Stft;

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

  fn seperate(&self, path: impl AsRef<Path>) {
    todo!()
  }

  pub fn demix(&self, mix: Array2<f32>, is_match_mix: bool) {
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

    for i in (0..length).step_by(step) {
      let start = i;
      let end = (i + chunk_size).min(length);

      let mut mix_part = mixture.slice(s![.., start..end]).to_owned();

      if end != i + chunk_size {
        let pad_size = i + chunk_size - end;
        mix_part = concatenate(
          Axis(1),
          &[mix_part.view(), Array2::zeros((2, pad_size)).view()],
        )
        .unwrap();
      }

      // TODO: split, run model for each

      self.run_model(mix_part.insert_axis(Axis(0)).to_owned());

      todo!()
    }

    todo!()
  }

  fn run_model(&self, mix: Array3<f32>) {
    let mut spek = self.stft.apply(mix);
    spek.slice_mut(s![.., .., ..3, ..]).fill(0.0);

    let spec_pred = self.model.run(ort::inputs![spek].unwrap()).unwrap();
    let spec_pred: Array4<f32> = spec_pred["output"]
      .extract_tensor::<f32>()
      .unwrap()
      .view()
      .to_owned()
      .into_dimensionality()
      .unwrap();

    println!("{spec_pred}");

    todo!("istft")
  }
}
