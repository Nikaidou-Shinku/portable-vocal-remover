use burn::{
  module::{AutodiffModule, ConstantRecord, Devices, Module, ModuleMapper, ModuleVisitor},
  nn::ReLU,
  tensor::{
    backend::{AutodiffBackend, Backend},
    module::adaptive_avg_pool2d,
    Data, ElementConversion, Shape, Tensor,
  },
};

#[derive(Clone, Debug, Module)]
pub struct LeakyReLU {
  negative_slope: f64,
}

impl LeakyReLU {
  pub fn new(negative_slope: f64) -> Self {
    Self { negative_slope }
  }

  pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
    let input = input.into_primitive();
    let positive_part = B::float_clamp_min(input.clone(), 0.elem());
    let negative_part = B::float_clamp_max(input, 0.elem());
    let negative_part = B::float_mul_scalar(negative_part, self.negative_slope.elem());
    Tensor::new(B::float_add(positive_part, negative_part))
  }
}

#[derive(Clone, Debug)]
pub enum Activ {
  ReLU(ReLU),
  LeakyReLU(LeakyReLU),
}

impl<B: Backend> Module<B> for Activ {
  type Record = ConstantRecord;

  fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
    devices
  }

  fn fork(self, _device: &<B as Backend>::Device) -> Self {
    self
  }

  fn to_device(self, _device: &<B as Backend>::Device) -> Self {
    self
  }

  fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
    // Nothing to do
  }

  fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
    self
  }

  fn load_record(self, _record: Self::Record) -> Self {
    self
  }

  fn into_record(self) -> Self::Record {
    ConstantRecord::new()
  }
}

impl<B: AutodiffBackend> AutodiffModule<B> for Activ {
  type InnerModule = Activ;

  fn valid(&self) -> Self::InnerModule {
    self.clone()
  }
}

impl Activ {
  pub fn relu() -> Self {
    Self::ReLU(ReLU::new())
  }

  pub fn leaky_relu(negative_slope: f64) -> Self {
    Self::LeakyReLU(LeakyReLU::new(negative_slope))
  }

  pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
    match self {
      Self::ReLU(module) => module.forward(input),
      Self::LeakyReLU(module) => module.forward(input),
    }
  }
}

#[derive(Clone, Debug, Module)]
pub struct AdaptiveAvgPool2d {
  output_size: [Option<usize>; 2],
}

impl AdaptiveAvgPool2d {
  pub fn new(output_size: [Option<usize>; 2]) -> Self {
    Self { output_size }
  }

  pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
    let output_size = match self.output_size {
      [Some(height), Some(width)] => [height, width],
      [height, width] => {
        let input_dims = input.dims();
        [
          height.unwrap_or(input_dims[2]),
          width.unwrap_or(input_dims[3]),
        ]
      }
    };

    adaptive_avg_pool2d(input, output_size)
  }
}

// TODO: maybe replace this with convolution
pub fn bilinear_interpolate<B: Backend>(
  input: Tensor<B, 4>,
  h_out: usize,
  w_out: usize,
) -> Tensor<B, 4> {
  let device = input.device();
  let [batch, chan, h_in, w_in] = input.dims();
  let input = input.into_data().value;

  let istr = [chan * h_in * w_in, h_in * w_in, w_in, 1];
  let ostr = [chan * h_out * w_out, h_out * w_out, w_out, 1];

  let y_ratio = ((h_in - 1) as f64) / ((h_out - 1) as f64);
  let x_ratio = ((w_in - 1) as f64) / ((w_out - 1) as f64);

  let mut res = vec![0.elem(); batch * chan * h_out * w_out];

  for b in 0..batch {
    for c in 0..chan {
      for y_out in 0..h_out {
        for x_out in 0..w_out {
          let x_frac = x_ratio * x_out as f64;
          let x0 = x_frac.floor().min((w_in - 1) as f64);
          let x1 = x_frac.ceil().min((w_in - 1) as f64);
          let xw = x_frac - x0;

          let y_frac = y_ratio * y_out as f64;
          let y0 = y_frac.floor().min((h_in - 1) as f64);
          let y1 = y_frac.ceil().min((h_in - 1) as f64);
          let yw = y_frac - y0;

          let [x0, x1, y0, y1] = [x0, x1, y0, y1].map(|q| q as usize);

          let p_a = input[b * istr[0] + c * istr[1] + y0 * istr[2] + x0 * istr[3]];
          let p_b = input[b * istr[0] + c * istr[1] + y0 * istr[2] + x1 * istr[3]];
          let p_c = input[b * istr[0] + c * istr[1] + y1 * istr[2] + x0 * istr[3]];
          let p_d = input[b * istr[0] + c * istr[1] + y1 * istr[2] + x1 * istr[3]];

          let p_a = p_a.elem::<f64>() * (1.0 - xw) * (1.0 - yw);
          let p_b = p_b.elem::<f64>() * xw * (1.0 - yw);
          let p_c = p_c.elem::<f64>() * (1.0 - xw) * yw;
          let p_d = p_d.elem::<f64>() * xw * yw;

          res[b * ostr[0] + c * ostr[1] + y_out * ostr[2] + x_out * ostr[3]] =
            (p_a + p_b + p_c + p_d).elem();
        }
      }
    }
  }

  Tensor::from_data(
    Data::new(res, Shape::new([batch, chan, h_out, w_out])),
    &device,
  )
}

pub fn crop_center<B: Backend>(h1: Tensor<B, 4>, h2: Tensor<B, 4>) -> Tensor<B, 4> {
  let [b, c, h0, w0] = h1.dims();
  let [_, _, _, w1] = h2.dims();

  if w0 == w1 {
    return h1;
  } else if w0 < w1 {
    unreachable!("w0 = {w0}, w1 = {w1}")
  }

  let s_time = (w0 - w1) / 2;
  let e_time = s_time + w1;

  h1.slice([0..b, 0..c, 0..h0, s_time..e_time])
}
