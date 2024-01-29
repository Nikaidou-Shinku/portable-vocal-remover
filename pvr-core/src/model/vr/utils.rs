use burn::{
  module::{AutodiffModule, ConstantRecord, Devices, Module, ModuleMapper, ModuleVisitor},
  nn::ReLU,
  tensor::{
    backend::{AutodiffBackend, Backend},
    module::adaptive_avg_pool2d,
    ElementConversion, Tensor,
  },
};

#[derive(Clone, Debug, Module)]
struct LeakyReLU {
  negative_slope: f64,
}

impl LeakyReLU {
  pub fn new(negative_slope: f64) -> Self {
    Self { negative_slope }
  }

  pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
    let input = input.into_primitive();
    let positive_part = B::clamp_min(input.clone(), 0.elem());
    let negative_part = B::clamp_max(input, 0.elem());
    let negative_part = B::mul_scalar(negative_part, self.negative_slope.elem());
    Tensor::new(B::add(positive_part, negative_part))
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
