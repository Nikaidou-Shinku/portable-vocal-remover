mod stft;

use candle_core::Module;
use candle_nn::{
  batch_norm, conv2d, group_norm, seq, Activation, BatchNormConfig, Conv2dConfig, Sequential,
  VarBuilder,
};

use crate::model::Norm;

enum Optimizer {
  RMSprop,
  AdamW,
}

pub struct ConvTDFNetConfig {
  dim_c: usize,
  g: usize,
  optimizer: Optimizer,
  num_blocks: usize,
}

pub struct ConvTDFNet {
  first_conv: Sequential,
  num_blocks: usize,
  encoding_blocks: Vec<Box<dyn Module>>,
  ds: Vec<Box<dyn Module>>,
  us: Vec<Box<dyn Module>>,
}

impl ConvTDFNet {
  pub fn new(config: ConvTDFNetConfig, vb: VarBuilder) -> Result<Self, candle_core::Error> {
    let norm = match config.optimizer {
      Optimizer::RMSprop => {
        |n, vb| batch_norm(n, BatchNormConfig::default(), vb).map(Norm::RMSprop)
      }
      Optimizer::AdamW => |n, vb| group_norm(2, n, 1e-5, vb).map(Norm::AdamW),
    };

    let mut first_conv = seq();

    first_conv = first_conv.add(conv2d(
      config.dim_c,
      config.g,
      1,
      Conv2dConfig::default(),
      vb.pp("first_conv.0"),
    )?);

    first_conv = first_conv.add(norm(config.g, vb.pp("first_conv.1"))?);
    first_conv = first_conv.add(Activation::Relu);

    let n = config.num_blocks / 2;

    let encoding_blocks = Vec::with_capacity(n);
    let ds = Vec::with_capacity(n);

    for _ in 0..n {}

    let us = Vec::with_capacity(n);

    for _ in 0..n {}

    Ok(Self {
      first_conv,
      num_blocks: config.num_blocks,
      encoding_blocks,
      ds,
      us,
    })
  }
}
