mod mdxnet;

use candle_core::{Module, Tensor};
use candle_nn::{
  conv2d, linear, linear_no_bias, seq, Activation, BatchNorm, Conv2dConfig, GroupNorm, Sequential,
  VarBuilder,
};

use crate::utils::OptionExt;

pub enum Norm {
  RMSprop(BatchNorm),
  AdamW(GroupNorm),
}

impl Module for Norm {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    match self {
      Norm::RMSprop(m) => m.forward(x),
      Norm::AdamW(m) => m.forward(x),
    }
  }
}

struct Tfc {
  h: Sequential,
}

impl Tfc {
  fn new(
    c: usize,
    l: usize,
    k: usize,
    norm: impl Fn(usize, VarBuilder) -> Result<Norm, candle_core::Error>,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let mut h = seq();

    for i in 0..l {
      h = h.add(conv2d(
        c,
        c,
        k,
        Conv2dConfig {
          padding: k / 2,
          stride: 1,
          dilation: 1,
          groups: 1,
        },
        vb.pp(format!("h.{i}.0")),
      )?);

      h = h.add(norm(c, vb.pp(format!("h.{i}.1")))?);
      h = h.add(Activation::Relu);
    }

    Ok(Self { h })
  }
}

impl Module for Tfc {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    self.h.forward(x)
  }
}

struct DenseTfc {
  conv: Vec<Box<dyn Module>>,
}

impl DenseTfc {
  fn new(
    c: usize,
    l: usize,
    k: usize,
    norm: impl Fn(usize, VarBuilder) -> Result<Norm, candle_core::Error>,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let mut conv: Vec<Box<dyn Module>> = Vec::with_capacity(l * 3);

    for i in 0..l {
      conv.push(Box::new(conv2d(
        c,
        c,
        k,
        Conv2dConfig {
          padding: k / 2,
          stride: 1,
          dilation: 1,
          groups: 1,
        },
        vb.pp(format!("conv.{i}.0")),
      )?));

      conv.push(Box::new(norm(c, vb.pp(format!("conv.{i}.1")))?));
      conv.push(Box::new(Activation::Relu));
    }

    Ok(Self { conv })
  }
}

impl Module for DenseTfc {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    match self.conv.split_last() {
      Some((last, layers)) => {
        let res = layers
          .iter()
          .try_fold(x.clone(), |acc, v| Tensor::cat(&[v.forward(&acc)?, acc], 1))?;

        last.forward(&res)
      }
      None => Ok(x.clone()),
    }
  }
}

enum MaybeDenseTfc {
  Dense(DenseTfc),
  Else(Tfc),
}

impl Module for MaybeDenseTfc {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    match self {
      MaybeDenseTfc::Dense(m) => m.forward(x),
      MaybeDenseTfc::Else(m) => m.forward(x),
    }
  }
}

pub struct TfcTdf {
  tfc: MaybeDenseTfc,
  tdf: Option<Sequential>,
}

impl TfcTdf {
  pub fn new(
    c: usize,
    l: usize,
    f: usize,
    k: usize,
    bn: Option<usize>,
    dense: bool,
    bias: bool,
    norm: impl Fn(usize, VarBuilder) -> Result<Norm, candle_core::Error>,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let tfc = if dense {
      MaybeDenseTfc::Dense(DenseTfc::new(c, l, k, &norm, vb.pp("tfc"))?)
    } else {
      MaybeDenseTfc::Else(Tfc::new(c, l, k, &norm, vb.pp("tfc"))?)
    };

    let tdf = bn.try_map(|bn| {
      let mut tdf = seq();

      if bn == 0 {
        tdf = tdf.add(if bias { linear } else { linear_no_bias }(
          f,
          f,
          vb.pp("tdf.0"),
        )?);
        tdf = tdf.add(norm(c, vb.pp("tdf.1"))?);
        tdf = tdf.add(Activation::Relu);
      } else {
        tdf = tdf.add(if bias { linear } else { linear_no_bias }(
          f,
          f / bn,
          vb.pp("tdf.0"),
        )?);
        tdf = tdf.add(norm(c, vb.pp("tdf.1"))?);
        tdf = tdf.add(Activation::Relu);
        tdf = tdf.add(if bias { linear } else { linear_no_bias }(
          f / bn,
          f,
          vb.pp("tdf.3"),
        )?);
        tdf = tdf.add(norm(c, vb.pp("tdf.4"))?);
        tdf = tdf.add(Activation::Relu);
      }

      Ok::<_, candle_core::Error>(tdf)
    })?;

    Ok(Self { tfc, tdf })
  }
}

impl Module for TfcTdf {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mut x = self.tfc.forward(x)?;

    if let Some(tdf) = &self.tdf {
      x = (&x + tdf.forward(&x))?;
    }

    Ok(x)
  }
}
