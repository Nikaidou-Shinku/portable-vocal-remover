use burn::{
  config::Config,
  module::Module,
  nn::{
    conv::{Conv2d, Conv2dConfig},
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, PaddingConfig2d,
  },
  tensor::{backend::Backend, Tensor},
};

use super::utils::{Activ, AdaptiveAvgPool2d};

#[derive(Debug, Module)]
pub struct Conv2DBNActiv<B: Backend> {
  conv0: Conv2d<B>,
  conv1: BatchNorm<B, 2>,
  conv2: Activ,
}

impl<B: Backend> Conv2DBNActiv<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv0.forward(x);
    let x = self.conv1.forward(x);
    self.conv2.forward(x)
  }
}

#[derive(Config)]
struct Conv2DBNActivConfig {
  nin: usize,
  nout: usize,
  #[config(default = 3)]
  ksize: usize,
  #[config(default = 1)]
  stride: usize,
  #[config(default = 1)]
  pad: usize,
  #[config(default = 1)]
  dilation: usize,
  #[config(default = false)]
  leaky: bool,
}

impl Conv2DBNActivConfig {
  fn init<B: Backend>(&self) -> Conv2DBNActiv<B> {
    Conv2DBNActiv {
      conv0: Conv2dConfig::new([self.nin, self.nout], [self.ksize, self.ksize])
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(self.pad, self.pad))
        .with_dilation([self.dilation, self.dilation])
        .with_bias(false)
        .init(),
      conv1: BatchNormConfig::new(self.nout).init(),
      conv2: if self.leaky {
        Activ::leaky_relu(0.01)
      } else {
        Activ::relu()
      },
    }
  }
}

#[derive(Debug, Module)]
pub struct Encoder<B: Backend> {
  conv1: Conv2DBNActiv<B>,
  conv2: Conv2DBNActiv<B>,
}

impl<B: Backend> Encoder<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let skip = self.conv1.forward(x);
    let h = self.conv2.forward(skip.clone());
    (h, skip)
  }
}

#[derive(Config)]
pub struct EncoderConfig {
  nin: usize,
  nout: usize,
  #[config(default = 3)]
  ksize: usize,
  #[config(default = 1)]
  stride: usize,
  #[config(default = 1)]
  pad: usize,
  #[config(default = true)]
  leaky: bool,
}

impl EncoderConfig {
  pub fn init<B: Backend>(&self) -> Encoder<B> {
    Encoder {
      conv1: Conv2DBNActivConfig::new(self.nin, self.nout)
        .with_ksize(self.ksize)
        .with_pad(self.pad)
        .with_leaky(self.leaky)
        .init(),
      conv2: Conv2DBNActivConfig::new(self.nout, self.nout)
        .with_ksize(self.ksize)
        .with_stride(self.stride)
        .with_pad(self.pad)
        .with_leaky(self.leaky)
        .init(),
    }
  }
}

#[derive(Debug, Module)]
pub struct Decoder<B: Backend> {
  conv: Conv2DBNActiv<B>,
  dropout: Option<Dropout>,
}

impl<B: Backend> Decoder<B> {
  pub fn forward(&self, x: Tensor<B, 4>, skip: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
    todo!()
  }
}

#[derive(Config)]
pub struct DecoderConfig {
  nin: usize,
  nout: usize,
  #[config(default = 3)]
  ksize: usize,
  #[config(default = 1)]
  pad: usize,
  #[config(default = false)]
  leaky: bool,
  #[config(default = false)]
  dropout: bool,
}

impl DecoderConfig {
  pub fn init<B: Backend>(&self) -> Decoder<B> {
    Decoder {
      conv: Conv2DBNActivConfig::new(self.nin, self.nout)
        .with_ksize(self.ksize)
        .with_pad(self.pad)
        .with_leaky(self.leaky)
        .init(),
      dropout: if self.dropout {
        Some(DropoutConfig::new(0.1).init())
      } else {
        None
      },
    }
  }
}

#[derive(Debug, Module)]
struct SeperableConv2DBNActiv<B: Backend> {
  conv0: Conv2d<B>,
  conv1: Conv2d<B>,
  conv2: BatchNorm<B, 2>,
  conv3: Activ,
}

impl<B: Backend> SeperableConv2DBNActiv<B> {
  fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv0.forward(x);
    let x = self.conv1.forward(x);
    let x = self.conv2.forward(x);
    self.conv3.forward(x)
  }
}

#[derive(Config)]
struct SeperableConv2DBNActivConfig {
  nin: usize,
  nout: usize,
  #[config(default = 3)]
  ksize: usize,
  #[config(default = 1)]
  stride: usize,
  #[config(default = 1)]
  pad: usize,
  #[config(default = 1)]
  dilation: usize,
  #[config(default = false)]
  leaky: bool,
}

impl SeperableConv2DBNActivConfig {
  fn init<B: Backend>(&self) -> SeperableConv2DBNActiv<B> {
    SeperableConv2DBNActiv {
      conv0: Conv2dConfig::new([self.nin, self.nin], [self.ksize, self.ksize])
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(self.pad, self.pad))
        .with_dilation([self.dilation, self.dilation])
        .with_groups(self.nin)
        .with_bias(false)
        .init(),
      conv1: Conv2dConfig::new([self.nin, self.nout], [1, 1])
        .with_bias(false)
        .init(),
      conv2: BatchNormConfig::new(self.nout).init(),
      conv3: if self.leaky {
        Activ::leaky_relu(0.01)
      } else {
        Activ::relu()
      },
    }
  }
}

#[derive(Debug, Module)]
pub struct ASPPModule<B: Backend> {
  conv10: AdaptiveAvgPool2d,
  conv11: Conv2DBNActiv<B>,
  conv2: Conv2DBNActiv<B>,
  conv3: SeperableConv2DBNActiv<B>,
  conv4: SeperableConv2DBNActiv<B>,
  conv5: SeperableConv2DBNActiv<B>,
  conv6: Option<SeperableConv2DBNActiv<B>>,
  conv7: Option<SeperableConv2DBNActiv<B>>,
  bottleneck0: Conv2DBNActiv<B>,
  bottleneck1: Dropout,
}

impl<B: Backend> ASPPModule<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    todo!()
  }
}

#[derive(Config)]
pub struct ASPPModuleConfig {
  nn_architecture: usize,
  nin: usize,
  nout: usize,
  #[config(default = "[4, 8, 16]")]
  dilations: [usize; 3],
  #[config(default = false)]
  leaky: bool,
}

impl ASPPModuleConfig {
  pub fn init<B: Backend>(&self) -> ASPPModule<B> {
    const SIX_LAYER: [usize; 1] = [129605];
    const SEVEN_LAYER: [usize; 3] = [537238, 537227, 33966];

    let (conv6, conv7, nin_x) = if SIX_LAYER.contains(&self.nn_architecture) {
      (
        Some(
          SeperableConv2DBNActivConfig::new(self.nin, self.nin)
            .with_pad(self.dilations[2])
            .with_dilation(self.dilations[2])
            .with_leaky(self.leaky)
            .init(),
        ),
        None,
        6,
      )
    } else if SEVEN_LAYER.contains(&self.nn_architecture) {
      let extra_conv_config = SeperableConv2DBNActivConfig::new(self.nin, self.nin)
        .with_pad(self.dilations[2])
        .with_dilation(self.dilations[2])
        .with_leaky(self.leaky);

      (
        Some(extra_conv_config.init()),
        Some(extra_conv_config.init()),
        7,
      )
    } else {
      (None, None, 5)
    };

    ASPPModule {
      conv10: AdaptiveAvgPool2d::new([Some(1), None]),
      conv11: Conv2DBNActivConfig::new(self.nin, self.nin)
        .with_ksize(1)
        .with_pad(0)
        .with_leaky(self.leaky)
        .init(),
      conv2: Conv2DBNActivConfig::new(self.nin, self.nin)
        .with_ksize(1)
        .with_pad(0)
        .with_leaky(self.leaky)
        .init(),
      conv3: SeperableConv2DBNActivConfig::new(self.nin, self.nin)
        .with_pad(self.dilations[0])
        .with_dilation(self.dilations[0])
        .with_leaky(self.leaky)
        .init(),
      conv4: SeperableConv2DBNActivConfig::new(self.nin, self.nin)
        .with_pad(self.dilations[1])
        .with_dilation(self.dilations[1])
        .with_leaky(self.leaky)
        .init(),
      conv5: SeperableConv2DBNActivConfig::new(self.nin, self.nin)
        .with_pad(self.dilations[2])
        .with_dilation(self.dilations[2])
        .with_leaky(self.leaky)
        .init(),
      conv6,
      conv7,
      bottleneck0: Conv2DBNActivConfig::new(self.nin * nin_x, self.nout)
        .with_ksize(1)
        .with_pad(0)
        .with_leaky(self.leaky)
        .init(),
      bottleneck1: DropoutConfig::new(0.1).init(),
    }
  }
}
