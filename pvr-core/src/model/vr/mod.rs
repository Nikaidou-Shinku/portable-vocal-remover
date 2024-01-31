mod layers;
mod utils;

use burn::{
  config::Config,
  module::Module,
  nn::conv::{Conv2d, Conv2dConfig},
  tensor::{activation::sigmoid, backend::Backend, Tensor},
};

#[derive(Debug, Module)]
struct BaseASPPNet<B: Backend> {
  enc1: layers::Encoder<B>,
  enc2: layers::Encoder<B>,
  enc3: layers::Encoder<B>,
  enc4: layers::Encoder<B>,
  enc5: Option<layers::Encoder<B>>,
  aspp: layers::ASPPModule<B>,
  dec5: Option<layers::Decoder<B>>,
  dec4: layers::Decoder<B>,
  dec3: layers::Decoder<B>,
  dec2: layers::Decoder<B>,
  dec1: layers::Decoder<B>,
}

impl<B: Backend> BaseASPPNet<B> {
  fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let (h, e1) = self.enc1.forward(x);
    let (h, e2) = self.enc2.forward(h);
    let (h, e3) = self.enc3.forward(h);
    let (h, e4) = self.enc4.forward(h);

    let h = if let (Some(enc5), Some(dec5)) = (&self.enc5, &self.dec5) {
      let (h, e5) = enc5.forward(h);
      let h = self.aspp.forward(h);
      dec5.forward(h, Some(e5))
    } else {
      self.aspp.forward(h)
    };

    let h = self.dec4.forward(h, Some(e4));
    let h = self.dec3.forward(h, Some(e3));
    let h = self.dec2.forward(h, Some(e2));
    self.dec1.forward(h, Some(e1))
  }
}

#[derive(Config)]
struct BaseASPPNetConfig {
  nn_architecture: usize,
  nin: usize,
  ch: usize,
  #[config(default = "[4, 8, 16]")]
  dilations: [usize; 3],
}

impl BaseASPPNetConfig {
  fn init_with<B: Backend>(&self, record: BaseASPPNetRecord<B>) -> BaseASPPNet<B> {
    let (enc5, aspp, dec5) = if self.nn_architecture == 129605 {
      let enc5 = layers::EncoderConfig::new(self.ch * 8, self.ch * 16)
        .with_stride(2)
        .init_with(record.enc5.expect("shit"));
      let aspp = layers::ASPPModuleConfig::new(self.nn_architecture, self.ch * 16, self.ch * 32)
        .with_dilations(self.dilations)
        .init_with(record.aspp);
      let dec5 = layers::DecoderConfig::new(self.ch * (16 + 32), self.ch * 16)
        .init_with(record.dec5.expect("shit"));

      (Some(enc5), aspp, Some(dec5))
    } else {
      (
        None,
        layers::ASPPModuleConfig::new(self.nn_architecture, self.ch * 8, self.ch * 16)
          .with_dilations(self.dilations)
          .init_with(record.aspp),
        None,
      )
    };

    BaseASPPNet {
      enc1: layers::EncoderConfig::new(self.nin, self.ch)
        .with_stride(2)
        .init_with(record.enc1),
      enc2: layers::EncoderConfig::new(self.ch, self.ch * 2)
        .with_stride(2)
        .init_with(record.enc2),
      enc3: layers::EncoderConfig::new(self.ch * 2, self.ch * 4)
        .with_stride(2)
        .init_with(record.enc3),
      enc4: layers::EncoderConfig::new(self.ch * 4, self.ch * 8)
        .with_stride(2)
        .init_with(record.enc4),
      enc5,
      aspp,
      dec5,
      dec4: layers::DecoderConfig::new(self.ch * (8 + 16), self.ch * 8).init_with(record.dec4),
      dec3: layers::DecoderConfig::new(self.ch * (4 + 8), self.ch * 4).init_with(record.dec3),
      dec2: layers::DecoderConfig::new(self.ch * (2 + 4), self.ch * 2).init_with(record.dec2),
      dec1: layers::DecoderConfig::new(self.ch * (1 + 2), self.ch).init_with(record.dec1),
    }
  }
}

#[derive(Debug, Module)]
struct CascadedASPPNet<B: Backend> {
  stg1_low_band_net: BaseASPPNet<B>,
  stg1_high_band_net: BaseASPPNet<B>,
  stg2_bridge: layers::Conv2DBNActiv<B>,
  stg2_full_band_net: BaseASPPNet<B>,
  stg3_bridge: layers::Conv2DBNActiv<B>,
  stg3_full_band_net: BaseASPPNet<B>,
  out: Conv2d<B>,
  max_bin: usize,
  output_bin: usize,
}

impl<B: Backend> CascadedASPPNet<B> {
  fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [d0, d1, _, d3] = x.dims();
    let x = x.slice([0..d0, 0..d1, 0..self.max_bin, 0..d3]);

    let [d0, d1, d2, d3] = x.dims();
    let bandw = d2 / 2;

    let aux1 = Tensor::cat(
      vec![
        self
          .stg1_low_band_net
          .forward(x.clone().slice([0..d0, 0..d1, 0..bandw, 0..d3])),
        self
          .stg1_high_band_net
          .forward(x.clone().slice([0..d0, 0..d1, bandw..d2, 0..d3])),
      ],
      2,
    );

    let h = Tensor::cat(vec![x.clone(), aux1.clone()], 1);
    let aux2 = self.stg2_full_band_net.forward(self.stg2_bridge.forward(h));

    let h = Tensor::cat(vec![x, aux1, aux2], 1);
    let h = self.stg3_full_band_net.forward(self.stg3_bridge.forward(h));

    let mask = sigmoid(self.out.forward(h));
    let [d0, d1, d2, d3] = mask.dims();

    let border = mask.clone().slice([0..d0, 0..d1, (d2 - 1)..d2, 0..d3]);
    let border = border.repeat(2, self.output_bin - d2);

    Tensor::cat(vec![mask, border], 2)
  }
}

#[derive(Config)]
struct CascadedASPPNetConfig {
  n_fft: usize,
  nn_architecture: usize,
}

impl CascadedASPPNetConfig {
  fn init_with<B: Backend>(&self, record: CascadedASPPNetRecord<B>) -> CascadedASPPNet<B> {
    const SP_MODEL_ARCH: [usize; 3] = [31191, 33966, 129605];
    const HP_MODEL_ARCH: [usize; 2] = [123821, 123812];
    const HP2_MODEL_ARCH: [usize; 2] = [537238, 537227];

    if SP_MODEL_ARCH.contains(&self.nn_architecture) {
      CascadedASPPNet {
        stg1_low_band_net: BaseASPPNetConfig::new(self.nn_architecture, 2, 16)
          .init_with(record.stg1_low_band_net),
        stg1_high_band_net: BaseASPPNetConfig::new(self.nn_architecture, 2, 16)
          .init_with(record.stg1_high_band_net),
        stg2_bridge: layers::Conv2DBNActivConfig::new(18, 8)
          .with_ksize(1)
          .with_pad(0)
          .init_with(record.stg2_bridge),
        stg2_full_band_net: BaseASPPNetConfig::new(self.nn_architecture, 8, 16)
          .init_with(record.stg2_full_band_net),
        stg3_bridge: layers::Conv2DBNActivConfig::new(34, 16)
          .with_ksize(1)
          .with_pad(0)
          .init_with(record.stg3_bridge),
        stg3_full_band_net: BaseASPPNetConfig::new(self.nn_architecture, 16, 32)
          .init_with(record.stg3_full_band_net),
        out: Conv2dConfig::new([32, 2], [1, 1])
          .with_bias(false)
          .init_with(record.out),
        max_bin: self.n_fft / 2,
        output_bin: self.n_fft / 2 + 1,
      }
    } else if HP_MODEL_ARCH.contains(&self.nn_architecture) {
      CascadedASPPNet {
        stg1_low_band_net: BaseASPPNetConfig::new(self.nn_architecture, 2, 32)
          .init_with(record.stg1_low_band_net),
        stg1_high_band_net: BaseASPPNetConfig::new(self.nn_architecture, 2, 32)
          .init_with(record.stg1_high_band_net),
        stg2_bridge: layers::Conv2DBNActivConfig::new(34, 16)
          .with_ksize(1)
          .with_pad(0)
          .init_with(record.stg2_bridge),
        stg2_full_band_net: BaseASPPNetConfig::new(self.nn_architecture, 16, 32)
          .init_with(record.stg2_full_band_net),
        stg3_bridge: layers::Conv2DBNActivConfig::new(66, 32)
          .with_ksize(1)
          .with_pad(0)
          .init_with(record.stg3_bridge),
        stg3_full_band_net: BaseASPPNetConfig::new(self.nn_architecture, 32, 64)
          .init_with(record.stg3_full_band_net),
        out: Conv2dConfig::new([64, 2], [1, 1])
          .with_bias(false)
          .init_with(record.out),
        max_bin: self.n_fft / 2,
        output_bin: self.n_fft / 2 + 1,
      }
    } else if HP2_MODEL_ARCH.contains(&self.nn_architecture) {
      CascadedASPPNet {
        stg1_low_band_net: BaseASPPNetConfig::new(self.nn_architecture, 2, 64)
          .init_with(record.stg1_low_band_net),
        stg1_high_band_net: BaseASPPNetConfig::new(self.nn_architecture, 2, 64)
          .init_with(record.stg1_high_band_net),
        stg2_bridge: layers::Conv2DBNActivConfig::new(66, 32)
          .with_ksize(1)
          .with_pad(0)
          .init_with(record.stg2_bridge),
        stg2_full_band_net: BaseASPPNetConfig::new(self.nn_architecture, 32, 64)
          .init_with(record.stg2_full_band_net),
        stg3_bridge: layers::Conv2DBNActivConfig::new(130, 64)
          .with_ksize(1)
          .with_pad(0)
          .init_with(record.stg3_bridge),
        stg3_full_band_net: BaseASPPNetConfig::new(self.nn_architecture, 64, 128)
          .init_with(record.stg3_full_band_net),
        out: Conv2dConfig::new([128, 2], [1, 1])
          .with_bias(false)
          .init_with(record.out),
        max_bin: self.n_fft / 2,
        output_bin: self.n_fft / 2 + 1,
      }
    } else {
      unreachable!()
    }
  }
}
