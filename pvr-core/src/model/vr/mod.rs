mod layers;
mod utils;

use burn::{
  config::Config,
  module::Module,
  nn::conv::Conv2d,
  tensor::{backend::Backend, Tensor},
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
  fn init<B: Backend>(&self) -> BaseASPPNet<B> {
    let (enc5, aspp, dec5) = if self.nn_architecture == 129605 {
      let enc5 = layers::EncoderConfig::new(self.ch * 8, self.ch * 16)
        .with_stride(2)
        .init();
      let aspp = layers::ASPPModuleConfig::new(self.nn_architecture, self.ch * 16, self.ch * 32)
        .with_dilations(self.dilations)
        .init();
      let dec5 = layers::DecoderConfig::new(self.ch * (16 + 32), self.ch * 16).init();

      (Some(enc5), aspp, Some(dec5))
    } else {
      (
        None,
        layers::ASPPModuleConfig::new(self.nn_architecture, self.ch * 8, self.ch * 16)
          .with_dilations(self.dilations)
          .init(),
        None,
      )
    };

    BaseASPPNet {
      enc1: layers::EncoderConfig::new(self.nin, self.ch)
        .with_stride(2)
        .init(),
      enc2: layers::EncoderConfig::new(self.ch, self.ch * 2)
        .with_stride(2)
        .init(),
      enc3: layers::EncoderConfig::new(self.ch * 2, self.ch * 4)
        .with_stride(2)
        .init(),
      enc4: layers::EncoderConfig::new(self.ch * 4, self.ch * 8)
        .with_stride(2)
        .init(),
      enc5,
      aspp,
      dec5,
      dec4: layers::DecoderConfig::new(self.ch * (8 + 16), self.ch * 8).init(),
      dec3: layers::DecoderConfig::new(self.ch * (4 + 8), self.ch * 4).init(),
      dec2: layers::DecoderConfig::new(self.ch * (2 + 4), self.ch * 2).init(),
      dec1: layers::DecoderConfig::new(self.ch * (1 + 2), self.ch).init(),
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
  aux1_out: Conv2d<B>,
  aux2_out: Conv2d<B>,
}

impl<B: Backend> CascadedASPPNet<B> {
  fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    todo!()
  }
}
