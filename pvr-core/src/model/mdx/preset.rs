use super::{
  MdxConfig,
  MdxType::{Instrumental, Reverb, Vocals},
};

pub const MDX_PRESETS: [MdxConfig; 19] = [
  MdxConfig::new(
    "Kim Inst",
    "Kim_Inst.onnx",
    Instrumental,
    7680,
    8,
    3072,
    1.02,
  ),
  MdxConfig::new(
    "Kim Vocal 1",
    "Kim_Vocal_1.onnx",
    Vocals,
    7680,
    8,
    3072,
    1.043,
  ),
  MdxConfig::new(
    "Kim Vocal 2",
    "Kim_Vocal_2.onnx",
    Vocals,
    7680,
    8,
    3072,
    1.009,
  ),
  MdxConfig::new(
    "Reverb HQ",
    "Reverb_HQ_By_FoxJoy.onnx",
    Reverb,
    6144,
    9,
    3072,
    1.035,
  ),
  MdxConfig::new(
    "UVR-MDX-NET 1",
    "UVR_MDXNET_1_9703.onnx",
    Vocals,
    6144,
    8,
    2048,
    1.03,
  ),
  MdxConfig::new(
    "UVR-MDX-NET 2",
    "UVR_MDXNET_2_9682.onnx",
    Vocals,
    6144,
    8,
    2048,
    1.035,
  ),
  MdxConfig::new(
    "UVR-MDX-NET 3",
    "UVR_MDXNET_3_9662.onnx",
    Vocals,
    6144,
    8,
    2048,
    1.035,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Inst 1",
    "UVR-MDX-NET-Inst_1.onnx",
    Instrumental,
    7680,
    8,
    3072,
    1.045,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Inst 2",
    "UVR-MDX-NET-Inst_2.onnx",
    Instrumental,
    7680,
    8,
    3072,
    1.035,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Inst 3",
    "UVR-MDX-NET-Inst_3.onnx",
    Instrumental,
    7680,
    8,
    3072,
    1.028,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Inst HQ 1",
    "UVR-MDX-NET-Inst_HQ_1.onnx",
    Instrumental,
    6144,
    8,
    3072,
    1.035,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Inst HQ 2",
    "UVR-MDX-NET-Inst_HQ_2.onnx",
    Instrumental,
    6144,
    8,
    3072,
    1.033,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Inst HQ 3",
    "UVR-MDX-NET-Inst_HQ_3.onnx",
    Instrumental,
    6144,
    8,
    3072,
    1.022,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Inst Main",
    "UVR-MDX-NET-Inst_Main.onnx",
    Instrumental,
    5120,
    8,
    2048,
    1.025,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Karaoke",
    "UVR_MDXNET_KARA.onnx",
    Vocals,
    6144,
    8,
    2048,
    1.035,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Karaoke 2",
    "UVR_MDXNET_KARA_2.onnx",
    Instrumental,
    5120,
    8,
    2048,
    1.065,
  ),
  MdxConfig::new(
    "UVR-MDX-NET Main",
    "UVR_MDXNET_Main.onnx",
    Vocals,
    7680,
    8,
    3072,
    1.043,
  ),
  MdxConfig::new(
    "UVR-MDX-NET-Voc_FT",
    "UVR-MDX-NET-Voc_FT.onnx",
    Vocals,
    7680,
    8,
    3072,
    1.021,
  ),
  MdxConfig::new(
    "UVR_MDXNET_9482",
    "UVR_MDXNET_9482.onnx",
    Vocals,
    6144,
    8,
    2048,
    1.035,
  ),
];
