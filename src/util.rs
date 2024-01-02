use std::{fs::File, path::Path};

use anyhow::{anyhow, bail, Result};
use ndarray::Array2;
use symphonia::core::{
  audio::{AudioBufferRef, Signal},
  codecs::CODEC_TYPE_NULL,
  conv::IntoSample,
  errors::Error as SymphoniaError,
  io::MediaSourceStream,
  probe::Hint,
};

#[tracing::instrument(skip_all)]
fn resample() {
  todo!()
}

#[tracing::instrument(skip_all)]
pub fn read_audio(path: impl AsRef<Path>) -> Result<Array2<f32>> {
  let src = File::open(path)?;
  let mss = MediaSourceStream::new(Box::new(src), Default::default());

  let probed = symphonia::default::get_probe().format(
    &Hint::new(),
    mss,
    &Default::default(),
    &Default::default(),
  )?;

  let mut format = probed.format;
  let track = format
    .tracks()
    .iter()
    .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
    .ok_or_else(|| anyhow!("No supported audio tracks"))?;

  let mut decoder =
    symphonia::default::get_codecs().make(&track.codec_params, &Default::default())?;

  let mut samples: Vec<Vec<f32>> = Vec::new();
  let mut sample_rate = track.codec_params.sample_rate;

  let track_id = track.id;

  tracing::info!("Start decoding...");

  loop {
    let packet = match format.next_packet() {
      Ok(packet) => packet,
      Err(SymphoniaError::ResetRequired) => {
        unimplemented!();
      }
      Err(SymphoniaError::IoError(err))
        if err.kind() == std::io::ErrorKind::UnexpectedEof
          && err.to_string() == "end of stream" =>
      {
        break;
      }
      Err(err) => {
        bail!(err);
      }
    };

    if packet.track_id() != track_id {
      tracing::warn!(
        timestamp = packet.ts,
        "The packet does not belong to the selected track, skip..."
      );
      continue;
    }

    match decoder.decode(&packet) {
      Ok(decoded) => {
        let spec = decoded.spec();

        if sample_rate.is_none() {
          sample_rate = Some(spec.rate);
        }

        let channel_num = spec.channels.count();

        if samples.len() < channel_num {
          samples.resize_with(channel_num, Vec::new);
        }

        match &decoded {
          AudioBufferRef::U8(_) => todo!(),
          AudioBufferRef::U16(_) => todo!(),
          AudioBufferRef::U24(_) => todo!(),
          AudioBufferRef::U32(_) => todo!(),
          AudioBufferRef::S8(_) => todo!(),
          AudioBufferRef::S16(_) => todo!(),
          AudioBufferRef::S24(_) => todo!(),
          AudioBufferRef::S32(buf) => {
            let f = |&v| <i32 as IntoSample<f32>>::into_sample(v);
            for ch in 0..channel_num {
              samples[ch].extend(buf.chan(ch).iter().map(f));
            }
          }
          AudioBufferRef::F32(buf) => {
            for ch in 0..channel_num {
              samples[ch].extend_from_slice(buf.chan(ch));
            }
          }
          AudioBufferRef::F64(_) => todo!(),
        }
      }
      Err(SymphoniaError::IoError(_)) => {
        tracing::error!(
          timestamp = packet.ts,
          "The packet failed to decode due to an IO error, skip..."
        );
        continue;
      }
      Err(SymphoniaError::DecodeError(_)) => {
        tracing::warn!(
          timestamp = packet.ts,
          "The packet failed to decode due to invalid data, skip..."
        );
        continue;
      }
      Err(err) => {
        bail!(err);
      }
    }
  }

  tracing::info!("Audio decoded");

  let sample_rate = sample_rate.ok_or_else(|| anyhow!("Can not get sample rate"))?;

  const SAMPLE_RATE: u32 = 44100;

  // TODO: resampling
  if sample_rate != SAMPLE_RATE {
    tracing::info!(sample_rate, "Start resampling...");
    resample();
  }

  let channel_num = samples.len();
  let length = samples
    .iter()
    .map(|c| c.len())
    .max()
    .ok_or_else(|| anyhow!("No channel found"))?;

  let res = Array2::from_shape_vec(
    (channel_num, length),
    samples.into_iter().flatten().collect(),
  )?;

  Ok(res)
}
