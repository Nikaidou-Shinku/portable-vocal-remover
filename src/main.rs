#![feature(slice_first_last_chunk)]

mod model;
mod setup;
mod utils;

use std::fs::File;

use symphonia::core::{
  audio::SampleBuffer, codecs::CODEC_TYPE_NULL, errors::Error as SymphoniaError,
  io::MediaSourceStream, probe::Hint,
};

use setup::setup_tracing;

fn main() {
  setup_tracing();

  let src = File::open("input.flac").expect("Failed to open audio file");
  let mss = MediaSourceStream::new(Box::new(src), Default::default());

  let probed = symphonia::default::get_probe()
    .format(&Hint::new(), mss, &Default::default(), &Default::default())
    .expect("Unsupported format");

  let mut format = probed.format;
  let track = format
    .tracks()
    .iter()
    .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
    .expect("No supported audio tracks");

  let mut decoder = symphonia::default::get_codecs()
    .make(&track.codec_params, &Default::default())
    .expect("Unsupported codec");

  let track_id = track.id;

  let mut res: Vec<f32> = Vec::new();

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
        panic!("{err}");
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
        let &spec = decoded.spec();
        let duration = decoded.capacity().try_into().unwrap();
        let mut buf = SampleBuffer::<f32>::new(duration, spec);
        buf.copy_interleaved_ref(decoded);
        res.extend(buf.samples());
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
        panic!("{err}");
      }
    }
  }

  tracing::info!("End decode, {} samples read", res.len());
}
