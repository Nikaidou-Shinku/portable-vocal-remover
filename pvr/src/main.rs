mod cli;
mod setup;
mod util;

use clap::Parser;
use pvr_core::mdx::MDX_PRESETS;

use cli::Cli;
use setup::{setup_ort, setup_tracing};
use util::{read_audio, write_audio, AudioFormat};

fn main() {
  let args = Cli::parse();

  let Some(preset) = args.preset else {
    println!("Please specify the model you wish to use");
    println!("All available models:");
    for (id, p) in MDX_PRESETS.iter().enumerate() {
      if p.exists() {
        println!("{id}. {} ({})", p.name, p.model_type);
      }
    }
    return;
  };

  setup_tracing();
  setup_ort(&args);

  let output_format = match args.format.as_str() {
    "wav" => AudioFormat::Wav,
    "flac" => AudioFormat::Flac,
    _ => {
      tracing::error!(format = args.format, "Unknown output audio format");
      return;
    }
  };

  if !args.input_path.is_file() {
    tracing::error!(input = ?args.input_path, "Input path is not regular file");
    return;
  }

  if !args.output_path.is_dir() {
    tracing::error!(output = ?args.output_path, "Output path is not directory");
    return;
  }

  let preset = &MDX_PRESETS[preset];
  let mdx = match preset.build() {
    Ok(mdx) => mdx,
    Err(err) => {
      tracing::error!(%err, "Failed to build the model");
      return;
    }
  };

  let mix = match read_audio(&args.input_path) {
    Ok(mix) => mix,
    Err(err) => {
      tracing::error!(%err, "Failed to read audio");
      return;
    }
  };

  let res = match mdx.demix(mix.view()) {
    Ok(res) => res,
    Err(err) => {
      tracing::error!(%err, "Failed to inference");
      return;
    }
  };

  let origin_filename = args
    .input_path
    .file_stem()
    .expect("Failed to get input file stem")
    .to_string_lossy();

  let primary_filename = format!(
    "{origin_filename}_{}.{}",
    preset.model_type.get_primary_stem(),
    output_format.extension()
  );
  let secondary_filename = format!(
    "{origin_filename}_{}.{}",
    preset.model_type.get_secondary_stem(),
    output_format.extension()
  );

  if let Err(err) = write_audio(
    args.output_path.join(primary_filename),
    res.view(),
    &output_format,
  ) {
    tracing::error!(%err, "Failed to write the primary stem");
  }

  if let Err(err) = write_audio(
    args.output_path.join(secondary_filename),
    (mix - res).view(),
    &output_format,
  ) {
    tracing::error!(%err, "Failed to write the secondary stem");
  }
}
