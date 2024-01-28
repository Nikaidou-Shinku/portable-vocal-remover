mod cli;
mod model;
mod setup;
mod util;

use clap::Parser;

use cli::Cli;
use model::mdx::preset::MDX_PRESETS;
use setup::{setup_ort, setup_tracing};
use util::{read_audio, write_audio, AudioFormat};

fn main() {
  let args = Cli::parse();

  let Some(preset) = args.preset else {
    println!("Please specify the model you wish to use");
    println!("All available models:");
    for (id, p) in MDX_PRESETS.iter().enumerate() {
      if p.exists() {
        println!("{id}. {}", p.name);
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

  let mdx = MDX_PRESETS[preset].build();

  let mix = match read_audio(&args.input_path) {
    Ok(mix) => mix,
    Err(err) => {
      tracing::error!(%err, "Failed to read audio");
      return;
    }
  };

  let res = mdx.demix(mix.view()).unwrap();

  let filename = args
    .input_path
    .file_stem()
    .expect("Failed to get input file stem")
    .to_string_lossy();

  let primary_filename = format!("{filename}_primary.{}", output_format.extension());
  let others_filename = format!("{filename}_others.{}", output_format.extension());

  if let Err(err) = write_audio(
    args.output_path.join(primary_filename),
    res.view(),
    &output_format,
  ) {
    tracing::error!(%err, "Failed to write the primary audio");
  }

  if let Err(err) = write_audio(
    args.output_path.join(others_filename),
    (mix - res).view(),
    &output_format,
  ) {
    tracing::error!(%err, "Failed to write the others audio");
  }
}
