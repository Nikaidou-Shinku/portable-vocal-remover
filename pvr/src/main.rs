mod cli;
mod model;
mod setup;
mod util;

use clap::Parser;

use cli::Cli;
use model::mdx::MdxSeperator;
use setup::{setup_ort, setup_tracing};
use util::{read_audio, write_flac};

fn main() {
  let args = Cli::parse();

  setup_tracing();
  setup_ort(&args);

  if !args.input_path.is_file() {
    tracing::error!(input = ?args.input_path, "Input path is not regular file");
    return;
  }

  if !args.output_path.is_dir() {
    tracing::error!(output = ?args.output_path, "Output path is not directory");
    return;
  }

  let mdx = MdxSeperator::new();

  let mix = match read_audio(&args.input_path) {
    Ok(mix) => mix,
    Err(err) => {
      tracing::error!(%err, "Failed to read audio");
      return;
    }
  };

  let res = mdx.demix(mix.view());

  let filename = args
    .input_path
    .file_stem()
    .expect("Failed to get input file stem")
    .to_string_lossy();

  {
    let vocal_filename = format!("{filename}_vocal.flac");
    let inst_filename = format!("{filename}_inst.flac");

    write_flac(args.output_path.join(vocal_filename), res.view());
    write_flac(args.output_path.join(inst_filename), (mix - res).view());
  }

  // let vocal_filename = format!("{filename}_vocal.wav");
  // let inst_filename = format!("{filename}_inst.wav");

  // if let Err(err) = write_audio(args.output_path.join(vocal_filename), res.view()) {
  //   tracing::error!(%err, "Failed to write the vocal audio");
  // }

  // if let Err(err) = write_audio(args.output_path.join(inst_filename), (mix - res).view()) {
  //   tracing::error!(%err, "Failed to write the instrument audio");
  // }
}
