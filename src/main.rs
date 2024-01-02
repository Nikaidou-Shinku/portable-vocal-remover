mod model;
mod setup;
mod util;

use model::mdx::MdxSeperator;
use setup::{setup_ort, setup_tracing};
use util::read_audio;

fn main() {
  setup_tracing();
  setup_ort();

  let mdx = MdxSeperator::new();
  let mix = read_audio("input.flac").unwrap();

  tracing::info!("Start seperating...");

  // TODO
  mdx.demix(mix, false);
}
