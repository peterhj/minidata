extern crate superdata;

use superdata::datasets::imagenet::*;

use std::env;
use std::path::{PathBuf};

fn main() {
  let args: Vec<_> = env::args().collect();
  let path = PathBuf::from(&args[1]);
  let meta_path = PathBuf::from(&args[2]);
  //let dataset = ImagenetVal::open(path, meta_path).unwrap();
  let dataset = ImagenetTrain::open(path, meta_path).unwrap();
  dataset.test_images();
}
