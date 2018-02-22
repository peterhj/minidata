extern crate superdata;

use superdata::datasets::imagenet::*;

use std::env;
use std::path::{PathBuf};

fn main() {
  let args: Vec<_> = env::args().collect();
  let path = PathBuf::from(&args[1]);
  //let dataset = ImagenetVal::open(path).unwrap();
  let dataset = ImagenetTrain::open(path).unwrap();
  dataset.test_images();
}
