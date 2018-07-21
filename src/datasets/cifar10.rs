use ::*;

use rand::*;
use sharedmem::*;

use std::fs::{File};
//use std::io::{BufRead, Read, BufReader, Cursor};
use std::path::{PathBuf};

#[derive(Clone, Default, Debug)]
pub struct Cifar10Config {
  pub train_data:   Option<PathBuf>,
  pub test_data:    Option<PathBuf>,
}

pub struct Cifar10Data {
  cfg:  Cifar10Config,
  esz:  usize,
  num:  usize,
  mmap: SharedMem<u8>,
}

impl Cifar10Data {
  pub fn open_train(cfg: Cifar10Config) -> Result<Self, ()> {
    let file = File::open(cfg.train_data.as_ref().unwrap()).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let entry_sz = 1 + 32 * 32 * 3;
    let num_entries = file_len / entry_sz;
    assert_eq!(0, file_len % entry_sz);
    let mmap = SharedMem::from(MemoryMap::open_with_offset(file, 0, file_len).unwrap());
    Ok(Cifar10Data{
      cfg:  cfg,
      esz:  entry_sz,
      num:  num_entries,
      mmap: mmap,
    })
  }

  pub fn open_test(cfg: Cifar10Config) -> Result<Self, ()> {
    let file = File::open(cfg.test_data.as_ref().unwrap()).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let entry_sz = 1 + 32 * 32 * 3;
    let num_entries = file_len / entry_sz;
    assert_eq!(0, file_len % entry_sz);
    let mmap = SharedMem::from(MemoryMap::open_with_offset(file, 0, file_len).unwrap());
    Ok(Cifar10Data{
      cfg:  cfg,
      esz:  entry_sz,
      num:  num_entries,
      mmap: mmap,
    })
  }
}

impl RandomAccess for Cifar10Data {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.num
  }

  fn at(&mut self, idx: usize) -> (SharedMem<u8>, u32) {
    assert!(idx < self.num);
    let offset = idx * self.esz;
    let size = self.esz;
    let entry = self.mmap.shared_slice(offset .. offset + size);
    let label = entry[0] as u32;
    let value = entry.shared_slice(1 ..);
    (value, label)
  }
}
