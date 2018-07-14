use ::*;

use byteorder::*;
use memarray::*;
use sharedmem::*;

use std::fs::{File};
use std::io::{Read, Cursor};
use std::path::{PathBuf};

#[derive(Clone, Default, Debug)]
pub struct MnistConfig {
  pub path:         Option<PathBuf>,
  pub train_images: Option<PathBuf>,
  pub train_labels: Option<PathBuf>,
  pub test_images:  Option<PathBuf>,
  pub test_labels:  Option<PathBuf>,
}

pub struct MnistData {
  cfg:      MnistConfig,
  num:      usize,
  nrows:    usize,
  ncols:    usize,
  images:   SharedMem<u8>,
  labels:   SharedMem<u8>,
}

impl MnistData {
  fn _open(cfg: MnistConfig, images_path: PathBuf, labels_path: PathBuf) -> Result<Self, ()> {
    let images_file = File::open(images_path).unwrap();
    let images_file_len = images_file.metadata().unwrap().len() as usize;
    let images_mmap = SharedMem::new(MemoryMap::open_with_offset(images_file, 0, images_file_len).unwrap());
    let (num_items, num_rows, num_cols) = {
      let mut reader = Cursor::new(images_mmap.clone());
      let magic = reader.read_u32::<BigEndian>().unwrap();
      assert_eq!(0x803, magic);
      let num_items = reader.read_u32::<BigEndian>().unwrap();
      let num_rows = reader.read_u32::<BigEndian>().unwrap();
      let num_cols = reader.read_u32::<BigEndian>().unwrap();
      (num_items, num_rows, num_cols)
    };
    let labels_file = File::open(labels_path).unwrap();
    let labels_file_len = labels_file.metadata().unwrap().len() as usize;
    let labels_mmap = SharedMem::new(MemoryMap::open_with_offset(labels_file, 0, labels_file_len).unwrap());
    let num_labels = {
      let mut reader = Cursor::new(labels_mmap.clone());
      let magic = reader.read_u32::<BigEndian>().unwrap();
      assert_eq!(0x801, magic);
      let num_items = reader.read_u32::<BigEndian>().unwrap();
      num_items
    };
    assert_eq!(num_items, num_labels);
    let entry_sz = num_rows * num_cols;
    assert_eq!(16 + (entry_sz * num_items) as usize, images_file_len);
    assert_eq!(8 + num_items as usize, labels_file_len);
    Ok(MnistData{
      cfg:      cfg,
      num:      num_items as usize,
      nrows:    num_rows as usize,
      ncols:    num_cols as usize,
      images:   images_mmap,
      labels:   labels_mmap,
    })
  }

  pub fn open_train(cfg: MnistConfig) -> Result<Self, ()> {
    let train_images_path = cfg.train_images.as_ref().map(|path| path.clone()).unwrap_or(cfg.path.as_ref().map(|path| path.join("train-images-idx3-ubyte")).unwrap());
    let train_labels_path = cfg.train_labels.as_ref().map(|path| path.clone()).unwrap_or(cfg.path.as_ref().map(|path| path.join("train-labels-idx1-ubyte")).unwrap());
    Self::_open(cfg, train_images_path, train_labels_path)
  }

  pub fn open_test(cfg: MnistConfig) -> Result<Self, ()> {
    let test_images_path = cfg.test_images.as_ref().map(|path| path.clone()).unwrap_or(cfg.path.as_ref().map(|path| path.join("t10k-images-idx3-ubyte")).unwrap());
    let test_labels_path = cfg.test_labels.as_ref().map(|path| path.clone()).unwrap_or(cfg.path.as_ref().map(|path| path.join("t10k-labels-idx1-ubyte")).unwrap());
    Self::_open(cfg, test_images_path, test_labels_path)
  }
}

impl RandomAccess for MnistData {
  type Item = (MemArray3d<u8>, u32);

  fn len(&self) -> usize {
    self.num
  }

  fn at(&mut self, idx: usize) -> (MemArray3d<u8>, u32) {
    assert!(idx < self.num);
    let im_size = self.nrows * self.ncols;
    let im_offset = 16 + idx * im_size;
    let im_value = self.images.shared_slice(im_offset .. im_offset + im_size);
    let mut image = MemArray3d::zeros([self.ncols, self.nrows, 1]);
    image.flat_view_mut().unwrap().as_mut_slice().copy_from_slice(&*im_value);
    let label_offset = 8 + idx;
    let label = self.labels.shared_slice(label_offset .. label_offset + 1)[0] as u32;
    (image, label)
  }
}
