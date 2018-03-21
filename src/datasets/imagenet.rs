use ::*;
use colorimage::*;
use extar::*;
use sharedmem::*;

use std::collections::{HashMap};
use std::fs::{File};
use std::io::{BufRead, Read, BufReader, Cursor};
use std::path::{PathBuf};

#[derive(Clone, Default, Debug)]
pub struct ImagenetConfig {
  pub train_data:       Option<PathBuf>,
  pub val_data:         Option<PathBuf>,
  pub wordnet_ids:      Option<PathBuf>,
  pub val_ground_truth: Option<PathBuf>,
}

#[derive(Clone)]
pub struct ImagenetVal {
  cfg:      ImagenetConfig,
  labels:   Vec<u32>,
  mmap:     SharedMem<u8>,
  index:    Vec<(usize, usize, u32)>,
}

impl ImagenetVal {
  pub fn open(cfg: ImagenetConfig) -> Result<ImagenetVal, ()> {
    let mut labels = vec![];
    let mut label_file = BufReader::new(File::open(cfg.val_ground_truth.as_ref().unwrap()).unwrap());
    for line in label_file.lines() {
      let line = line.unwrap();
      let token = line.split_whitespace().next().unwrap();
      let raw_label: u32 = token.parse().unwrap();
      labels.push(raw_label - 1);
    }
    println!("DEBUG: imagenet val: labels: {}", labels.len());
    let file = File::open(cfg.val_data.as_ref().unwrap()).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    let mut data = ImagenetVal{
      cfg:      cfg,
      labels:   labels,
      mmap:     SharedMem::new(mmap),
      index:    vec![],
    };
    data._build_index();
    Ok(data)
  }

  pub fn _build_index(&mut self) {
    self.index.clear();
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    for (idx, im_entry) in tar.raw_entries().enumerate() {
      let im_entry = im_entry.unwrap();
      let offset = im_entry.entry_pos as _;
      let size = im_entry.entry_sz as _;
      let label = self.labels[idx];
      self.index.push((offset, size, label));
    }
  }

  pub fn validate(&self) {
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    let mut jpeg_ct = 0;
    for entry in tar.raw_entries() {
      let entry = entry.unwrap();
      let buf = &self.mmap[entry.entry_pos as _ .. (entry.entry_pos + entry.entry_sz) as _];
      match guess_image_format_from_magicnum(buf) {
        Some(ImageFormat::Jpeg) => jpeg_ct += 1,
        fm => println!("DEBUG: found non-jpeg: {:?}", fm),
      }
    }
    println!("DEBUG: val set: jpegs: {}", jpeg_ct);
  }
}

impl RandomAccess for ImagenetVal {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.index.len()
  }

  fn at(&self, idx: usize) -> (SharedMem<u8>, u32) {
    let (offset, size, label) = self.index[idx];
    let value = self.mmap.shared_slice(offset .. offset + size);
    (value, label)
  }
}

#[derive(Clone)]
pub struct ImagenetTrain {
  cfg:      ImagenetConfig,
  labels:   HashMap<String, u32>,
  mmap:     SharedMem<u8>,
  index:    Vec<(usize, usize, u32)>,
}

impl ImagenetTrain {
  pub fn open(cfg: ImagenetConfig) -> Result<ImagenetTrain, ()> {
    let mut labels = HashMap::new();
    let mut label_file = BufReader::new(File::open(cfg.wordnet_ids.as_ref().unwrap()).unwrap());
    for (row_idx, line) in label_file.lines().enumerate() {
      let line = line.unwrap();
      let token = line.split_whitespace().next().unwrap();
      labels.insert(token.to_owned(), row_idx as _);
    }
    println!("DEBUG: imagenet train: labels: {}", labels.len());
    let file = File::open(cfg.train_data.as_ref().unwrap()).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    let mut data = ImagenetTrain{
      cfg:      cfg,
      labels:   labels,
      mmap:     SharedMem::new(mmap),
      index:    vec![],
    };
    data._build_index();
    Ok(data)
  }

  pub fn _build_index(&mut self) {
    self.index.clear();
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    for tar_entry in tar.raw_entries() {
      let tar_entry = tar_entry.unwrap();
      let archive_buf = &self.mmap[tar_entry.entry_pos as _ .. (tar_entry.entry_pos + tar_entry.entry_sz) as _];
      let archive_cursor = Cursor::new(archive_buf);
      let mut archive_tar = BufferedTarFile::new(archive_cursor);
      for im_entry in archive_tar.raw_entries() {
        let im_entry = im_entry.unwrap();
        let im_filename_toks: Vec<_> = im_entry.path.as_os_str().to_str().unwrap().splitn(2, ".").collect();
        let im_stem_toks: Vec<_> = im_filename_toks[0].splitn(2, "_").collect();
        let im_wnid = im_stem_toks[0].to_owned();
        let offset = (tar_entry.entry_pos + im_entry.entry_pos) as _;
        let size = im_entry.entry_sz as _;
        let label = *self.labels.get(&im_wnid).unwrap();
        self.index.push((offset, size, label));
      }
    }
  }

  pub fn validate(&self) {
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    let mut jpeg_ct = 0;
    for tar_entry in tar.raw_entries() {
      let tar_entry = tar_entry.unwrap();
      let archive_buf = &self.mmap[tar_entry.entry_pos as _ .. (tar_entry.entry_pos + tar_entry.entry_sz) as _];
      let archive_cursor = Cursor::new(archive_buf);
      let mut archive_tar = BufferedTarFile::new(archive_cursor);
      for im_entry in archive_tar.raw_entries() {
        let im_entry = im_entry.unwrap();
        let im_buf = &archive_buf[im_entry.entry_pos as _ .. (im_entry.entry_pos + im_entry.entry_sz) as _];
        match guess_image_format_from_magicnum(im_buf) {
          Some(ImageFormat::Jpeg) => jpeg_ct += 1,
          fm => println!("DEBUG: found non-jpeg: {:?}", fm),
        }
      }
    }
    println!("DEBUG: train set: jpegs: {}", jpeg_ct);
  }
}

impl RandomAccess for ImagenetTrain {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.index.len()
  }

  fn at(&self, idx: usize) -> (SharedMem<u8>, u32) {
    let (offset, size, label) = self.index[idx];
    let value = self.mmap.shared_slice(offset .. offset + size);
    (value, label)
  }
}
