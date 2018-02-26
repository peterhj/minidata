use ::*;
use colorimage::*;
use extar::*;
use sharedmem::*;

use std::fs::{File};
use std::io::{Read, Cursor};
use std::path::{PathBuf};

#[derive(Clone)]
pub struct ImagenetVal {
  path:     PathBuf,
  mmap:     SharedMem<u8>,
  index:    Vec<(usize, usize, u32)>,
}

impl ImagenetVal {
  pub fn open(path: PathBuf) -> Result<ImagenetVal, ()> {
    let file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    let mut data = ImagenetVal{
      path: path,
      mmap: SharedMem::new(mmap),
      index:    vec![],
    };
    data._build_index();
    Ok(data)
  }

  pub fn _build_index(&mut self) {
    self.index.clear();
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    for im_entry in tar.raw_entries() {
      let im_entry = im_entry.unwrap();
      // TODO: read label.
      let offset = im_entry.entry_pos as _;
      let size = im_entry.entry_sz as _;
      let label = 0; // TODO
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
  path:     PathBuf,
  mmap:     SharedMem<u8>,
  index:    Vec<(usize, usize, u32)>,
}

impl ImagenetTrain {
  pub fn open(path: PathBuf) -> Result<ImagenetTrain, ()> {
    let file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    let mut data = ImagenetTrain{
      path: path,
      mmap: SharedMem::new(mmap),
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
        // TODO: read label.
        let offset = (tar_entry.entry_pos + im_entry.entry_pos) as _;
        let size = im_entry.entry_sz as _;
        let label = 0; // TODO
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
