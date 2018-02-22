use extar::*;
use imgio::*;
use sharedmem::*;

use std::fs::{File};
use std::io::{Read, Cursor};
use std::path::{PathBuf};

pub struct ImagenetVal {
  path: PathBuf,
  mmap: MemoryMap<u8>,
}

impl ImagenetVal {
  pub fn open(path: PathBuf) -> Result<ImagenetVal, ()> {
    let file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    Ok(ImagenetVal{
      path: path,
      mmap: mmap,
    })
  }

  pub fn test_images(&self) {
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    let mut jpeg_ct = 0;
    for entry in tar.raw_entries() {
      let entry = entry.unwrap();
      let buf = &self.mmap[entry.entry_pos as _ .. (entry.entry_pos + entry.entry_sz) as _];
      match guess_image_format_from_signature(buf) {
        Some(ImageFormat::Jpeg) => jpeg_ct += 1,
        fm => println!("DEBUG: found non-jpeg: {:?}", fm),
      }
    }
    // TODO
    //unimplemented!();
    println!("DEBUG: val set: jpegs: {}", jpeg_ct);
  }
}

pub struct ImagenetTrain {
  path: PathBuf,
  mmap: MemoryMap<u8>,
}

impl ImagenetTrain {
  pub fn open(path: PathBuf) -> Result<ImagenetTrain, ()> {
    let file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    Ok(ImagenetTrain{
      path: path,
      mmap: mmap,
    })
  }

  pub fn test_images(&self) {
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    let mut jpeg_ct = 0;
    for entry in tar.raw_entries() {
      let entry = entry.unwrap();
      let archive_buf = &self.mmap[entry.entry_pos as _ .. (entry.entry_pos + entry.entry_sz) as _];
      let archive_cursor = Cursor::new(archive_buf);
      let mut archive_tar = BufferedTarFile::new(archive_cursor);
      for im_entry in archive_tar.raw_entries() {
        let im_entry = im_entry.unwrap();
        let im_buf = &archive_buf[im_entry.entry_pos as _ .. (im_entry.entry_pos + im_entry.entry_sz) as _];
        match guess_image_format_from_signature(im_buf) {
          Some(ImageFormat::Jpeg) => jpeg_ct += 1,
          fm => println!("DEBUG: found non-jpeg: {:?}", fm),
        }
      }
    }
    // TODO
    //unimplemented!();
    println!("DEBUG: val set: jpegs: {}", jpeg_ct);
  }
}
