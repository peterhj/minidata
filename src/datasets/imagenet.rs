use ::*;

use byteorder::*;
use colorimage::*;
use extar::*;
#[cfg(feature = "mpi")] use mpich::*;
use rand::*;
use sharedmem::*;

use std::cmp::{min};
use std::collections::{HashMap};
use std::fs::{File};
use std::io::{BufRead, Read, BufReader, BufWriter, Cursor};
use std::path::{PathBuf};

pub fn load_tar_index(path: PathBuf) -> Result<Vec<(usize, usize, u32)>, ()> {
  let file = File::open(path).unwrap();
  let file_len = file.metadata().unwrap().len() as usize;
  assert_eq!(0, file_len % 20);
  let num_entries = file_len / 20;
  let mut index = Vec::with_capacity(num_entries);
  let mut reader = BufReader::new(file);
  for _ in 0 .. num_entries {
    let offset = reader.read_u64::<LittleEndian>().unwrap() as usize;
    let size = reader.read_u64::<LittleEndian>().unwrap() as usize;
    let label = reader.read_u32::<LittleEndian>().unwrap();
    index.push((offset, size, label));
  }
  Ok(index)
}

pub fn save_tar_index(index: &[(usize, usize, u32)], path: PathBuf) -> Result<(), ()> {
  let file = File::create(path).unwrap();
  let mut writer = BufWriter::new(file);
  for entry in index {
    writer.write_u64::<LittleEndian>(entry.0 as u64).unwrap();
    writer.write_u64::<LittleEndian>(entry.1 as u64).unwrap();
    writer.write_u32::<LittleEndian>(entry.2).unwrap();
  }
  let _ = writer.into_inner().unwrap();
  Ok(())
}

#[cfg(feature = "mpi")]
pub struct ImagenetShardMPIData {
  cfg:      ImagenetConfig,
  mmap:     SharedMem<u8>,
  index:    Vec<(usize, usize, u32)>,
  //window:   MPIWindow<u8>,
}

#[cfg(feature = "mpi")]
impl RandomAccess for ImagenetShardMPIData {
  type Item = (SharedMem<u8>, u32);
  //type Item = (Vec<u8>, u32);

  fn len(&self) -> usize {
    self.index.len()
  }

  fn at(&self, idx: usize) -> (SharedMem<u8>, u32) {
    // TODO
    let (offset, size, label) = self.index[idx];
    let value = self.mmap.shared_slice(offset .. offset + size);
    (value, label)
  }
}

#[derive(Clone, Default, Debug)]
pub struct ImagenetConfig {
  pub train_data:       Option<PathBuf>,
  pub val_data:         Option<PathBuf>,
  pub wordnet_ids:      Option<PathBuf>,
  pub val_ground_truth: Option<PathBuf>,
}

#[derive(Clone)]
pub struct ImagenetValData {
  cfg:      ImagenetConfig,
  //labels:   Vec<u32>,
  mmap:     SharedMem<u8>,
  index:    Vec<(usize, usize, u32)>,
}

impl ImagenetValData {
  pub fn open(cfg: ImagenetConfig) -> Result<ImagenetValData, ()> {
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
    let mut data = ImagenetValData{
      cfg:      cfg,
      //labels:   labels,
      mmap:     SharedMem::new(mmap),
      index:    vec![],
    };
    data._build_index(&labels);
    Ok(data)
  }

  fn _build_index(&mut self, labels: &[u32]) {
    self.index.clear();
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    for (idx, im_entry) in tar.raw_entries().enumerate() {
      let im_entry = im_entry.unwrap();
      let offset = im_entry.entry_pos as _;
      let size = im_entry.entry_sz as _;
      let label = labels[idx];
      self.index.push((offset, size, label));
    }
  }

  pub fn load_index(cfg: ImagenetConfig) -> Result<Self, ()> {
    let val_data_path = cfg.val_data.clone().unwrap();
    assert!("tar" == val_data_path.extension().unwrap());
    let mut val_index_path = val_data_path.clone();
    assert!(val_index_path.set_extension("tar_index"));
    load_tar_index(val_index_path).and_then(|index| {
      let file = File::open(cfg.val_data.as_ref().unwrap()).unwrap();
      let file_len = file.metadata().unwrap().len() as usize;
      let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
      Ok(ImagenetValData{
        cfg:    cfg,
        mmap:   SharedMem::new(mmap),
        index:  index,
      })
    })
  }

  pub fn save_index(&self) -> Result<(), ()> {
    let val_data_path = self.cfg.val_data.clone().unwrap();
    assert!("tar" == val_data_path.extension().unwrap());
    let mut val_index_path = val_data_path.clone();
    assert!(val_index_path.set_extension("tar_index"));
    save_tar_index(&self.index, val_index_path)
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

  #[cfg(feature = "mpi")]
  pub fn shard_mpi(self) -> ImagenetShardMPIData {
    // TODO
    unimplemented!();
    /*let window = unsafe { MPIWindow::new(self.mmap.as_ptr() as *mut u8, self.mmap.len(), &mut MPIComm::world()).unwrap() };
    ImagenetShardMPIData{
      cfg:      self.cfg,
      mmap:     self.mmap,
      index:    self.index,
      window,
    }*/
  }
}

impl RandomAccess for ImagenetValData {
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
pub struct ImagenetTrainData {
  cfg:      ImagenetConfig,
  //labels:   HashMap<String, u32>,
  mmap:     SharedMem<u8>,
  index:    Vec<(usize, usize, u32)>,
}

impl ImagenetTrainData {
  pub fn open(cfg: ImagenetConfig) -> Result<ImagenetTrainData, ()> {
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
    let mut data = ImagenetTrainData{
      cfg:      cfg,
      //labels:   labels,
      mmap:     SharedMem::new(mmap),
      index:    vec![],
    };
    data._build_index(&labels);
    Ok(data)
  }

  fn _build_index(&mut self, labels: &HashMap<String, u32>) {
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
        let label = *labels.get(&im_wnid).unwrap();
        self.index.push((offset, size, label));
      }
    }
  }

  pub fn load_index(cfg: ImagenetConfig) -> Result<Self, ()> {
    let train_data_path = cfg.train_data.clone().unwrap();
    assert!("tar" == train_data_path.extension().unwrap());
    let mut train_index_path = train_data_path.clone();
    assert!(train_index_path.set_extension("tar_index"));
    load_tar_index(train_index_path).and_then(|index| {
      let file = File::open(cfg.train_data.as_ref().unwrap()).unwrap();
      let file_len = file.metadata().unwrap().len() as usize;
      let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
      Ok(ImagenetTrainData{
        cfg:    cfg,
        mmap:   SharedMem::new(mmap),
        index:  index,
      })
    })
  }

  pub fn save_index(&self) -> Result<(), ()> {
    let train_data_path = self.cfg.train_data.clone().unwrap();
    assert!("tar" == train_data_path.extension().unwrap());
    let mut train_index_path = train_data_path.clone();
    assert!(train_index_path.set_extension("tar_index"));
    save_tar_index(&self.index, train_index_path)
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

  #[cfg(feature = "mpi")]
  pub fn shard_mpi(self) -> ImagenetShardMPIData {
    // TODO
    unimplemented!();
    /*let rank = MPIComm::world().rank() as usize;
    let nranks = MPIComm::world().num_ranks() as usize;
    let rdup_shard_len = (self.len() + rank - 1) / nranks;
    let shard_off = rank * rdup_shard_len;
    let shard_len = min(self.len(), (rank + 1) * rdup_shard_len) - shard_off;
    let data_start = self.index[shard_off].0;
    let data_end = self.index[shard_off + shard_len - 1].0 + self.index[shard_off + shard_len - 1].1;
    assert!(data_start <= data_end);
    let data_len = data_end - data_start;
    /*let file = File::open(self.cfg.train_data.as_ref().unwrap()).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, data_start, data_len).unwrap();*/
    //let window = unsafe { MPIWindow::new(self.mmap.as_ptr() as *mut u8, self.mmap.len(), &mut MPIComm::world()).unwrap() };
    let window = unsafe { MPIWindow::new(
        self.mmap.as_ptr().offset(data_start as _) as *mut u8,
        data_len,
        &mut MPIComm::world()).unwrap() };
    ImagenetShardMPIData{
      cfg:      self.cfg,
      mmap:     self.mmap,
      index:    self.index,
      window,
    }*/
  }
}

impl RandomAccess for ImagenetTrainData {
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
