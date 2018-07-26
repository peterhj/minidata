/*
Copyright 2018 Peter Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use ::*;

use byteorder::*;
use colorimage::*;
use extar::*;
#[cfg(feature = "mpi")] use mpich::*;
use rand::prelude::*;
use sharedmem::*;

use std::cmp::{max, min};
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

struct ScatterSplit {
  index_offset: usize,
  index_len:    usize,
  mem_dist_off: usize,
  mem_occ_sz:   usize,
}

#[cfg(feature = "mpi")]
pub struct ImagenetMPIRMAScatterData {
  cfg:      ImagenetConfig,
  index:    Vec<(usize, usize, u32)>,
  rank:     i32,
  nranks:   i32,
  block_sz: usize,
  nblocks:  usize,
  splits:   Vec<ScatterSplit>,
  mems:     Vec<MPIMem>,
  rma_wins: Vec<MPIRMAWin<u8>>,
}

#[cfg(feature = "mpi")]
impl ImagenetMPIRMAScatterData {
  fn new(cfg: ImagenetConfig, index: Vec<(usize, usize, u32)>, mmap: &SharedMem<u8>) -> Self {
    let nranks = MPIComm::world().num_ranks() as usize;
    let total_len = index.len();
    let max_shard_len = (total_len + nranks - 1) / nranks;
    let mut splits = Vec::with_capacity(nranks);
    let mut max_shard_mem_sz = 0;
    for r in 0 .. nranks {
      let start_idx = r * max_shard_len;
      let end_idx = min(total_len, (r + 1) * max_shard_len);
      let shard_mem_off = index[start_idx].0;
      let req_shard_mem_sz = index[end_idx - 1].0 + index[end_idx - 1].1 - index[start_idx].0;
      let split = ScatterSplit{
        index_offset:   start_idx,
        index_len:      end_idx - start_idx,
        mem_dist_off:   shard_mem_off,
        mem_occ_sz:     req_shard_mem_sz,
      };
      splits.push(split);
      max_shard_mem_sz = max(max_shard_mem_sz, req_shard_mem_sz);
    }

    let block_sz = 2 * 1024 * 1024;
    let rdup_shard_mem_sz = (max_shard_mem_sz + block_sz - 1) / block_sz * block_sz;
    let num_blocks = rdup_shard_mem_sz / block_sz;
    assert_eq!(0, rdup_shard_mem_sz % block_sz);
    println!("DEBUG: ImagenetMPIRMAScatterData: block sz: {} num blocks: {} rdup shard sz: {}",
        block_sz,
        num_blocks,
        rdup_shard_mem_sz);

    let mut mems = Vec::with_capacity(num_blocks);
    let mut rma_wins = Vec::with_capacity(num_blocks);
    for _ in 0 .. num_blocks {
      let mem = unsafe { MPIMem::alloc(block_sz) };
      let rma_win = match unsafe { MPIRMAWin::new(mem.as_mut_ptr(), mem.size_bytes(), &mut MPIComm::world()) } {
        Err(e) => panic!("failed to create rma win: {:?}", e),
        Ok(win) => win,
      };
      mems.push(mem);
      rma_wins.push(rma_win);
    }

    let rank = MPIComm::world().rank() as usize;
    let mut zero_block_buf = Vec::with_capacity(block_sz);
    for k in 0 .. block_sz {
      zero_block_buf[k] = 0;
    }
    for blk in 0 .. num_blocks {
      let local_start_pos = blk * block_sz;
      let local_end_pos = min(splits[rank].mem_occ_sz, (blk + 1) * block_sz);
      let rma_win = rma_wins[blk].lock();
      if local_start_pos < local_end_pos {
        let shared_start_pos = splits[rank].mem_dist_off + local_start_pos;
        let shared_end_pos = splits[rank].mem_dist_off + local_end_pos;
        let shared_buf = mmap.shared_slice(shared_start_pos .. shared_end_pos);
        rma_win.put_mem(
            &*shared_buf,
            rank as _,
            0,
            local_end_pos - local_start_pos);
      }
      if local_end_pos < local_start_pos + block_sz {
        let end_pos = max(local_start_pos, local_end_pos) - local_start_pos;
        rma_win.put_mem(
            &zero_block_buf[end_pos .. ],
            rank as _,
            end_pos,
            block_sz - end_pos);
      }
    }

    ImagenetMPIRMAScatterData{
      cfg:      cfg,
      index:    index,
      rank:     MPIComm::world().rank(),
      nranks:   MPIComm::world().num_ranks(),
      block_sz: block_sz,
      nblocks:  num_blocks,
      splits:   splits,
      mems:     mems,
      rma_wins: rma_wins,
    }
  }
}

#[cfg(feature = "mpi")]
impl RandomAccess for ImagenetMPIRMAScatterData {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.index.len()
  }

  fn at(&mut self, idx: usize) -> (SharedMem<u8>, u32) {
    let (offset, size, label) = self.index[idx];
    let mut value = Vec::with_capacity(size);
    unsafe { value.set_len(size) };
    for (rank, split) in self.splits.iter().enumerate() {
      if split.index_offset <= idx && idx < split.index_offset + split.index_len {
        let local_offset = offset - split.mem_dist_off;
        let local_end_pos = local_offset + size;
        let local_start_blk = local_offset / self.block_sz;
        let local_end_blk = (local_end_pos - 1) / self.block_sz + 1;
        assert!(local_offset + size <= split.mem_occ_sz);
        assert!(local_start_blk < self.nblocks);
        assert!(local_start_blk < local_end_blk);
        assert!(local_end_blk <= self.nblocks);
        let mut val_offset = 0;
        let mut tmp_offset = local_offset;
        for blk in local_start_blk .. local_end_blk {
          let blk_start_pos = blk * self.block_sz;
          let blk_end_pos = min(split.mem_occ_sz, (blk + 1) * self.block_sz);
          let blk_data_start = tmp_offset - blk_start_pos;
          let blk_data_end = min(local_end_pos, blk_end_pos) - blk_start_pos;
          let blk_data_len = blk_data_end - blk_data_start;
          assert!(blk_data_len > 0);
          assert!(blk_data_len <= self.block_sz);
          {
            let rma_win = self.rma_wins[blk].lock_shared();
            rma_win.get_mem(
                &mut value[val_offset .. val_offset + blk_data_len],
                rank as _,
                blk_data_start,
                blk_data_len);
          }
          val_offset += blk_data_len;
          tmp_offset += blk_data_len;
        }
        assert_eq!(val_offset, size);
        return (SharedMem::from(value), label);
      }
    }
    unreachable!();
  }
}

/*#[cfg(feature = "shmem")]
pub struct ImagenetShmemShardData {
  cfg:      ImagenetConfig,
  index:    Vec<(usize, usize, u32)>,
  rank:     i32,
  nranks:   i32,
  splits:   Vec<(usize, usize)>,
  mem:      ShmemHeapMem,
}

#[cfg(feature = "shmem")]
impl ImagenetShmemShardData {
  fn new(cfg: ImagenetConfig, index: Vec<(usize, usize, u32)>, mmap: &SharedMem<u8>) -> Self {
    let nranks = Shmem::num_ranks();
    let total_len = index.len();
    let max_shard_len = (total_len + nranks as usize - 1) / nranks as usize;
    let mut splits: Vec<(usize, usize)> = Vec::with_capacity(nranks as usize);
    let mut max_shard_mem_sz = 0;
    for r in 0 .. nranks as usize {
      let start_idx = r * max_shard_len;
      let end_idx = min(total_len, (r + 1) * max_shard_len);
      let req_shard_mem_sz = index[end_idx - 1].0 + index[end_idx - 1].1 - index[start_idx].0;
      splits.push((start_idx, end_idx - start_idx));
      max_shard_mem_sz = max(max_shard_mem_sz, req_shard_mem_sz);
    }
    let block_sz = 1024 * 1024;
    let rdup_shard_mem_sz = (max_shard_mem_sz + block_sz - 1) / block_sz * block_sz;
    let mut mem = unsafe { ShmemHeapMem::alloc(rdup_shard_mem_sz) };
    let num_blocks = rdup_shard_mem_sz / block_sz;
    assert_eq!(0, rdup_shard_mem_sz % block_sz);
    // TODO: copy memory from the mmap to the symmetric heap.
    let priv_rank = Shmem::rank();
    let priv_start_idx = splits[priv_rank as usize].0;
    let priv_end_idx = splits[priv_rank as usize].0 + splits[priv_rank as usize].1;
    let priv_mmap_pos = index[priv_start_idx].0;
    let priv_mmap_sz = index[priv_end_idx - 1].0 + index[priv_end_idx - 1].1 - priv_mmap_pos;
    for blk in 0 .. num_blocks {
      let start_pos = blk * block_sz;
      let end_pos = min(priv_mmap_sz, (blk + 1) * block_sz);
      mem.put_mem(priv_rank, start_pos, &*mmap.shared_slice(priv_mmap_pos + start_pos .. priv_mmap_pos + end_pos));
    }
    ImagenetShmemShardData{
      cfg:      cfg,
      index:    index,
      rank:     priv_rank,
      nranks:   nranks,
      splits:   splits,
      mem:      mem,
    }
  }

  fn _find_split(&self, idx: usize) -> (i32, usize, usize) {
    // TODO
    for (rank, &(offset, len)) in self.splits.iter().enumerate() {
      if idx >= offset && idx < offset + len {
        return (rank as _, offset, idx - offset);
      }
    }
    unreachable!();
  }

  fn _get(&mut self, idx: usize) -> (Vec<u8>, u32) {
    let (src_rank, src_index_offset, _) = self._find_split(idx);
    assert!(src_rank >= 0);
    assert!(src_rank < self.nranks);
    let src_entry = self.index[idx];
    let src_mem_offset = src_entry.0 - self.index[src_index_offset].0;
    let src_mem_size = src_entry.1;
    let label = src_entry.2;
    let mut value = Vec::with_capacity(src_mem_size);
    self.mem.get_mem(src_rank, src_mem_offset, &mut value);
    (value, label)
  }
}

#[cfg(feature = "shmem")]
impl RandomAccess for ImagenetShmemShardData {
  type Item = (Vec<u8>, u32);

  fn len(&self) -> usize {
    self.index.len()
  }

  fn at(&mut self, idx: usize) -> (Vec<u8>, u32) {
    self._get(idx)
  }
}*/

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
      mmap:     SharedMem::from(mmap),
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
        mmap:   SharedMem::from(mmap),
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
  pub fn scatter_mpi_rma(self) -> ImagenetMPIRMAScatterData {
    ImagenetMPIRMAScatterData::new(self.cfg, self.index, &self.mmap)
  }
}

impl RandomAccess for ImagenetValData {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.index.len()
  }

  fn at(&mut self, idx: usize) -> (SharedMem<u8>, u32) {
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
      mmap:     SharedMem::from(mmap),
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
        mmap:   SharedMem::from(mmap),
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
  pub fn scatter_mpi_rma(self) -> ImagenetMPIRMAScatterData {
    ImagenetMPIRMAScatterData::new(self.cfg, self.index, &self.mmap)
  }
}

impl RandomAccess for ImagenetTrainData {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.index.len()
  }

  fn at(&mut self, idx: usize) -> (SharedMem<u8>, u32) {
    let (offset, size, label) = self.index[idx];
    let value = self.mmap.shared_slice(offset .. offset + size);
    (value, label)
  }
}
