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

use arrayidx::*;
use byteorder::*;
use memarray::*;
use memarray::io::*;
use rand::prelude::*;
use sharedmem::*;

use std::cmp::{max, min};
use std::collections::{HashMap};
use std::fs::{File};
use std::io::{BufRead, Read, BufReader, BufWriter, Cursor};
use std::path::{PathBuf};

pub struct NpyMmapData<Idx, T> where T: Copy {
  mmap: SharedMem<u8>,
  arr:  MemArray<Idx, T, SharedMem<T>>,
}

impl<Idx, T> NpyMmapData<Idx, T> where Idx: ArrayIndex, T: ToNpyDtypeDesc + Copy {
  pub fn open(path: PathBuf) -> Result<NpyMmapData<Idx, T>, ()> {
    let file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    let mem = SharedMem::from(mmap);
    let header = {
      let mut reader = Cursor::new(&mem as &[u8]);
      match read_npy_header(&mut reader) {
        Err(_) => panic!(),
        Ok(header) => header,
      }
    };
    assert!(header.dtype_desc.matches::<T>());
    let raw_data = mem.shared_slice(header.data_offset .. );
    let data = raw_data.as_typed_slice();
    let size = <Idx as ArrayIndex>::from_nd(header.nd_size);
    let arr = MemArray::with_memory(size, data);
    Ok(NpyMmapData{
      mmap: mem,
      arr:  arr,
    })
  }
}

impl<Idx, T> RandomAccess for NpyMmapData<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Item = MemArray<Idx::Below, T, SharedMem<T>>;

  fn len(&self) -> usize {
    let size = self.arr.size();
    size.index_at((size.dim() - 1) as _)
  }

  fn at(&mut self, idx: usize) -> MemArray<Idx::Below, T, SharedMem<T>> {
    let arr_size = self.arr.size();
    let item_size = arr_size.index_cut((arr_size.dim() - 1) as _);
    if self.arr.is_packed() {
      let item_sz = item_size.flat_len();
      let item_data = self.arr.memory().shared_slice(idx * item_sz .. (idx + 1) * item_sz);
      MemArray::with_memory(item_size, item_data)
    } else {
      unimplemented!();
    }
  }
}
