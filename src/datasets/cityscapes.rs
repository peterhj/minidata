use ::*;

use extar::*;
use sharedmem::*;

use std::collections::{BTreeSet, HashMap};
use std::collections::hash_map::{Entry};
use std::fs::{File};
use std::io::{BufRead, Read, BufReader, BufWriter, Cursor};
use std::path::{PathBuf};

enum CityscapesImageType {
  Color,
  LabelIds,
  InstanceIds,
  Polygons,
}

#[derive(Clone)]
pub struct CityscapesTarData {
  px_pairs:     Vec<(usize, usize)>,
  //tuple_idxs:   Vec<(usize, usize, usize)>,
  data_index:   Vec<(usize, usize)>,
  mem:          SharedMem<u8>,
}

impl CityscapesTarData {
  pub fn open(tar_path: PathBuf) -> Result<Self, ()> {
    let file = File::open(&tar_path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    let mem = SharedMem::new(mmap);
    let mut data = CityscapesTarData{
      px_pairs:     vec![],
      data_index:   vec![],
      mem:          mem,
    };
    data._build_index();
    Ok(data)
  }

  fn _build_index(&mut self) {
    self.px_pairs.clear();
    self.data_index.clear();
    let cursor = Cursor::new(&self.mem as &[u8]);
    //let mut keys = vec![];
    let mut keys = BTreeSet::new();
    let mut kv_color_idxs = HashMap::new();
    let mut kv_labelid_idxs = HashMap::new();
    let mut tar = BufferedTarFile::new(cursor);
    for entry in tar.raw_entries() {
      let entry = entry.unwrap();
      if !entry.is_file {
        continue;
      }
      let file_stem = entry.path.file_stem().unwrap();
      let file_stem_toks: Vec<_> = file_stem.to_str().unwrap().split("_").collect();
      let series_key = file_stem_toks[0].to_owned();
      let series_nr = u32::from_str_radix(file_stem_toks[1], 10).unwrap();
      let frame_nr = u32::from_str_radix(file_stem_toks[2], 10).unwrap();
      let image_ty = match file_stem_toks[4] {
        "color"         => CityscapesImageType::Color,
        "instanceIds"   => continue,
        "labelIds"      => CityscapesImageType::LabelIds,
        "polygons"      => continue,
        _ => unreachable!(),
      };
      let idx = self.data_index.len();
      let key: (String, u32, u32) = (series_key, series_nr, frame_nr);
      let datum: (usize, usize) = (entry.entry_pos as _, entry.entry_sz as _);
      keys.insert(key.clone());
      self.data_index.push(datum);
      match image_ty {
        CityscapesImageType::Color => {
          match kv_color_idxs.entry(key) {
            Entry::Occupied(_) => panic!(),
            Entry::Vacant(e) => {
              e.insert(idx);
            }
          }
        }
        CityscapesImageType::LabelIds => {
          match kv_labelid_idxs.entry(key) {
            Entry::Occupied(_) => panic!(),
            Entry::Vacant(e) => {
              e.insert(idx);
            }
          }
        }
        _ => unimplemented!(),
      }
    }
    println!("DEBUG: cityscapes: num frames: {}", keys.len());
    for key in keys.iter() {
      let img_idx = *kv_color_idxs.get(key).unwrap();
      let target_idx = *kv_labelid_idxs.get(key).unwrap();
      self.px_pairs.push((img_idx, target_idx));
    }
  }
}

impl RandomAccess for CityscapesTarData {
  type Item = (SharedMem<u8>, SharedMem<u8>);

  fn len(&self) -> usize {
    self.px_pairs.len()
  }

  fn at(&mut self, idx: usize) -> (SharedMem<u8>, SharedMem<u8>) {
    let (img_idx, target_idx) = self.px_pairs[idx];
    let (img_offset, img_size) = self.data_index[img_idx];
    let (target_offset, target_size) = self.data_index[target_idx];
    let img = self.mem.shared_slice(img_offset .. img_offset + img_size);
    let target = self.mem.shared_slice(target_offset .. target_offset + target_size);
    (img, target)
  }
}
