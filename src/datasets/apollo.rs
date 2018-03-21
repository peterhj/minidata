use ::*;
//use colorimage::*;
use extar::*;
use sharedmem::*;
use string_cache::DefaultAtom as Atom;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{File};
use std::io::{BufRead, Read, BufReader, Cursor};
use std::path::{PathBuf};

#[derive(Clone, Default, Debug)]
pub struct ApolloConfig {
  pub scene_data:       Option<PathBuf>,
  pub train_scene_ids:  Option<Vec<String>>,
  pub val_scene_ids:    Option<Vec<String>>,
}

#[derive(Clone)]
pub struct ApolloSceneCollection {
  pub scenes:   Vec<ApolloSceneData>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ApolloSceneFrameKey {
  pub date: u64,
  pub time: u64,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ApolloSceneCameraFrameInfo {
  pub image_pos:    usize,
  pub image_sz:     usize,
  pub s_label_pos:  usize,
  pub s_label_sz:   usize,
  pub i_label_pos:  usize,
  pub i_label_sz:   usize,
}

#[derive(Clone, Debug)]
pub struct ApolloSceneFrameInfo {
  pub record:   Atom,
  pub camera5:  ApolloSceneCameraFrameInfo,
  pub camera6:  ApolloSceneCameraFrameInfo,
}

impl ApolloSceneFrameInfo {
  pub fn default_with_record(record: Atom) -> Self {
    ApolloSceneFrameInfo{
      record:   record,
      camera5:  Default::default(),
      camera6:  Default::default(),
    }
  }
}

enum FrameTy {
  Image,
  Label,
  Pose,
}

enum FrameCam {
  Camera5,
  Camera6,
}

enum LabelTy {
  Semantic,
  Instance,
}

#[derive(Clone)]
pub struct ApolloSceneData {
  cfg:      ApolloConfig,
  scene_id: String,
  mmap:     SharedMem<u8>,
  // TODO
  all_keys: Vec<(ApolloSceneFrameKey, ApolloSceneFrameInfo)>,
  infos:    HashMap<ApolloSceneFrameKey, ApolloSceneFrameInfo>,
  //keys_to_records:  BTreeMap<ApolloSceneFrameKey, Atom>,
  //records_to_keys:  HashMap<Atom, BTreeSet<ApolloSceneFrameKey>>,
}

impl ApolloSceneData {
  pub fn open(cfg: ApolloConfig, scene_id: String) -> Result<ApolloSceneData, ()> {
    let data_root = cfg.scene_data.as_ref().unwrap().clone();
    let scene_path = data_root.join(format!("{}_ins.tar", scene_id));
    let file = File::open(&scene_path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    let mmap = MemoryMap::open_with_offset(file, 0, file_len).unwrap();
    let mut data = ApolloSceneData{
      cfg:      cfg,
      scene_id: scene_id,
      mmap:     SharedMem::new(mmap),
      all_keys: Vec::new(),
      infos:    HashMap::new(),
      //keys_to_records:  BTreeMap::new(),
      //records_to_keys:  HashMap::new(),
    };
    data._inspect_debug();
    data._build_index();
    Ok(data)
  }

  pub fn _build_index(&mut self) {
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    self.infos.clear();
    for entry in tar.raw_entries() {
      let entry = entry.unwrap();
      if !entry.is_file {
        continue;
      }
      let path_toks: Vec<_> = entry.path.components().collect();
      if path_toks.len() != 5 {
        continue;
      }
      let filename_str = path_toks[4].as_os_str().to_str().unwrap();
      if filename_str == "pose.txt" {
        continue;
      } else if filename_str.contains("json") {
        continue;
      }
      let data_ty_str = path_toks[1].as_os_str().to_str().unwrap();
      let data_ty = match data_ty_str {
        "ColorImage"    => FrameTy::Image,
        "Label"         => FrameTy::Label,
        "Pose"          => FrameTy::Pose,
        _ => unimplemented!(),
      };
      let record = Atom::from(path_toks[2].as_os_str().to_str().unwrap());
      let camera_str = path_toks[3].as_os_str().to_str().unwrap();
      let camera = match camera_str {
        "Camera 5" => FrameCam::Camera5,
        "Camera 6" => FrameCam::Camera6,
        _ => unimplemented!(),
      };
      let label_ty = if filename_str.contains("_bin") {
        Some(LabelTy::Semantic)
      } else if filename_str.contains("_instanceIds") {
        Some(LabelTy::Instance)
      } else {
        None
      };
      let filename_toks: Vec<_> = filename_str.split('_').collect();
      let date: u64 = match filename_toks[0].parse() {
        Ok(x) => x,
        Err(e) => panic!("fatal: {} e: {:?}", filename_toks[0], e),
      };
      let time: u64 = match filename_toks[1].parse() {
        Ok(x) => x,
        Err(e) => panic!("fatal: {:?}", e),
      };
      let key = ApolloSceneFrameKey{
        date:   date,
        time:   time,
      };
      if !self.infos.contains_key(&key) {
        let info = ApolloSceneFrameInfo::default_with_record(record);
        self.infos.insert(key, info);
      }
      let entry_pos = entry.entry_pos as _;
      let entry_sz = entry.entry_sz as _;
      match (camera, data_ty, label_ty) {
        (FrameCam::Camera5, FrameTy::Image, None) => {
          self.infos.get_mut(&key).unwrap().camera5.image_pos = entry_pos;
          self.infos.get_mut(&key).unwrap().camera5.image_sz = entry_sz;
        }
        (FrameCam::Camera5, FrameTy::Image, _) => unreachable!(),
        (FrameCam::Camera5, FrameTy::Label, Some(LabelTy::Semantic)) |
        (FrameCam::Camera5, FrameTy::Label, None) => {
          self.infos.get_mut(&key).unwrap().camera5.s_label_pos = entry_pos;
          self.infos.get_mut(&key).unwrap().camera5.s_label_sz = entry_sz;
        }
        (FrameCam::Camera5, FrameTy::Label, Some(LabelTy::Instance)) => {
          self.infos.get_mut(&key).unwrap().camera5.i_label_pos = entry_pos;
          self.infos.get_mut(&key).unwrap().camera5.i_label_sz = entry_sz;
        }
        (FrameCam::Camera6, FrameTy::Image, None) => {
          self.infos.get_mut(&key).unwrap().camera6.image_pos = entry_pos;
          self.infos.get_mut(&key).unwrap().camera6.image_sz = entry_sz;
        }
        (FrameCam::Camera6, FrameTy::Image, _) => unreachable!(),
        (FrameCam::Camera6, FrameTy::Label, Some(LabelTy::Semantic)) |
        (FrameCam::Camera6, FrameTy::Label, None) => {
          //println!("DEBUG: semantic label path: {:?}", entry.path);
          self.infos.get_mut(&key).unwrap().camera6.s_label_pos = entry_pos;
          self.infos.get_mut(&key).unwrap().camera6.s_label_sz = entry_sz;
        }
        (FrameCam::Camera6, FrameTy::Label, Some(LabelTy::Instance)) => {
          //println!("DEBUG: instance label path: {:?}", entry.path);
          self.infos.get_mut(&key).unwrap().camera6.i_label_pos = entry_pos;
          self.infos.get_mut(&key).unwrap().camera6.i_label_sz = entry_sz;
        }
        _ => {}
      }
    }
    self.all_keys.clear();
    {
      let &mut Self{ref mut all_keys, ref infos, ..} = self;
      for (key, info) in infos.iter() {
        all_keys.push((key.clone(), info.clone()));
      }
    }
    self.all_keys.sort_by(|a, b| a.0.cmp(&b.0));
  }

  fn _inspect_debug(&self) {
    let cursor = Cursor::new(&self.mmap as &[u8]);
    let mut tar = BufferedTarFile::new(cursor);
    let mut jpeg_count = 0;
    for (idx, entry) in tar.raw_entries().enumerate() {
      let entry = entry.unwrap();
      //let offset = entry.entry_pos as _;
      //let size = entry.entry_sz as _;
      //let label = self.labels[idx];
      //self.index.push((offset, size, label));
      if entry.path.as_os_str().to_str().unwrap().contains(".jpg") {
        jpeg_count += 1;
      } else if entry.path.as_os_str().to_str().unwrap().contains(".jpeg") {
        jpeg_count += 1;
      } else if entry.path.as_os_str().to_str().unwrap().contains(".JPG") {
        jpeg_count += 1;
      } else if entry.path.as_os_str().to_str().unwrap().contains(".JPEG") {
        jpeg_count += 1;
      }
      //println!("DEBUG: tar entry: idx: {} key: {:?}", idx, entry.file_path);
    }
    println!("DEBUG: num jpegs: {}", jpeg_count);
  }
}
