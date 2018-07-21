use ::*;

use lmdb;
use lmdb::{Cursor, Transaction};
use sharedmem::*;

use std::collections::{HashMap};
use std::ops::{Deref};
use std::path::{PathBuf};
use std::sync::{Arc};

struct LmdbState {
  env:  lmdb::Environment,
  db:   lmdb::Database,
}

impl Deref for LmdbState {
  type Target = [u8];

  fn deref(&self) -> &[u8] {
    unreachable!();
  }
}

pub struct LmdbData {
  state:    Arc<LmdbState>,
  num:      usize,
  keys:     Vec<Vec<u8>>,
  //keys_rev: HashMap<Vec<u8>, usize>,
}

impl LmdbData {
  pub fn open(path: PathBuf) -> Result<Self, ()> {
    let env = lmdb::Environment::new()
      .set_flags(
          lmdb::EnvironmentFlags::READ_ONLY |
          lmdb::EnvironmentFlags::NO_TLS |
          lmdb::EnvironmentFlags::NO_LOCK)
      .open(&path)
      .map_err(|_| ())
      ?;
    let db = env.open_db(None)
      .map_err(|_| ())
      ?;
    let state = Arc::new(LmdbState{env, db});
    let mut data = LmdbData{
      state:    state,
      num:  0,
      keys: vec![],
      //keys_rev: HashMap::new(),
    };
    data._fill_keys();
    Ok(data)
  }

  fn _fill_keys(&mut self) {
    self.keys.clear();
    //self.keys_rev.clear();
    let rtxn = self.state.env.begin_ro_txn().unwrap();
    let mut cursor = rtxn.open_ro_cursor(self.state.db).unwrap();
    for (idx, kv) in cursor.iter_start().enumerate() {
      assert_eq!(idx, self.keys.len());
      //assert_eq!(idx, self.keys_rev.len());
      let key = kv.0.to_owned();
      self.keys.push(key.clone());
      //self.keys_rev.insert(key, idx);
    }
    //assert_eq!(self.keys.len(), self.keys_rev.len());
    self.num = self.keys.len();
  }
}

impl RandomAccess for LmdbData {
  //type Item = (Vec<u8>, Vec<u8>);
  type Item = (SharedMem<u8>, SharedMem<u8>);

  fn len(&self) -> usize {
    self.num
  }

  //fn at(&mut self, idx: usize) -> (Vec<u8>, Vec<u8>) {
  fn at(&mut self, idx: usize) -> (SharedMem<u8>, SharedMem<u8>) {
    let key = &self.keys[idx];
    let rtxn = self.state.env.begin_ro_txn().unwrap();
    let mut cursor = rtxn.open_ro_cursor(self.state.db).unwrap();
    match cursor.iter_from(key).next() {
      None => panic!(),
      Some(kv) => {
        //(key.clone(), kv.1.to_owned())
        unsafe { (SharedMem::from_raw(kv.0.as_ptr(), kv.0.len(), self.state.clone()),
                  SharedMem::from_raw(kv.1.as_ptr(), kv.1.len(), self.state.clone())) }
      }
    }
  }
}
