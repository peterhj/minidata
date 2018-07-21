use ::*;

use lmdb;
use lmdb::{Cursor, Transaction};
//use sharedmem::*;

use std::collections::{HashMap};
use std::path::{PathBuf};

pub struct LmdbData {
  env:      lmdb::Environment,
  db:       lmdb::Database,
  num:      usize,
  keys:     Vec<Vec<u8>>,
  keys_rev: HashMap<Vec<u8>, usize>,
}

impl LmdbData {
  pub fn open(path: PathBuf) -> Self {
    let env = lmdb::Environment::new()
      .set_flags(
          lmdb::EnvironmentFlags::READ_ONLY |
          lmdb::EnvironmentFlags::NO_TLS |
          lmdb::EnvironmentFlags::NO_LOCK)
      .open(&path)
      .unwrap();
    let db = env.open_db(None).unwrap();
    let mut data = LmdbData{
      env:  env,
      db:   db,
      num:  0,
      keys: vec![],
      keys_rev: HashMap::new(),
    };
    data._fill_keys();
    data
  }

  fn _fill_keys(&mut self) {
    self.keys.clear();
    self.keys_rev.clear();
    let rtxn = self.env.begin_ro_txn().unwrap();
    let mut cursor = rtxn.open_ro_cursor(self.db).unwrap();
    for (idx, kv) in cursor.iter_start().enumerate() {
      assert_eq!(idx, self.keys.len());
      assert_eq!(idx, self.keys_rev.len());
      let key = kv.0.to_owned();
      self.keys.push(key.clone());
      self.keys_rev.insert(key, idx);
    }
    assert_eq!(self.keys.len(), self.keys_rev.len());
    self.num = self.keys.len();
  }
}

impl RandomAccess for LmdbData {
  //type Item = (SharedMem<u8>, SharedMem<u8>);
  type Item = (Vec<u8>, Vec<u8>);

  fn len(&self) -> usize {
    self.num
  }

  fn at(&mut self, idx: usize) -> (Vec<u8>, Vec<u8>) {
    let key = self.keys[idx].clone();
    let rtxn = self.env.begin_ro_txn().unwrap();
    let mut cursor = rtxn.open_ro_cursor(self.db).unwrap();
    match cursor.iter_from(&key).next() {
      None => panic!(),
      Some(kv) => {
        assert_eq!(&key as &[u8], kv.0);
        (key, kv.1.to_owned())
      }
    }
  }
}
