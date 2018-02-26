extern crate colorimage;
extern crate extar;
extern crate rand;
extern crate sharedmem;

use rand::*;
use std::sync::mpsc::*;

pub mod datasets;
pub mod imageproc;

pub trait RandomAccess {
  type Item;

  fn len(&self) -> usize;
  fn at(&self, idx: usize) -> Self::Item;
}

pub trait RandomAccessExt: RandomAccess + Sized {
  fn one_pass(self) -> OnePassDataSrc<Self>;
  fn uniform_shuffle(self) -> UniformShuffleDataSrc<Self>;
  fn uniform_random(self) -> UniformShuffleDataSrc<Self>;
}

pub trait DataIter: Iterator {
  fn reseed(&mut self, seed_rng: &mut Rng);
  fn reset(&mut self);
}

pub trait DataIterExt: DataIter + Sized {
  fn map<F, V>(self, f: F) -> MapDataIter<Self, F, V> where F: Fn(<Self as Iterator>::Item) -> V {
    MapDataIter::new(self, f)
  }

  fn round_up_repeat(self, rdup_sz: usize) -> RoundUpRepeatDataIter<Self> where <Self as Iterator>::Item: Clone {
    RoundUpRepeatDataIter::new(self, rdup_sz)
  }
}

pub struct OnePassDataSrc<R> where R: RandomAccess {
  data:     R,
  count:    usize,
}

impl<R> Iterator for OnePassDataSrc<R> where R: RandomAccess {
  type Item = <R as RandomAccess>::Item;

  fn next(&mut self) -> Option<Self::Item> {
    if self.count < self.data.len() {
      let item = self.data.at(self.count);
      self.count += 1;
      Some(item)
    } else {
      None
    }
  }
}

impl<R> DataIter for OnePassDataSrc<R> where R: RandomAccess {
  fn reseed(&mut self, seed_rng: &mut Rng) {
  }

  fn reset(&mut self) {
    self.count = 0;
  }
}

pub struct UniformShuffleDataSrc<R> where R: RandomAccess {
  data:     R,
  //rng:      _,
  count:    usize,
}

pub struct UniformRandomDataSrc<R> where R: RandomAccess {
  data:     R,
  //rng:      _,
}

pub struct MapDataIter<I, F, V> where I: DataIter, F: Fn(<I as Iterator>::Item) -> V {
  iter: I,
  mapf: F,
}

impl<I, F, V> MapDataIter<I, F, V> where I: DataIter, F: Fn(<I as Iterator>::Item) -> V {
  pub fn new(iter: I, mapf: F) -> Self {
    MapDataIter{
      iter: iter,
      mapf: mapf,
    }
  }
}

impl<I, F, V> Iterator for MapDataIter<I, F, V> where I: DataIter, F: Fn(<I as Iterator>::Item) -> V {
  type Item = V;

  fn next(&mut self) -> Option<Self::Item> {
    match self.iter.next() {
      None => None,
      Some(x) => Some((self.mapf)(x))
    }
  }
}

impl<I, F, V> DataIter for MapDataIter<I, F, V> where I: DataIter, F: Fn(<I as Iterator>::Item) -> V {
  fn reseed(&mut self, seed_rng: &mut Rng) {
    self.iter.reseed(seed_rng);
  }

  fn reset(&mut self) {
    self.iter.reset();
  }
}

pub struct RoundUpRepeatDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  rdup_sz:  usize,
  iter:     I,
  closed:   bool,
  rep_ct:   usize,
  rep_item: Option<<I as Iterator>::Item>,
}

impl<I> RoundUpRepeatDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  pub fn new(iter: I, rdup_sz: usize) -> Self {
    RoundUpRepeatDataIter{
      rdup_sz:  rdup_sz,
      iter:     iter,
      closed:   false,
      rep_ct:   0,
      rep_item: None,
    }
  }
}

impl<I> Iterator for RoundUpRepeatDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  type Item = (<I as Iterator>::Item, bool);

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    match self.iter.next() {
      None => {
        if self.rep_ct > 0 {
          self.rep_ct += 1;
          if self.rep_ct == self.rdup_sz {
            self.rep_ct = 0;
          }
          self.rep_item.clone().map(|item| (item, true))
        } else {
          self.closed = true;
          self.rep_item = None;
          None
        }
      }
      Some(item) => {
        self.rep_ct += 1;
        if self.rep_ct == self.rdup_sz {
          self.rep_ct = 0;
        }
        self.rep_item = Some(item.clone());
        Some((item, false))
      }
    }
  }
}

impl<I> DataIter for RoundUpRepeatDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  fn reseed(&mut self, seed_rng: &mut Rng) {
    self.iter.reseed(seed_rng);
  }

  fn reset(&mut self) {
    self.iter.reset();
    self.closed = false;
    self.rep_ct = 0;
    self.rep_item = None;
  }
}

pub struct ParallelJoinDataIter {
}
