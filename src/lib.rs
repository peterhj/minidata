extern crate colorimage;
extern crate extar;
extern crate rand;
extern crate sharedmem;

use rand::*;
use std::sync::mpsc::*;

pub mod datasets;
pub mod image;

pub trait RandomAccess {
  type Item;

  fn len(&self) -> usize;
  fn at(&self, idx: usize) -> Self::Item;

  fn one_pass(self) -> OnePassDataSrc<Self> where Self: Sized {
    OnePassDataSrc::new(self)
  }

  fn uniform_shuffle(self) -> UniformShuffleDataSrc<Self> where Self: Sized {
    // TODO
    unimplemented!();
  }

  fn uniform_random(self) -> UniformShuffleDataSrc<Self> where Self: Sized {
    // TODO
    unimplemented!();
  }
}

pub trait DataIter: Iterator {
  fn reseed(&mut self, seed_rng: &mut Rng);
  fn reset(&mut self);

  fn map_data<F, V>(self, f: F) -> MapDataIter<Self, F, V> where F: FnMut(<Self as Iterator>::Item) -> V, Self: Sized {
    MapDataIter::new(self, f)
  }

  fn round_up_data(self, rdup_sz: usize) -> RoundUpDataIter<Self> where <Self as Iterator>::Item: Clone, Self: Sized {
    RoundUpDataIter::new(self, rdup_sz)
  }
}

pub struct OnePassDataSrc<R> where R: RandomAccess {
  data:     R,
  count:    usize,
}

impl<R> OnePassDataSrc<R> where R: RandomAccess {
  pub fn new(data: R) -> Self {
    OnePassDataSrc{
      data:     data,
      count:    0,
    }
  }
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

pub struct MapDataIter<I, F, V> where I: DataIter, F: FnMut(<I as Iterator>::Item) -> V {
  iter: I,
  mapf: F,
}

impl<I, F, V> MapDataIter<I, F, V> where I: DataIter, F: FnMut(<I as Iterator>::Item) -> V {
  pub fn new(iter: I, mapf: F) -> Self {
    MapDataIter{
      iter: iter,
      mapf: mapf,
    }
  }
}

impl<I, F, V> Iterator for MapDataIter<I, F, V> where I: DataIter, F: FnMut(<I as Iterator>::Item) -> V {
  type Item = V;

  fn next(&mut self) -> Option<Self::Item> {
    match self.iter.next() {
      None => None,
      Some(x) => Some((self.mapf)(x))
    }
  }
}

impl<I, F, V> DataIter for MapDataIter<I, F, V> where I: DataIter, F: FnMut(<I as Iterator>::Item) -> V {
  fn reseed(&mut self, seed_rng: &mut Rng) {
    self.iter.reseed(seed_rng);
  }

  fn reset(&mut self) {
    self.iter.reset();
  }
}

pub struct RoundUpDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  rdup_sz:  usize,
  iter:     I,
  closed:   bool,
  rep_ct:   usize,
  rep_item: Option<<I as Iterator>::Item>,
}

impl<I> RoundUpDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  pub fn new(iter: I, rdup_sz: usize) -> Self {
    RoundUpDataIter{
      rdup_sz:  rdup_sz,
      iter:     iter,
      closed:   false,
      rep_ct:   0,
      rep_item: None,
    }
  }
}

impl<I> Iterator for RoundUpDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
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

impl<I> DataIter for RoundUpDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
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
