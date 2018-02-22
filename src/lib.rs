extern crate extar;
extern crate imgio;
//extern crate rand;
extern crate sharedmem;

use std::sync::mpsc::*;

pub mod datasets;
pub mod imageproc;

pub trait RandomAccess {
  type Item;

  fn len(&self) -> usize;
  fn at(&self, idx: usize) -> Self::Item;
}

pub trait DataIter: Iterator {
  fn reset(&mut self, /*seed_rng: &mut Rng*/);
}

pub struct LoopOnceDataSrc<R> where R: RandomAccess {
  data:     R,
  count:    usize,
}

impl<R> Iterator for LoopOnceDataSrc<R> where R: RandomAccess {
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

impl<R> DataIter for LoopOnceDataSrc<R> where R: RandomAccess {
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

pub struct RoundupRepeatDataIter<I> where I: DataIter {
  rdup_sz:  usize,
  iter:     I,
  closed:   bool,
  rep_ct:   usize,
  rep_item: Option<<I as Iterator>::Item>,
}

impl<I> Iterator for RoundupRepeatDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  type Item = <I as Iterator>::Item;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    match self.next() {
      None => {
        if self.rep_ct < self.rdup_sz {
          self.rep_ct += 1;
          self.rep_item.clone()
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
        Some(item)
      }
    }
  }
}

impl<I> DataIter for RoundupRepeatDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  fn reset(&mut self, /*seed_rng: &mut Rng*/) {
    self.iter.reset(/*seed_rng*/);
    self.closed = false;
    self.rep_ct = 0;
    self.rep_item = None;
  }
}

pub struct ParallelJoinDataIter {
}
