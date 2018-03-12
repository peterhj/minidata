extern crate colorimage;
extern crate extar;
extern crate rand;
//extern crate rng;
extern crate sharedmem;

//use rng::*;

use rand::*;
use std::cmp::{min};
use std::collections::{VecDeque};
use std::sync::mpsc::*;
use std::thread;

pub mod datasets;
pub mod image;

pub trait RandomAccess {
  type Item;

  fn len(&self) -> usize;
  fn at(&self, idx: usize) -> Self::Item;

  fn partition(&self, num_parts: usize) -> Vec<RangeData<Self>> where Self: Clone + Sized {
    let total_len = self.len();
    let max_part_len = (total_len + num_parts - 1) / num_parts;
    let mut parts = vec![];
    for p in 0 .. num_parts {
      let part_start_idx = p * max_part_len;
      let part_end_idx = min(total_len, (p + 1) * max_part_len);
      parts.push(RangeData::new(self.clone(), part_start_idx, part_end_idx));
    }
    parts
  }

  fn one_pass(self) -> OnePassDataSrc<Self> where Self: Sized {
    OnePassDataSrc::new(self)
  }

  fn uniform_shuffle(self, seed_rng: &mut Rng) -> UniformShuffleDataSrc<Self> where Self: Sized {
    // TODO
    unimplemented!();
  }

  fn uniform_random(self, seed_rng: &mut Rng) -> UniformRandomDataSrc<Self> where Self: Sized {
    // TODO
    unimplemented!();
  }
}

pub trait DataIter: Iterator {
  fn reset(&mut self);
  fn reseed(&mut self, seed_rng: &mut Rng) { unimplemented!(); }

  fn loop_reset(self) -> LoopResetDataIter<Self> where Self: Sized {
    LoopResetDataIter::new(self)
  }

  fn map_data<F, V>(self, f: F) -> MapDataIter<Self, F, V> where F: FnMut(<Self as Iterator>::Item) -> V, Self: Sized {
    MapDataIter::new(self, f)
  }

  fn round_up_data(self, rdup_sz: usize) -> RoundUpDataIter<Self> where <Self as Iterator>::Item: Clone, Self: Sized {
    RoundUpDataIter::new(self, rdup_sz)
  }
}

pub struct RangeData<R> where R: RandomAccess {
  data:     R,
  offset:   usize,
  len:      usize,
}

impl<R> RangeData<R> where R: RandomAccess {
  pub fn new(data: R, start_idx: usize, end_idx: usize) -> Self {
    assert!(start_idx <= end_idx);
    RangeData{
      data:     data,
      offset:   start_idx,
      len:      end_idx - start_idx,
    }
  }
}

impl<R> RandomAccess for RangeData<R> where R: RandomAccess {
  type Item = <R as RandomAccess>::Item;

  fn len(&self) -> usize {
    self.len
  }

  fn at(&self, idx: usize) -> Self::Item {
    assert!(idx < self.len);
    self.data.at(self.offset + idx)
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
  //rng:      Xorshiftplus128Rng,
  data:     R,
  count:    usize,
}

pub struct UniformRandomDataSrc<R> where R: RandomAccess {
  //rng:      Xorshiftplus128Rng,
  data:     R,
}

pub struct LoopResetDataIter<I> where I: DataIter {
  iter: I,
}

impl<I> LoopResetDataIter<I> where I: DataIter {
  pub fn new(iter: I) -> Self {
    LoopResetDataIter{iter: iter}
  }
}

impl<I> Iterator for LoopResetDataIter<I> where I: DataIter {
  type Item = <I as Iterator>::Item;

  fn next(&mut self) -> Option<Self::Item> {
    match self.iter.next() {
      None => {
        self.iter.reset();
        match self.iter.next() {
          None => panic!(),
          Some(item) => Some(item),
        }
      }
      Some(item) => Some(item),
    }
  }
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

enum AsyncWorkerMsg<Item> {
  Closed,
  Next(Item),
}

enum AsyncCtrlMsg {
  Reset,
  Reseed,
}

struct AsyncWorkerState<I> where I: Iterator {
  closed:   bool,
  iter:     I,
  w_tx:     Sender<AsyncWorkerMsg<<I as Iterator>::Item>>,
  c_rx:     Receiver<AsyncCtrlMsg>,
}

impl<I> AsyncWorkerState<I> where I: Iterator {
  fn runloop(&mut self) {
    loop {
      if self.closed {
        match self.c_rx.recv() {
          Err(_) => {
            break;
          }
          Ok(cmsg) => match cmsg {
            AsyncCtrlMsg::Reset => {
              self.closed = false;
              continue;
            }
            _ => {}
          },
        }
      }
      match self.c_rx.try_recv() {
        Err(TryRecvError::Empty) => {}
        Err(_) => {
          self.closed = true;
          continue;
        }
        Ok(cmsg) => match cmsg {
          AsyncCtrlMsg::Reset => {
            self.closed = false;
            continue;
          }
          _ => {
            // TODO
            unimplemented!();
          }
        }
      }
      match self.iter.next() {
        None => {
          self.closed = true;
          self.w_tx.send(AsyncWorkerMsg::Closed).unwrap();
        }
        Some(item) => {
          self.w_tx.send(AsyncWorkerMsg::Next(item)).unwrap();
        }
      }
    }
  }
}

pub fn async_join_data<F, I>(num_workers: usize, f: F) -> AsyncJoinDataIter<<I as Iterator>::Item> where F: Fn(usize) -> I, I: DataIter {
  let mut rclosed = vec![];
  let mut w_rxs = vec![];
  let mut c_txs = vec![];
  for rank in 0 .. num_workers {
    let (w_tx, w_rx) = channel();
    let (c_tx, c_rx) = channel();
    // TODO
    rclosed.push(false);
    w_rxs.push(w_rx);
    c_txs.push(c_tx);
  }
  AsyncJoinDataIter{
    nworkers:   num_workers,
    nclosed:    0,
    rclosed:    rclosed,
    w_rxs:      w_rxs,
    c_txs:      c_txs,
  }
}

pub struct AsyncJoinDataIter<Item> {
  nworkers: usize,
  nclosed:  usize,
  rclosed:  Vec<bool>,
  w_rxs:    Vec<Receiver<AsyncWorkerMsg<Item>>>,
  c_txs:    Vec<Sender<AsyncCtrlMsg>>,
}

impl<Item> Iterator for AsyncJoinDataIter<Item> {
  type Item = Item;

  fn next(&mut self) -> Option<Item> {
    while self.nworkers > self.nclosed {
      let rank = 0; // TODO: sample uniformly? or round robin?
      if self.rclosed[rank] {
        continue;
      }
      match self.w_rxs[rank].recv() {
        Err(_) => {
          self.nclosed += 1;
          self.rclosed[rank] = true;
          continue;
        }
        Ok(msg) => match msg {
          AsyncWorkerMsg::Closed => {
            self.nclosed += 1;
            self.rclosed[rank] = true;
            continue;
          }
          AsyncWorkerMsg::Next(item) => {
            return Some(item);
          }
        },
      }
    }
    None
  }
}

impl<Item> DataIter for AsyncJoinDataIter<Item> {
  fn reseed(&mut self, seed_rng: &mut Rng) {
    // TODO
    unimplemented!();
  }

  fn reset(&mut self) {
    // TODO: flush receivers.
    for tx in self.c_txs.iter() {
      tx.send(AsyncCtrlMsg::Reset).unwrap();
    }
  }
}

pub fn async_prefetch_data<I>(capacity: usize, iter: I) -> AsyncPrefetchDataIter<<I as Iterator>::Item> where I: DataIter + Send + 'static, <I as Iterator>::Item: Send {
  let (w_tx, w_rx) = channel();
  let (c_tx, c_rx) = channel();
  let mut state = AsyncWorkerState{
    closed: false,
    iter:   iter,
    w_tx:   w_tx,
    c_rx:   c_rx,
  };
  // TODO: keep thread handle to join.
  let _ = thread::spawn(move || {
    state.runloop();
  });
  AsyncPrefetchDataIter{
    closed: false,
    flag:   false,
    queue:  VecDeque::with_capacity(capacity),
    w_rx:   w_rx,
    c_tx:   c_tx,
  }
}

pub struct AsyncPrefetchDataIter<Item> {
  closed:   bool,
  flag:     bool,
  queue:    VecDeque<Option<Item>>,
  w_rx:     Receiver<AsyncWorkerMsg<Item>>,
  c_tx:     Sender<AsyncCtrlMsg>,
}

impl<Item> Iterator for AsyncPrefetchDataIter<Item> {
  type Item = Item;

  fn next(&mut self) -> Option<Item> {
    if self.closed {
      return None;
    }
    if !self.flag && self.queue.is_empty() {
      match self.w_rx.recv() {
        Err(_) => {
          self.flag = true;
          self.queue.push_back(None);
        }
        Ok(msg) => match msg {
          AsyncWorkerMsg::Closed => {
            self.flag = true;
            self.queue.push_back(None);
          }
          AsyncWorkerMsg::Next(item) => {
            self.queue.push_back(Some(item));
          }
        },
      }
    }
    while !self.flag && self.queue.len() < self.queue.capacity() {
      match self.w_rx.try_recv() {
        Err(TryRecvError::Empty) => {
          break;
        }
        Err(_) => {
          self.flag = true;
          self.queue.push_back(None);
        }
        Ok(msg) => match msg {
          AsyncWorkerMsg::Closed => {
            self.flag = true;
            self.queue.push_back(None);
          }
          AsyncWorkerMsg::Next(item) => {
            self.queue.push_back(Some(item));
          }
        },
      }
    }
    match self.queue.pop_front() {
      None => {
        assert!(self.flag);
        self.closed = true;
        None
      }
      Some(maybe_item) => {
        if maybe_item.is_none() {
          assert!(self.flag);
          assert!(self.queue.is_empty());
          self.closed = true;
        }
        maybe_item
      }
    }
  }
}

impl<Item> DataIter for AsyncPrefetchDataIter<Item> {
  fn reset(&mut self) {
    self.closed = false;
    self.flag = false;
    self.queue.clear();
    // TODO: flush channels.
    self.c_tx.send(AsyncCtrlMsg::Reset).unwrap();
  }
}
