extern crate byteorder;
extern crate colorimage;
extern crate extar;
#[cfg(feature = "mpi")] extern crate mpich;
extern crate rand;
//extern crate rng;
extern crate sharedmem;
extern crate string_cache;
extern crate time;

use rand::prelude::*;
use rand::distributions::*;
//use rng::*;

use std::cmp::{min};
use std::collections::{VecDeque};
use std::marker::{PhantomData};
use std::sync::{Arc};
use std::sync::mpsc::*;
use std::thread;

pub mod datasets;
pub mod image;
pub mod utils;

pub trait RandomAccess {
  type Item;

  fn len(&self) -> usize;
  fn at(&self, idx: usize) -> Self::Item;

  fn partitions(&self, num_parts: usize) -> Vec<RangeData<Self>> where Self: Clone + Sized {
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

  fn partition(self, p: usize, num_parts: usize) -> RangeData<Self> where Self: Sized {
    let total_len = self.len();
    let max_part_len = (total_len + num_parts - 1) / num_parts;
    let part_start_idx = p * max_part_len;
    let part_end_idx = min(total_len, (p + 1) * max_part_len);
    RangeData::new(self, part_start_idx, part_end_idx)
  }

  fn one_pass(self) -> OnePassDataSrc<Self> where Self: Sized {
    OnePassDataSrc::new(self)
  }
}

pub trait RandomSample<R: Rng>: RandomAccess {
  fn uniform_shuffle(self, seed_rng: &mut R) -> UniformShuffleDataSrc<Self, R> where Self: Sized {
    // TODO
    unimplemented!();
  }

  fn uniform_random(self, seed_rng: &mut R) -> UniformRandomDataSrc<Self, R> where Self: Sized {
    UniformRandomDataSrc::new(self, seed_rng)
  }
}

impl<R: Rng, D: RandomAccess> RandomSample<R> for D {
}

pub trait DataIter: Iterator {
  /*fn reseed(&mut self, seed_rng: &mut Rng) { unimplemented!(); }*/
  fn reset(&mut self);

  fn loop_reset_data(self) -> LoopResetDataIter<Self> where Self: Sized {
    LoopResetDataIter::new(self)
  }

  fn map_data<F, V>(self, f: F) -> MapDataIter<Self, F, V> where F: FnMut(<Self as Iterator>::Item) -> V, Self: Sized {
    MapDataIter::new(self, f)
  }

  fn round_up_data(self, rdup_sz: usize) -> RoundUpDataIter<Self> where <Self as Iterator>::Item: Clone, Self: Sized {
    RoundUpDataIter::new(self, rdup_sz)
  }

  fn batch_data(self, batch_sz: usize) -> BatchDataIter<Self> where Self: Sized {
    BatchDataIter::new(self, batch_sz)
  }

  fn async_prefetch_data(self, capacity: usize) -> AsyncPrefetchDataIter<<Self as Iterator>::Item> where Self: Sized + Send + 'static, <Self as Iterator>::Item: Send {
    async_prefetch_data(capacity, self)
  }
}

pub struct RangeData<D> where D: RandomAccess {
  data:     D,
  offset:   usize,
  len:      usize,
  //_mrk:     PhantomData<fn (&mut R)>,
}

impl<D> RangeData<D> where D: RandomAccess {
  pub fn new(data: D, start_idx: usize, end_idx: usize) -> Self {
    assert!(start_idx <= end_idx);
    RangeData{
      data:     data,
      offset:   start_idx,
      len:      end_idx - start_idx,
      //_mrk:     PhantomData,
    }
  }
}

impl<D> RandomAccess for RangeData<D> where D: RandomAccess {
  type Item = <D as RandomAccess>::Item;

  fn len(&self) -> usize {
    self.len
  }

  fn at(&self, idx: usize) -> Self::Item {
    assert!(idx < self.len);
    self.data.at(self.offset + idx)
  }
}

pub struct OnePassDataSrc<D> where D: RandomAccess {
  data:     D,
  counter:  usize,
}

impl<D> OnePassDataSrc<D> where D: RandomAccess {
  pub fn new(data: D) -> Self {
    OnePassDataSrc{
      data:     data,
      counter:  0,
    }
  }
}

impl<D> Iterator for OnePassDataSrc<D> where D: RandomAccess {
  type Item = <D as RandomAccess>::Item;

  fn next(&mut self) -> Option<Self::Item> {
    if self.counter < self.data.len() {
      let item = self.data.at(self.counter);
      self.counter += 1;
      Some(item)
    } else {
      None
    }
  }
}

impl<D> DataIter for OnePassDataSrc<D> where D: RandomAccess {
  /*fn reseed(&mut self, seed_rng: &mut Rng) {
  }*/

  fn reset(&mut self) {
    self.counter = 0;
  }
}

pub struct UniformShuffleDataSrc<D, R> where D: RandomAccess, R: Rng {
  //rng:      Xorshiftplus128Rng,
  data:     D,
  counter:  usize,
  _mrk:     PhantomData<R>,
}

pub struct UniformRandomDataSrc<D, R> where D: RandomAccess, R: Rng {
  //rng:      Xorshiftplus128Rng,
  data:     D,
  dist:     Uniform<usize>,
  _mrk:     PhantomData<R>,
}

impl<D, R> UniformRandomDataSrc<D, R> where D: RandomAccess, R: Rng {
  pub fn new(data: D, seed_rng: &mut R) -> Self {
    // TODO: use the seed RNG.
    let dist = Uniform::new(0, data.len());
    UniformRandomDataSrc{
      data:     data,
      dist:     dist,
      _mrk:     PhantomData,
    }
  }
}

impl<D, R> Iterator for UniformRandomDataSrc<D, R> where D: RandomAccess, R: Rng {
  type Item = <D as RandomAccess>::Item;

  fn next(&mut self) -> Option<Self::Item> {
    // TODO: use locally seeded RNG.
    let idx = self.dist.sample(&mut thread_rng());
    Some(self.data.at(idx))
  }
}

impl<D, R> DataIter for UniformRandomDataSrc<D, R> where D: RandomAccess, R: Rng {
  fn reset(&mut self) {
    // TODO
  }
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
  /*fn reseed(&mut self, seed_rng: &mut Rng) {
    self.iter.reseed(seed_rng);
  }*/

  fn reset(&mut self) {
    self.iter.reset();
  }
}

pub struct RoundUpDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  rdup_sz:  usize,
  iter:     I,
  closed:   bool,
  rep_ctr:  usize,
  rep_item: Option<<I as Iterator>::Item>,
}

impl<I> RoundUpDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  pub fn new(iter: I, rdup_sz: usize) -> Self {
    RoundUpDataIter{
      rdup_sz:  rdup_sz,
      iter:     iter,
      closed:   false,
      rep_ctr:  0,
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
        if self.rep_ctr > 0 {
          self.rep_ctr += 1;
          if self.rep_ctr == self.rdup_sz {
            self.rep_ctr = 0;
          }
          self.rep_item.clone().map(|item| (item, true))
        } else {
          self.closed = true;
          self.rep_item = None;
          None
        }
      }
      Some(item) => {
        self.rep_ctr += 1;
        if self.rep_ctr == self.rdup_sz {
          self.rep_ctr = 0;
        }
        self.rep_item = Some(item.clone());
        Some((item, false))
      }
    }
  }
}

impl<I> DataIter for RoundUpDataIter<I> where I: DataIter, <I as Iterator>::Item: Clone {
  /*fn reseed(&mut self, seed_rng: &mut Rng) {
    self.iter.reseed(seed_rng);
  }*/

  fn reset(&mut self) {
    self.iter.reset();
    self.closed = false;
    self.rep_ctr = 0;
    self.rep_item = None;
  }
}

pub struct BatchDataIter<I> where I: DataIter {
  batch_sz: usize,
  iter:     I,
  closed:   bool,
  cache:    Vec<<I as Iterator>::Item>,
}

impl<I> BatchDataIter<I> where I: DataIter {
  pub fn new(iter: I, batch_sz: usize) -> Self {
    BatchDataIter{
      batch_sz: batch_sz,
      iter:     iter,
      closed:   false,
      cache:    Vec::with_capacity(batch_sz),
    }
  }
}

impl<I> Iterator for BatchDataIter<I> where I: DataIter {
  type Item = Vec<<I as Iterator>::Item>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    for _ in 0 .. self.batch_sz {
      match self.iter.next() {
        None => {
          self.closed = true;
          let mut batch = Vec::with_capacity(self.cache.len());
          for item in self.cache.drain(..) {
            batch.push(item);
          }
          return Some(batch);
        }
        Some(item) => {
          self.cache.push(item);
          if self.cache.len() == self.batch_sz {
            let mut batch = Vec::with_capacity(self.batch_sz);
            for item in self.cache.drain(..) {
              batch.push(item);
            }
            return Some(batch);
          }
        }
      }
    }
    unreachable!();
  }
}

impl<I> DataIter for BatchDataIter<I> where I: DataIter {
  /*fn reseed(&mut self, seed_rng: &mut Rng) {
    // TODO
  }*/

  fn reset(&mut self) {
    self.iter.reset();
    self.closed = false;
    self.cache.clear();
  }
}

enum AsyncWorkerMsg<Item> {
  Closed,
  Next(Item),
}

enum AsyncCtrlMsg {
  Reset,
  Reseed,
  Stop,
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
            _ => {
              // TODO
              unimplemented!();
            }
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

struct AsyncSplitWorkerState<I> where I: Iterator {
  nworkers: usize,
  closed:   bool,
  ctr:      usize,
  iter:     I,
  w_txs:    Vec<Sender<AsyncWorkerMsg<<I as Iterator>::Item>>>,
  c_rxs:    Vec<Receiver<AsyncCtrlMsg>>,
}

impl<I> AsyncSplitWorkerState<I> where I: Iterator {
  fn runloop(&mut self) {
    loop {
      if self.closed {
        // TODO
        break;
        /*match self.c_rx.recv() {
          Err(_) => {
            break;
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
          },
        }*/
      }
      /*match self.c_rx.try_recv() {
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
      }*/
      let idx = self.ctr;
      let rank = idx % self.nworkers;
      match self.iter.next() {
        None => {
          self.closed = true;
          for offset in 0 .. self.nworkers {
            self.w_txs[(rank + offset) % self.nworkers].send(AsyncWorkerMsg::Closed).unwrap();
          }
        }
        Some(item) => {
          self.w_txs[rank].send(AsyncWorkerMsg::Next(item)).unwrap();
          self.ctr += 1;
        }
      }
    }
  }
}

pub fn async_split_data<F, I>(num_workers: usize, f: F) -> Vec<AsyncSplitDataIter<<I as Iterator>::Item>>
where F: FnOnce() -> I,
      I: DataIter + Send + 'static,
      <I as Iterator>::Item: Send + 'static,
{
  let mut splits = Vec::with_capacity(num_workers);
  let mut w_txs = Vec::with_capacity(num_workers);
  let mut c_rxs = Vec::with_capacity(num_workers);
  let mut w_rx_c_tx_s = Vec::with_capacity(num_workers);
  for rank in 0 .. num_workers {
    let (w_tx, w_rx) = channel();
    let (c_tx, c_rx) = channel();
    w_txs.push(w_tx);
    c_rxs.push(c_rx);
    w_rx_c_tx_s.push((w_rx, c_tx));
  }
  let iter = f();
  let h = thread::spawn(move || {
    let mut state = AsyncSplitWorkerState{
      nworkers: num_workers,
      closed:   false,
      ctr:      0,
      iter:     iter,
      w_txs:    w_txs,
      c_rxs:    c_rxs,
    };
    state.runloop();
  });
  let drop_h = Arc::new(DropJoinHandle{handle: Some(h)});
  for (rank, (w_rx, c_tx)) in w_rx_c_tx_s.drain(..).enumerate() {
    let split = AsyncSplitDataIter{
      rank:     rank,
      closed:   false,
      w_rx:     w_rx,
      c_tx:     c_tx,
      w_h:      drop_h.clone(),
    };
    splits.push(split);
  }
  splits
}

struct DropJoinHandle {
  handle:   Option<thread::JoinHandle<()>>,
}

impl Drop for DropJoinHandle {
  fn drop(&mut self) {
    assert!(self.handle.take().unwrap().join().is_ok());
  }
}

pub struct AsyncSplitDataIter<Item> {
  rank:     usize,
  closed:   bool,
  //ctr:      usize,
  w_rx:     Receiver<AsyncWorkerMsg<Item>>,
  c_tx:     Sender<AsyncCtrlMsg>,
  //w_hs:     Vec<thread::JoinHandle<()>>,
  w_h:      Arc<DropJoinHandle>,
}

impl<Item> Iterator for AsyncSplitDataIter<Item> {
  type Item = Item;

  fn next(&mut self) -> Option<Item> {
    if self.closed {
      return None;
    }
    match self.w_rx.recv() {
      Err(_) => {
        self.closed = true;
        return None;
      }
      Ok(msg) => match msg {
        AsyncWorkerMsg::Closed => {
          self.closed = true;
          return None;
        }
        AsyncWorkerMsg::Next(item) => {
          return Some(item);
        }
      },
    }
  }
}

impl<Item> DataIter for AsyncSplitDataIter<Item> {
  /*fn reseed(&mut self, seed_rng: &mut Rng) {
    // TODO
    unimplemented!();
  }*/

  fn reset(&mut self) {
    // TODO
  }
}

pub fn async_join_data<F, I>(num_workers: usize, mut f: F) -> AsyncJoinDataIter<<I as Iterator>::Item>
where F: FnMut(usize) -> I,
      I: DataIter + Send + 'static,
      <I as Iterator>::Item: Send + 'static,
{
  let mut rclosed = vec![];
  let mut w_rxs = vec![];
  let mut c_txs = vec![];
  let mut w_hs = vec![];
  for rank in 0 .. num_workers {
    let (w_tx, w_rx) = channel();
    let (c_tx, c_rx) = channel();
    let iter = f(rank);
    let w_h = thread::spawn(move || {
      let mut state = AsyncWorkerState{
        closed: false,
        iter:   iter,
        w_tx:   w_tx,
        c_rx:   c_rx,
      };
      state.runloop();
    });
    rclosed.push(false);
    w_rxs.push(w_rx);
    c_txs.push(c_tx);
    w_hs.push(w_h);
  }
  AsyncJoinDataIter{
    ctr:        0,
    nworkers:   num_workers,
    nclosed:    0,
    rclosed:    rclosed,
    w_rxs:      w_rxs,
    c_txs:      c_txs,
    w_hs:       w_hs,
  }
}

pub struct AsyncJoinDataIter<Item> {
  ctr:      usize,
  nworkers: usize,
  nclosed:  usize,
  rclosed:  Vec<bool>,
  w_rxs:    Vec<Receiver<AsyncWorkerMsg<Item>>>,
  c_txs:    Vec<Sender<AsyncCtrlMsg>>,
  w_hs:     Vec<thread::JoinHandle<()>>,
}

impl<Item> Iterator for AsyncJoinDataIter<Item> {
  type Item = Item;

  fn next(&mut self) -> Option<Item> {
    while self.nworkers > self.nclosed {
      // TODO: sample uniformly? or round robin?
      let rank = self.ctr % self.nworkers;
      self.ctr += 1;
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
  /*fn reseed(&mut self, seed_rng: &mut Rng) {
    // TODO
    unimplemented!();
  }*/

  fn reset(&mut self) {
    // TODO: flush receivers.
    for tx in self.c_txs.iter() {
      tx.send(AsyncCtrlMsg::Reset).unwrap();
    }
  }
}

pub fn async_prefetch_data<I>(capacity: usize, iter: I) -> AsyncPrefetchDataIter<<I as Iterator>::Item>
where I: DataIter + Send + 'static,
      <I as Iterator>::Item: Send,
{
  let (w_tx, w_rx) = channel();
  let (c_tx, c_rx) = channel();
  let w_h = thread::spawn(move || {
    let mut state = AsyncWorkerState{
      closed: false,
      iter:   iter,
      w_tx:   w_tx,
      c_rx:   c_rx,
    };
    state.runloop();
  });
  AsyncPrefetchDataIter{
    closed: false,
    flag:   false,
    queue:  VecDeque::with_capacity(capacity),
    w_rx:   w_rx,
    c_tx:   c_tx,
    w_h:    w_h,
  }
}

pub struct AsyncPrefetchDataIter<Item> {
  closed:   bool,
  flag:     bool,
  queue:    VecDeque<Option<Item>>,
  w_rx:     Receiver<AsyncWorkerMsg<Item>>,
  c_tx:     Sender<AsyncCtrlMsg>,
  w_h:      thread::JoinHandle<()>,
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
