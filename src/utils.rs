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

use time::*;

pub fn duration_secs(d: Duration) -> f64 {
  d.num_seconds() as f64 +
      if let Some(ns) = d.num_nanoseconds() {
        ns as f64 * 1.0e-9
      } else if let Some(us) = d.num_microseconds() {
        us as f64 * 1.0e-6
      } else {
        d.num_milliseconds() as f64 * 1.0e-3
      }
}

pub struct Stopwatch {
  start:    PreciseTime,
  prev_lap: PreciseTime,
  curr_lap: PreciseTime,
}

impl Default for Stopwatch {
  fn default() -> Self {
    Stopwatch::new()
  }
}

impl Stopwatch {
  pub fn new() -> Self {
    let t = PreciseTime::now();
    Stopwatch{
      start:    t,
      prev_lap: t,
      curr_lap: t,
    }
  }

  pub fn click(&mut self) -> &mut Self {
    let t = PreciseTime::now();
    self.prev_lap = self.curr_lap;
    self.curr_lap = t;
    self
  }

  pub fn total_time(&self) -> f64 {
    duration_secs(self.start.to(self.curr_lap))
  }

  pub fn lap_time(&self) -> f64 {
    duration_secs(self.prev_lap.to(self.curr_lap))
  }
}
