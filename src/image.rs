use colorimage::{ColorImage};
use rand::prelude::*;
use rand::distributions::*;

use std::cmp::{min};

pub fn inception_crop_resize<R: Rng>(target_w: usize, target_h: usize, image: &mut ColorImage, rng: &mut R) {
  let old_w = image.width();
  let old_h = image.height();

  let area_dist = Uniform::new_inclusive(0.08, 1.0);
  let aspect_dist = Uniform::new_inclusive(0.75, 1.0);
  let rotate_dist = Bernoulli::new(0.5);
  let mut new_w = 0;
  let mut new_h = 0;
  loop {
    let area = (area_dist.sample(rng) * (old_w * old_h) as f64).max(1.0);
    let aspect = aspect_dist.sample(rng);
    match rotate_dist.sample(rng) {
      false => {
        new_w = (area / aspect).sqrt().round() as usize;
        new_h = (area * aspect).sqrt().round() as usize;
      }
      true  => {
        new_w = (area * aspect).sqrt().round() as usize;
        new_h = (area / aspect).sqrt().round() as usize;
      }
    }
    if new_w >= 1 && new_h >= 1 {
    //if new_w >= 1 && new_w <= old_w && new_h >= 1 && new_h <= old_h {
      break;
    }
  }
  new_w = min(new_w, old_w);
  new_h = min(new_h, old_h);

  let x_crop_dist = Uniform::new_inclusive(0, old_w - new_w);
  let y_crop_dist = Uniform::new_inclusive(0, old_h - new_h);
  let x_crop = x_crop_dist.sample(rng);
  let y_crop = y_crop_dist.sample(rng);
  image.crop(x_crop, y_crop, new_w, new_h);
  image.resize(target_w, target_h);
}

pub fn random_flip<R: Rng>(image: &mut ColorImage, rng: &mut R) {
  let flip_dist = Bernoulli::new(0.5);
  match flip_dist.sample(rng) {
    false => {}
    true  => {
      image.flip_left_right();
    }
  }
}

pub fn scale_resize(target_dim: usize, image: &mut ColorImage) {
  let old_w = image.width();
  let old_h = image.height();

  let (target_w, target_h) = if old_w > old_h {
    let new_w = (old_w as f64 * (target_dim as f64 / old_h as f64)).round() as usize;
    (new_w, target_dim)
  } else {
    let new_h = (old_h as f64 * (target_dim as f64 / old_w as f64)).round() as usize;
    (target_dim, new_h)
  };
  image.resize(target_w, target_h);
}

pub fn center_crop(target_w: usize, target_h: usize, image: &mut ColorImage) {
  let old_w = image.width();
  let old_h = image.height();

  assert!(old_w >= target_w);
  assert!(old_h >= target_h);
  let x_crop = (old_w - target_w) / 2;
  let y_crop = (old_h - target_h) / 2;
  image.crop(x_crop, y_crop, target_w, target_h);
}
