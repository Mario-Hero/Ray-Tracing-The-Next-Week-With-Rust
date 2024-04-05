use image::Rgb;
use num_traits::float::FloatConst;
use rand::{random, thread_rng, Rng};
use std::cmp::Ordering;

pub type Vec3 = nalgebra::Vector3<f64>;
pub type Vec2 = nalgebra::Vector2<f64>;
pub type UV = Vec2;
pub type Point = Vec3;
pub type Color = Vec3;
pub static ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
pub static WORLD_UP: Vec3 = Vec3::new(0.0, 1.0, 0.0);
pub static INFINITY: f64 = f64::MAX;
pub static NEG_INFINITY: f64 = f64::MIN;

#[inline]
pub fn f64cmp(a: f64, b: f64) -> Ordering {
    if a > b {
        Ordering::Greater
    } else if a < b {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}
#[inline]
pub fn rgb2vec3(rgb: &Rgb<u8>) -> Color {
    let x = rgb.0[0] as f64 / 256f64;
    let y = rgb.0[1] as f64 / 256f64;
    let z = rgb.0[2] as f64 / 256f64;
    Vec3::new(x, y, z)
}

#[inline]
pub fn random_half() -> f64 {
    random::<f64>() - 0.5
}

pub fn random_in_sphere() -> Vec3 {
    loop {
        let v = Vec3::new(
            (random::<f64>() - 0.5) * 2.0,
            (random::<f64>() - 0.5) * 2.0,
            (random::<f64>() - 0.5) * 2.0,
        );
        if length_squared(&v) < 1.0 {
            return v.normalize();
        }
    }
}
pub fn random_unit_vector() -> Vec3 {
    let a = random::<f64>() * 2.0 * f64::PI();
    let z = random_half() * 2.0;
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * a.cos(), r * a.sin(), z)
}
/*
fn random_in_hemisphere(normal: &Vec3) -> Vec3 {
    let in_unit_sphere = random_in_sphere();
    if in_unit_sphere.dot(normal) > 0.0 {
        in_unit_sphere
    } else {
        -in_unit_sphere
    }
}

 */
#[inline]
pub fn clamp_one(x: f64) -> f64 {
    if x > 0.999 {
        0.999
    } else if x < 0.0 {
        0.0
    } else {
        x
    }
}
#[inline]
pub fn length_squared(v: &Vec3) -> f64 {
    v.dot(&v)
}
#[inline]
pub fn length(v: &Vec3) -> f64 {
    v.dot(&v).sqrt()
}
#[inline]
pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    v - v.dot(n) * 2.0 * n
}
#[inline]
pub fn mul(a: &Vec3, b: &Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

#[inline]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * f64::PI() / 180.0
}

#[inline]
pub fn random_int(n: usize) -> usize {
    thread_rng().gen_range(0..n)
}
pub fn random_in_unit_disk() -> Vec3 {
    loop {
        let p = Vec3::new(random_half() * 2.0, random_half() * 2.0, 0.0);
        if length_squared(&p) < 1.0 {
            return p;
        }
    }
}
pub fn random_vec3() -> Vec3 {
    Vec3::new(random::<f64>(), random::<f64>(), random::<f64>())
}

pub fn random_range_f64(a: f64, b: f64) -> f64 {
    random::<f64>() * (b - a) + a
}
