use crate::hittable::{HitRecord, Ray};
use crate::util::*;
use image::RgbImage;
use rand::prelude::SliceRandom;
use rand::random;
use std::sync::Arc;
pub type MatPointer = Arc<dyn Material>;
pub type TexturePointer = Arc<dyn Texture>;
pub static BLACK_COLOR: Color = Vec3::new(0.0, 0.0, 0.0);
pub static WHITE_COLOR: Color = Vec3::new(1.0, 1.0, 1.0);

pub trait Texture: Send + Sync {
    fn value(&self, _uv: &UV, _p: &Point) -> Color {
        Vec3::new(0.0, 0.0, 0.0)
    }
}

pub struct SolidColor {
    pub(crate) color_value: Color,
}
impl SolidColor {
    pub fn new(color: Color) -> Self {
        Self { color_value: color }
    }
}
impl Texture for SolidColor {
    fn value(&self, _uv: &UV, _p: &Point) -> Color {
        self.color_value
    }
}
pub struct Picture {
    pixel: RgbImage,
}
impl Picture {
    pub fn new(file_path: &str) -> Self {
        let img = image::open(file_path).unwrap();
        Self {
            pixel: img.to_rgb8(),
        }
    }
}
impl Texture for Picture {
    fn value(&self, uv: &UV, _p: &Point) -> Color {
        let rgb = self.pixel.get_pixel(
            (self.pixel.width() as f64 * uv.x) as u32,
            (self.pixel.height() as f64 * uv.y) as u32,
        );
        rgb2vec3(rgb)
    }
}

pub struct CheckerTexture {
    pub inv_scale: f64,
    pub even: TexturePointer,
    pub odd: TexturePointer,
}
impl CheckerTexture {
    pub fn new_solid(scale: f64, color1: Color, color2: Color) -> Self {
        Self {
            inv_scale: 1.0 / scale,
            even: Arc::new(SolidColor {
                color_value: color1,
            }),
            odd: Arc::new(SolidColor {
                color_value: color2,
            }),
        }
    }
}

impl Texture for CheckerTexture {
    fn value(&self, uv: &UV, p: &Point) -> Color {
        if p.iter()
            .map(|x| (self.inv_scale * x).floor() as i32)
            .sum::<i32>()
            % 2
            == 0
        {
            self.even.value(uv, p)
        } else {
            self.odd.value(uv, p)
        }
    }
}
pub struct Perlin {
    point_count: usize,
    ranvec: Vec<Vec3>,
    perm_x: Vec<usize>,
    perm_y: Vec<usize>,
    perm_z: Vec<usize>,
}
impl Perlin {
    fn new() -> Self {
        let point_count = 256;
        let ranvec = (0..point_count).map(|_| random_in_sphere()).collect();
        Self {
            point_count,
            ranvec,
            perm_x: Self::perlin_generate_perm(point_count),
            perm_y: Self::perlin_generate_perm(point_count),
            perm_z: Self::perlin_generate_perm(point_count),
        }
    }
    fn noise(&self, p: &Point) -> f64 {
        let u = p.x - p.x.floor();
        let v = p.y - p.y.floor();
        let w = p.z - p.z.floor();
        let i = p.x.floor() as i32;
        let j = p.y.floor() as i32;
        let k = p.z.floor() as i32;
        let mut c = vec![vec![vec![ZERO; 2]; 2]; 2];
        let ct = self.point_count as i32 - 1;
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    c[di][dj][dk] = self.ranvec[self.perm_x[((i + di as i32) & ct) as usize]
                        ^ self.perm_y[((j + dj as i32) & ct) as usize]
                        ^ self.perm_z[((k + dk as i32) & ct) as usize]];
                }
            }
        }
        Self::trilinear_interp(&c, u, v, w)
    }
    fn turb(&self, p: &Point, depth: usize) -> f64 {
        let mut accum = 0.0;
        let mut temp_p = *p;
        let mut weight = 1.0;
        for _ in 0..depth {
            accum += weight * self.noise(&temp_p);
            weight *= 0.5;
            temp_p *= 2.0;
        }
        accum.abs()
    }
    fn perlin_generate_perm(point_count: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..point_count).collect();
        p.shuffle(&mut rand::thread_rng());
        //Self::permute(&mut p);
        p
    }
    fn permute(p: &mut Vec<usize>) {
        for i in (1..p.len()).rev() {
            p.swap(i, random_int(i));
        }
    }
    fn trilinear_interp(c: &Vec<Vec<Vec<Vec3>>>, u: f64, v: f64, w: f64) -> f64 {
        let mut accum = 0.0;
        let uu = u * u * (3.0 - 2.0 * u);
        let vv = v * v * (3.0 - 2.0 * v);
        let ww = w * w * (3.0 - 2.0 * w);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let weight_v = Vec3::new(u - i as f64, v - j as f64, w - k as f64);
                    accum += (i as f64 * uu + (1 - i) as f64 * (1.0 - uu))
                        * (j as f64 * vv + (1 - j) as f64 * (1.0 - vv))
                        * (k as f64 * ww + (1 - k) as f64 * (1.0 - ww))
                        * c[i][j][k].dot(&weight_v);
                }
            }
        }
        accum
    }
}

pub struct NoiseTexture {
    noise: Perlin,
    scale: f64,
}
impl NoiseTexture {
    pub fn new(scale: f64) -> Self {
        Self {
            noise: Perlin::new(),
            scale,
        }
    }
}
impl Texture for NoiseTexture {
    /*
    fn value(&self, _uv: &UV, p: &Point) -> Color {
        WHITE_COLOR * self.noise.noise(&(p * self.scale))
    }
    */
    fn value(&self, _uv: &UV, p: &Point) -> Color {
        let s = self.scale * p;
        WHITE_COLOR * 0.5 * (1.0 + (s.z + 10.0 * self.noise.turb(&s, 7)).sin())
    }
}

pub trait Material: Send + Sync {
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord) -> Option<(Color, Ray)> {
        None
    }
    fn emitted(&self, _uv: &UV, _p: &Point) -> Color {
        BLACK_COLOR
    }
}
pub struct Lambert {
    pub albedo: TexturePointer,
}
impl Material for Lambert {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        let scatter_direction = rec.normal + random_unit_vector();
        let scattered = Ray::new(rec.p, scatter_direction, r_in.time);
        let attenuation = self.albedo.value(&rec.uv, &rec.p);
        Some((attenuation, scattered))
    }
}

pub struct Metal {
    pub albedo: TexturePointer,
    pub fuzz: f64,
}
impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        let reflected = reflect(&r_in.dir.normalize(), &rec.normal);
        let scattered = Ray::new(rec.p, reflected + self.fuzz * random_in_sphere(), r_in.time);
        let attenuation = self.albedo.value(&rec.uv, &rec.p);
        if scattered.dir.dot(&rec.normal) > 0.0 {
            Some((attenuation, scattered))
        } else {
            None
        }
    }
}
fn schlick(cosine: f64, ref_idx: f64) -> f64 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * ((1.0 - cosine).powi(5))
}
fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = (-uv).dot(n);
    let r_out_parallel = etai_over_etat * (uv + cos_theta * n);
    let r_out_perp = -(1.0 - length_squared(&r_out_parallel)).sqrt() * n;
    r_out_parallel + r_out_perp
}
pub struct Dielectric {
    pub ref_idx: f64,
}
impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, _rec: &HitRecord) -> Option<(Color, Ray)> {
        let attenuation = Vec3::new(1.0, 1.0, 1.0);
        let etai_over_etat = if _rec.front_face {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };

        let unit_direction = r_in.dir.normalize();
        let cos_theta = (-unit_direction).dot(&_rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        if etai_over_etat * sin_theta > 1.0 {
            let reflected = reflect(&unit_direction, &_rec.normal);
            let scattered = Ray::new(_rec.p, reflected, r_in.time);
            return Some((attenuation, scattered));
        }
        let reflect_prob = schlick(cos_theta, etai_over_etat);
        if random::<f64>() < reflect_prob {
            let reflected = reflect(&unit_direction, &_rec.normal);
            let scattered = Ray::new(_rec.p, reflected, r_in.time);
            return Some((attenuation, scattered));
        }
        let refracted = refract(&unit_direction, &_rec.normal, etai_over_etat);
        let scattered = Ray::new(_rec.p, refracted, r_in.time);
        return Some((attenuation, scattered));
    }
}

pub struct Emit {
    pub texture: TexturePointer,
}

impl Material for Emit {
    fn emitted(&self, uv: &UV, p: &Point) -> Color {
        self.texture.value(uv, &p)
    }
}
pub struct Isotropic {
    pub albedo: TexturePointer,
}

impl Material for Isotropic {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        let scattered = Ray::new(rec.p, random_unit_vector(), r_in.time);
        let attenuation = self.albedo.value(&rec.uv, &rec.p);
        Some((attenuation, scattered))
    }
}
