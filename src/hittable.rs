use crate::material::*;
use crate::util::*;
use num_traits::FloatConst;
use rand::random;
use std::cmp::Ordering;
use std::default::Default;
use std::ops::Add;
use std::sync::Arc;

type ObjPointer = Arc<dyn Hittable>;
macro_rules! default_hittable_object {
    () => {
        Sphere {
            center: ZERO,
            radius: 1.0,
            mat_ptr: Arc::new(Lambert {
                albedo: Arc::new(SolidColor {
                    color_value: BLACK_COLOR,
                }),
            }),
        }
    };
}

#[derive(Default, Debug)]
pub struct Ray {
    pub orig: Vec3,
    pub dir: Vec3,
    pub time: f64,
}

impl Ray {
    pub fn new(orig: Vec3, dir: Vec3, time: f64) -> Self {
        Self { orig, dir, time }
    }
    pub fn at(&self, t: f64) -> Vec3 {
        self.orig + t * self.dir
    }
}

#[derive(Default, Copy, Clone)]
pub struct Interval {
    pub min: f64,
    pub max: f64,
}

impl Interval {
    pub fn new() -> Self {
        Interval { min: 0.0, max: 0.0 }
    }
    pub fn size(&self) -> f64 {
        self.max - self.min
    }
    pub fn expand(&self, delta: f64) -> Interval {
        let padding = delta / 2.0;
        Interval {
            min: self.min - padding,
            max: self.max + padding,
        }
    }
    pub fn interval2(a: &Interval, b: &Interval) -> Self {
        Self {
            min: a.min.min(b.min),
            max: a.max.max(b.max),
        }
    }
    pub fn add_f64(&self, rhs: f64) -> Self {
        Interval {
            min: self.min + rhs,
            max: self.max + rhs,
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct AABB([Interval; 3]);

impl Add for AABB {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_2aabb(&self, &rhs)
    }
}

impl AABB {
    fn new() -> AABB {
        AABB([Interval::new(), Interval::new(), Interval::new()])
    }
    fn add_offset(&self, rhs: &Vec3) -> AABB {
        AABB([
            self.0[0].add_f64(rhs.x),
            self.0[1].add_f64(rhs.y),
            self.0[2].add_f64(rhs.z),
        ])
    }
    fn from_point(a: &Point, b: &Point) -> AABB {
        let x = Interval {
            min: a.x.min(b.x),
            max: a.x.max(b.x),
        };
        let y = Interval {
            min: a.y.min(b.y),
            max: a.y.max(b.y),
        };
        let z = Interval {
            min: a.z.min(b.z),
            max: a.z.max(b.z),
        };
        AABB([x, y, z])
    }
    fn from_2aabb(a: &AABB, b: &AABB) -> AABB {
        let v: Vec<Interval> = (0..3)
            .map(|i| Interval::interval2(&a.0[i], &b.0[i]))
            .collect();
        AABB([v[0], v[1], v[2]])
    }
    fn pad(&self) -> AABB {
        let delta = 0.00001;
        let new_x = if self.0[0].size() >= delta {
            self.0[0]
        } else {
            self.0[0].expand(delta)
        };
        let new_y = if self.0[1].size() >= delta {
            self.0[1]
        } else {
            self.0[1].expand(delta)
        };
        let new_z = if self.0[2].size() >= delta {
            self.0[2]
        } else {
            self.0[2].expand(delta)
        };
        AABB([new_x, new_y, new_z])
    }
    fn hit(&self, r: &Ray, mut ray_t: Interval) -> bool {
        for i in 0..3 {
            let mut t0 = (self.0[i].min - r.orig[i]) / r.dir[i];
            let mut t1 = (self.0[i].max - r.orig[i]) / r.dir[i];
            (t0, t1) = (t0.min(t1), t0.max(t1));
            ray_t.min = ray_t.min.max(t0);
            ray_t.max = ray_t.max.min(t1);
            if ray_t.max <= ray_t.min {
                return false;
            }
        }
        true
    }
}
#[derive(Clone)]
pub struct HittableList {
    list: Vec<Arc<dyn Hittable>>,
    bbox: AABB,
}
impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut rec_closest = None;
        for obj in self.list.iter() {
            if let Some(rec) = obj.hit(r, t_min, t_max) {
                if rec_closest.is_none() {
                    rec_closest = Some(rec);
                } else {
                    if rec.t < rec_closest.as_ref().unwrap().t {
                        rec_closest = Some(rec);
                    }
                }
            }
        }
        rec_closest
    }
    fn bounding_box(&self) -> AABB {
        self.bbox
    }
}

impl HittableList {
    pub fn new() -> Self {
        Self {
            list: Vec::new(),
            bbox: AABB::new(),
        }
    }
    pub fn add(&mut self, obj: Arc<dyn Hittable>) {
        self.bbox = self.bbox + obj.bounding_box();
        self.list.push(obj);
    }
}
pub struct BVHNode {
    bbox: AABB,
    left: Arc<dyn Hittable>,
    right: Arc<dyn Hittable>,
}

impl Hittable for BVHNode {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        if !self.bbox.hit(
            r,
            Interval {
                min: t_min,
                max: t_max,
            },
        ) {
            return None;
        }
        if let Some(rec) = self.left.hit(r, t_min, t_max) {
            if let Some(rec2) = self.right.hit(r, t_min, rec.t) {
                Some(rec2)
            } else {
                Some(rec)
            }
        } else {
            if let Some(rec2) = self.right.hit(r, t_min, t_max) {
                Some(rec2)
            } else {
                None
            }
        }
    }
    fn bounding_box(&self) -> AABB {
        self.bbox
    }
}

impl BVHNode {
    pub fn new() -> Self {
        Self {
            bbox: Default::default(),
            left: Arc::new(default_hittable_object!()),
            right: Arc::new(default_hittable_object!()),
        }
    }
    pub fn bvh_node_build_root(mut src_objects: HittableList) -> BVHNode {
        let l = src_objects.list.len();
        let axis = random_int(3);
        src_objects
            .list
            .sort_unstable_by(|a, b| Self::box_comparator(a, b, axis));
        Self::bvh_node(src_objects, 0, l)
    }
    pub fn bvh_node(objects: HittableList, start: usize, end: usize) -> BVHNode {
        let object_span = end - start;
        let mut node = BVHNode::new();
        if object_span == 1 {
            node.left = objects.list[start].clone();
            node.right = objects.list[start].clone();
        } else if object_span == 2 {
            node.left = objects.list[start].clone();
            node.right = objects.list[start + 1].clone();
        } else {
            let mid = start + object_span / 2;
            node.left = Arc::new(BVHNode::bvh_node(objects.clone(), start, mid));
            node.right = Arc::new(BVHNode::bvh_node(objects.clone(), mid, end));
        }
        node.bbox = AABB::from_2aabb(&node.left.bounding_box(), &node.right.bounding_box());
        node
    }
    fn box_comparator(a: &Arc<dyn Hittable>, b: &Arc<dyn Hittable>, axis: usize) -> Ordering {
        f64cmp(a.bounding_box().0[axis].min, b.bounding_box().0[axis].min)
    }
}

pub struct HitRecord {
    pub p: Vec3,
    pub normal: Vec3,
    pub mat_ptr: MatPointer,
    pub t: f64,
    pub uv: Vec2,
    pub front_face: bool,
}

impl HitRecord {
    pub fn new() -> Self {
        Self {
            p: ZERO,
            normal: ZERO,
            mat_ptr: Arc::new(Lambert {
                albedo: Arc::new(SolidColor {
                    color_value: BLACK_COLOR,
                }),
            }),
            t: 0.0,
            uv: Vec2::new(0.0, 0.0),
            front_face: false,
        }
    }
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: &Vec3) {
        self.front_face = r.dir.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -outward_normal
        };
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, _r: &Ray, _t_min: f64, _t_max: f64) -> Option<HitRecord> {
        None
    }
    fn bounding_box(&self) -> AABB {
        AABB::from_point(&ZERO, &ZERO)
    }
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub mat_ptr: MatPointer,
}
impl Sphere {
    fn get_sphere_uv(&self, p: &Point) -> Vec2 {
        let theta = p.y.acos();
        let phi = p.x.atan2(p.z) + f64::PI();
        let u = phi / (2.0 * f64::PI());
        let v = theta / f64::PI();
        Vec2::new(u, v)
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = r.orig - self.center;
        let a = length_squared(&r.dir);
        let half_b = oc.dot(&r.dir);
        let c = length_squared(&oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            let t = (-half_b - root) / a;
            if t < t_max && t > t_min {
                let mut rec = HitRecord::new();
                rec.t = t;
                rec.p = r.at(t);
                rec.mat_ptr = Arc::clone(&self.mat_ptr);
                let outward_normal = (rec.p - self.center) / self.radius;
                rec.set_face_normal(r, &outward_normal);
                rec.uv = self.get_sphere_uv(&outward_normal);
                return Some(rec);
            }
            let t = (-half_b + root) / a;
            if t < t_max && t > t_min {
                let mut rec = HitRecord::new();
                rec.t = t;
                rec.p = r.at(t);
                rec.mat_ptr = self.mat_ptr.clone();
                let outward_normal = (rec.p - self.center) / self.radius;
                rec.set_face_normal(r, &outward_normal);
                rec.uv = self.get_sphere_uv(&outward_normal);
                return Some(rec);
            }
        }
        None
    }
    fn bounding_box(&self) -> AABB {
        let r_vec = Vec3::new(self.radius, self.radius, self.radius);
        AABB::from_point(&(self.center + r_vec), &(self.center - r_vec))
    }
}

pub struct MovingSphere {
    pub center0: Point,
    pub center1: Point,
    pub time0: f64,
    pub time1: f64,
    pub radius: f64,
    pub mat_ptr: MatPointer,
}

impl MovingSphere {
    fn center(&self, time: f64) -> Point {
        self.center0
            + ((time - self.time0) / (self.time1 - self.time0)) * (self.center1 - self.center0)
    }
}

impl Hittable for MovingSphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let center = self.center(r.time);
        let oc = r.orig - center;
        let a = length_squared(&r.dir);
        let half_b = oc.dot(&r.dir);
        let c = length_squared(&oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            let t = (-half_b - root) / a;
            if t < t_max && t > t_min {
                let mut rec = HitRecord::new();
                rec.t = t;
                rec.p = r.at(t);
                rec.mat_ptr = Arc::clone(&self.mat_ptr);
                let outward_normal = (rec.p - center) / self.radius;
                rec.set_face_normal(r, &outward_normal);
                return Some(rec);
            }
            let t = (-half_b + root) / a;
            if t < t_max && t > t_min {
                let mut rec = HitRecord::new();
                rec.t = t;
                rec.p = r.at(t);
                rec.mat_ptr = self.mat_ptr.clone();
                let outward_normal = (rec.p - center) / self.radius;
                rec.set_face_normal(r, &outward_normal);
                return Some(rec);
            }
        }
        None
    }
    fn bounding_box(&self) -> AABB {
        let r_vec = Vec3::new(self.radius, self.radius, self.radius);
        AABB::from_2aabb(
            &AABB::from_point(&(self.center0 + r_vec), &(self.center0 - r_vec)),
            &AABB::from_point(&(self.center1 + r_vec), &(self.center1 - r_vec)),
        )
    }
}

pub struct Quad {
    pub q: Point,
    pub u: Vec3,
    pub v: Vec3,
    pub mat: MatPointer,
    pub bbox: AABB,

    pub normal: Vec3,
    pub d: f64,
    pub w: Vec3,
}
impl Quad {
    pub fn new(q: &Point, u: &Vec3, v: &Vec3, mat: MatPointer) -> Self {
        let n = u.cross(v);
        let normal = n.normalize();
        Self {
            q: q.clone(),
            u: u.clone(),
            v: v.clone(),
            mat,
            bbox: AABB::from_point(&q, &(q + u + v)).pad(),
            normal,
            d: normal.dot(q),
            w: n / (n.dot(&n)),
        }
    }
}
impl Hittable for Quad {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let denom = self.normal.dot(&r.dir);
        if denom.abs() < 0.000000001 {
            return None;
        }
        let t = (self.d - self.normal.dot(&r.orig)) / denom;
        if t_min > t || t_max < t {
            return None;
        }
        let intersection = r.at(t);
        let planar_hitpt_vector = intersection - self.q;
        let alpha = self.w.dot(&planar_hitpt_vector.cross(&self.v));
        let beta = self.w.dot(&self.u.cross(&planar_hitpt_vector));
        if !(alpha >= 0.0 && alpha <= 1.0 && beta >= 0.0 && beta <= 1.0) {
            return None;
        }

        let mut rec = HitRecord::new();
        rec.t = t;
        rec.p = intersection;
        rec.mat_ptr = self.mat.clone();
        rec.set_face_normal(&r, &self.normal);
        rec.uv = Vec2::new(alpha, beta);
        Some(rec)
    }
    fn bounding_box(&self) -> AABB {
        self.bbox
    }
}

pub struct Translate {
    pub object: Arc<dyn Hittable>,
    pub offset: Vec3,
    pub bbox: AABB,
}
impl Translate {
    pub fn new(object: Arc<dyn Hittable>, offset: Vec3) -> Self {
        let bbox = object.bounding_box().add_offset(&offset);
        Self {
            object,
            offset,
            bbox,
        }
    }
}

impl Hittable for Translate {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let offset_r = Ray::new(r.orig - self.offset, r.dir, r.time);
        if let Some(mut rec) = self.object.hit(&offset_r, t_min, t_max) {
            rec.p += self.offset;
            Some(rec)
        } else {
            None
        }
    }
    fn bounding_box(&self) -> AABB {
        self.bbox
    }
}
pub struct RotateY {
    pub object: Arc<dyn Hittable>,
    pub sin_theta: f64,
    pub cos_theta: f64,
    pub bbox: AABB,
}
impl RotateY {
    pub fn new(object: Arc<dyn Hittable>, angle: f64) -> Self {
        let radians = degrees_to_radians(angle);
        let sin_theta = radians.sin();
        let cos_theta = radians.cos();
        let mut bbox = object.bounding_box();
        let mut min = Vec3::new(INFINITY, INFINITY, INFINITY);
        let mut max = Vec3::new(NEG_INFINITY, NEG_INFINITY, NEG_INFINITY);
        for i in 0..2 {
            let i = i as f64;
            for j in 0..2 {
                let j = j as f64;
                for k in 0..2 {
                    let k = k as f64;
                    let x = i * bbox.0[0].max + (1.0 - i) * bbox.0[0].min;
                    let y = j * bbox.0[1].max + (1.0 - j) * bbox.0[1].min;
                    let z = k * bbox.0[2].max + (1.0 - k) * bbox.0[2].min;
                    let new_x = cos_theta * x + sin_theta * z;
                    let new_z = -sin_theta * x + cos_theta * z;
                    let tester = Vec3::new(new_x, y, new_z);
                    for c in 0..3 {
                        min[c] = min[c].min(tester[c]);
                        max[c] = max[c].max(tester[c]);
                    }
                }
            }
        }
        bbox = AABB::from_point(&min, &max);
        Self {
            object,
            sin_theta,
            cos_theta,
            bbox,
        }
    }
}
impl Hittable for RotateY {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let origin = Vec3::new(
            self.cos_theta * r.orig[0] - self.sin_theta * r.orig[2],
            r.orig[1],
            self.sin_theta * r.orig[0] + self.cos_theta * r.orig[2],
        );
        let direction = Vec3::new(
            self.cos_theta * r.dir[0] - self.sin_theta * r.dir[2],
            r.dir[1],
            self.sin_theta * r.dir[0] + self.cos_theta * r.dir[2],
        );
        let rotated_r = Ray::new(origin, direction, r.time);
        let rec = self.object.hit(&rotated_r, t_min, t_max);
        if rec.is_none() {
            return None;
        }
        let mut rec = rec.unwrap();
        rec.p = Vec3::new(
            self.cos_theta * rec.p[0] + self.sin_theta * rec.p[2],
            rec.p[1],
            -self.sin_theta * rec.p[0] + self.cos_theta * rec.p[2],
        );
        rec.normal = Vec3::new(
            self.cos_theta * rec.normal[0] + self.sin_theta * rec.normal[2],
            rec.normal[1],
            -self.sin_theta * rec.normal[0] + self.cos_theta * rec.normal[2],
        );
        Some(rec)
    }
    fn bounding_box(&self) -> AABB {
        self.bbox
    }
}
pub struct ConstantMedium {
    boundary: Arc<dyn Hittable>,
    neg_inv_density: f64,
    phase_function: MatPointer,
}
impl ConstantMedium {
    pub fn new(obj: ObjPointer, d: f64, texture_pointer: TexturePointer) -> Self {
        Self {
            boundary: obj,
            neg_inv_density: -1.0 / d,
            phase_function: Arc::new(Isotropic {
                albedo: texture_pointer,
            }),
        }
    }
}
impl Hittable for ConstantMedium {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let rec1 = self.boundary.hit(r, NEG_INFINITY, INFINITY);
        if rec1.is_none() {
            return None;
        }
        let mut rec1 = rec1.unwrap();
        let rec2 = self.boundary.hit(r, rec1.t + 0.0001, INFINITY);
        if rec2.is_none() {
            return None;
        }
        let mut rec2 = rec2.unwrap();
        if rec1.t < t_min {
            rec1.t = t_min;
        }
        if rec2.t > t_max {
            rec2.t = t_max;
        }
        if rec1.t >= rec2.t {
            return None;
        }
        if rec1.t < 0. {
            rec1.t = 0.;
        }
        let ray_length = length(&r.dir);
        let distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
        let hit_distance = self.neg_inv_density * random::<f64>().ln();

        if hit_distance > distance_inside_boundary {
            return None;
        }
        let mut rec = HitRecord::new();
        rec.t = rec1.t + hit_distance / ray_length;
        rec.p = r.at(rec.t);
        rec.normal = Vec3::new(1., 0., 0.); // arbitrary
        rec.front_face = true; // also arbitrary
        rec.mat_ptr = self.phase_function.clone();
        Some(rec)
    }

    fn bounding_box(&self) -> AABB {
        self.boundary.bounding_box()
    }
}

pub fn hit_world(r: &Ray, world: &BVHNode) -> Option<HitRecord> {
    let mut rec_closest = None;
    if let Some(rec) = world.hit(r, 0.0001, INFINITY) {
        if rec_closest.is_none() {
            rec_closest = Some(rec);
        } else {
            if rec.t < rec_closest.as_ref().unwrap().t {
                rec_closest = Some(rec);
            }
        }
    }
    rec_closest
}
