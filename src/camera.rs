use crate::hittable::{hit_world, BVHNode, Ray};
use crate::util::*;
use rand::random;

#[derive(Default)]
pub struct Camera {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub background: Color,
    pub samples_per_pixel: usize,
    pub max_depth: usize,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    pixel00_loc: Point,
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
    center: Point,
    defocus_angle: f64,
    focus_dist:f64
}

impl Camera {
    pub fn new(
        look_from: Vec3,
        look_at: Vec3,
        vup: Vec3,
        fov: f64,
        defocus_angle: f64,
        background: Color,
        image_width: usize,
        image_height: usize,
        samples_per_pixel: usize,
        max_depth: usize,
    ) -> Self {
        let aspect_ratio = image_width as f64 / image_height as f64;
        let mut cam = Camera {
            ..Default::default()
        };
        cam.focus_dist = 10.0;
        cam.center = look_from;
        cam.defocus_angle = defocus_angle;
        let theta = degrees_to_radians(fov);
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * cam.focus_dist;
        let viewport_width = aspect_ratio * viewport_height;
        cam.w = (look_from - look_at).normalize();
        cam.u = vup.cross(&cam.w).normalize();
        cam.v = cam.w.cross(&cam.u);
        let viewport_u = viewport_width * cam.u;
        let viewport_v = viewport_height * (-cam.v);
        cam.pixel_delta_u = viewport_u / image_width as f64;
        cam.pixel_delta_v = viewport_v / image_height as f64;
        cam.max_depth = max_depth;
        cam.samples_per_pixel = samples_per_pixel;

        let viewport_upper_left =
            look_from - (cam.focus_dist * cam.w) - viewport_u / 2.0 - viewport_v / 2.0;
        cam.pixel00_loc = viewport_upper_left + 0.5 * (cam.pixel_delta_u + cam.pixel_delta_v);
        let focus_dist = length(&(look_from - look_at));
        let defocus_radius = focus_dist * degrees_to_radians(defocus_angle / 2.0).tan();
        cam.defocus_disk_u = cam.u * defocus_radius;
        cam.defocus_disk_v = cam.v * defocus_radius;
        cam.background = background;
        cam
    }
    pub fn get_ray(&self, i: usize, j: usize) -> Ray {
        let pixel_center = self.pixel00_loc + (i as f64 * self.pixel_delta_u) + (j as f64 * self.pixel_delta_v);
        let pixel_sample = pixel_center + self.pixel_sample_square();
        let ray_origin = if self.defocus_angle <= 0.0 {
            self.center
        } else {
            self.defocus_disk_sample()
        };
        let ray_direction = (pixel_sample - ray_origin).normalize();
        Ray::new(ray_origin, ray_direction, random::<f64>())
    }
    fn defocus_disk_sample(&self) -> Point{
        // Returns a random point in the camera defocus disk.
        let p = random_in_unit_disk();
        self.center + (p[0] * self.defocus_disk_u) + (p[1] * self.defocus_disk_v)
    }
    fn pixel_sample_square(&self) -> Vec3 {
    // Returns a random point in the square surrounding a pixel at the origin.
        let px = random_half();
        let py = random_half();
        (px * self.pixel_delta_u) + (py * self.pixel_delta_v)
    }
    pub fn render(&self, world: &BVHNode, new_color: &mut Color, i: usize, j: usize) {
        for _ in 0..self.samples_per_pixel {
            let r = self.get_ray(i, j);
            let color = self.ray_color(&r, world, self.max_depth);
            *new_color += color;
        }
    }
    pub fn sky_color(r: &Ray) -> Vec3 {
        let unit_direction = r.dir.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
    }

    pub fn ray_color(&self, r: &Ray, world: &BVHNode, depth: usize) -> Color {
        if depth == 0 {
            return ZERO;
        }
        if let Some(rec) = hit_world(&r, &world) {
            rec.mat_ptr.emitted(&rec.uv, &rec.p)
                + if let Some((attenuation, scattered)) = rec.mat_ptr.scatter(&r, &rec) {
                    mul(&attenuation, &self.ray_color(&scattered, world, depth - 1))
                } else {
                    ZERO
                }
        } else {
            self.background
            //Self::sky_color(&r)
        }
    }
}
