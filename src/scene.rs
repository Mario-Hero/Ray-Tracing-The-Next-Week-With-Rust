use crate::camera::Camera;
use crate::hittable::*;
use crate::material::*;
use crate::util::{length, random_vec3, Point, Vec3, WORLD_UP, ZERO};
use rand::random;
use std::sync::Arc;
macro_rules! green {
    () => {
        Arc::new(Lambert {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.12, 0.45, 0.15),
            }),
        })
    };
}
macro_rules! red {
    () => {
        Arc::new(Lambert {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.65, 0.05, 0.05),
            }),
        })
    };
}
macro_rules! white {
    () => {
        Arc::new(Lambert {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.73, 0.73, 0.73),
            }),
        })
    };
}
macro_rules! light {
    () => {
        Arc::new(Emit {
            texture: Arc::new(SolidColor {
                color_value: WHITE_COLOR * 15.0,
            }),
        })
    };
}
pub fn new_box(a: Point, b: Point, mat: MatPointer) -> HittableList {
    let mut instance = HittableList::new();
    let min = Vec3::new(a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2]));
    let max = Vec3::new(a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2]));
    let dx = Vec3::new(max.x - min.x, 0., 0.);
    let dy = Vec3::new(0., max.y - min.y, 0.);
    let dz = Vec3::new(0., 0., max.z - min.z);
    instance.add(Arc::new(Quad::new(
        &Vec3::new(min.x, min.y, max.z),
        &dx,
        &dy,
        mat.clone(),
    ))); // front
    instance.add(Arc::new(Quad::new(
        &Vec3::new(max.x, min.y, max.z),
        &-dz,
        &dy,
        mat.clone(),
    ))); // right
    instance.add(Arc::new(Quad::new(
        &Vec3::new(max.x, min.y, min.z),
        &-dx,
        &dy,
        mat.clone(),
    ))); // back
    instance.add(Arc::new(Quad::new(
        &Vec3::new(min.x, min.y, min.z),
        &dz,
        &dy,
        mat.clone(),
    ))); // left
    instance.add(Arc::new(Quad::new(
        &Vec3::new(min.x, max.y, max.z),
        &dx,
        &-dz,
        mat.clone(),
    ))); // top
    instance.add(Arc::new(Quad::new(
        &Vec3::new(min.x, min.y, min.z),
        &dx,
        &dz,
        mat.clone(),
    ))); // bottom
    instance
}

pub fn perlin_scene(
    image_width: usize,
    image_height: usize,
    samples_per_pixel: usize,
    max_depth: usize,
) -> (HittableList, Camera) {
    let mut world = HittableList::new();

    let pertext = Arc::new(NoiseTexture::new(4.0));
    world.add(Arc::new(Sphere {
        center: Vec3::new(0., -1000., 0.),
        radius: 1000.0,
        mat_ptr: Arc::new(Lambert {
            albedo: pertext.clone(),
        }),
    }));
    world.add(Arc::new(Sphere {
        center: Vec3::new(0., 2., 0.),
        radius: 2.0,
        mat_ptr: Arc::new(Lambert { albedo: pertext }),
    }));
    let look_from = Vec3::new(13.0, 2.0, 3.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    let defocus_angle = 0.0;
    let cam = Camera::new(
        look_from,
        look_at,
        WORLD_UP,
        20.0,
        defocus_angle,
        Vec3::new(0.7, 0.7, 0.7),
        image_width,
        image_height,
        samples_per_pixel,
        max_depth,
    );
    (world, cam)
}

pub fn cornell_box_scene(
    image_width: usize,
    image_height: usize,
    samples_per_pixel: usize,
    max_depth: usize,
) -> (HittableList, Camera) {
    let mut world = HittableList::new();
    let h = 555.0;
    world.add(Arc::new(Quad::new(
        &Vec3::new(h, 0.0, 0.0),
        &Vec3::new(0.0, h, 0.0),
        &Vec3::new(0.0, 0.0, h),
        green!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(0.0, 0.0, 0.0),
        &Vec3::new(0.0, h, 0.0),
        &Vec3::new(0.0, 0.0, h),
        red!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(343., 554., 332.),
        &Vec3::new(-130., 0., 0.0),
        &Vec3::new(0.0, 0.0, -105.),
        light!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(0.0, 0.0, 0.0),
        &Vec3::new(h, 0., 0.0),
        &Vec3::new(0.0, 0.0, h),
        white!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(h, h, h),
        &Vec3::new(-h, 0., 0.0),
        &Vec3::new(0.0, 0.0, -h),
        white!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(0., 0., h),
        &Vec3::new(h, 0., 0.0),
        &Vec3::new(0.0, h, 0.),
        white!(),
    )));
    let mut box1: Arc<dyn Hittable> = Arc::new(new_box(
        Vec3::new(0., 0., 0.),
        Vec3::new(165., 330., 165.),
        white!(),
    ));
    let mut box2: Arc<dyn Hittable> = Arc::new(new_box(
        Vec3::new(0., 0., 0.),
        Vec3::new(165., 165., 165.),
        white!(),
    ));
    box1 = Arc::new(RotateY::new(box1, 15.0));
    box1 = Arc::new(Translate::new(box1, Vec3::new(265., 0., 295.)));

    box2 = Arc::new(RotateY::new(box2, -18.0));
    box2 = Arc::new(Translate::new(box2, Vec3::new(130., 0., 65.)));

    world.add(box1);
    world.add(box2);

    let look_from = Vec3::new(278.0, 278.0, -800.0);
    let look_at = Vec3::new(278.0, 278.0, 0.0);
    let defocus_angle = 0.0;
    let cam = Camera::new(
        look_from,
        look_at,
        WORLD_UP,
        40.0,
        defocus_angle,
        BLACK_COLOR,
        image_width,
        image_height,
        samples_per_pixel,
        max_depth,
    );
    (world, cam)
}

pub fn cornell_smoke_scene(
    image_width: usize,
    image_height: usize,
    samples_per_pixel: usize,
    max_depth: usize,
) -> (HittableList, Camera) {
    let mut world = HittableList::new();
    let h = 555.0;
    let light7: Arc<dyn Material> = Arc::new(Emit {
        texture: Arc::new(SolidColor {
            color_value: WHITE_COLOR * 7.0,
        }),
    });
    world.add(Arc::new(Quad::new(
        &Vec3::new(h, 0.0, 0.0),
        &Vec3::new(0.0, h, 0.0),
        &Vec3::new(0.0, 0.0, h),
        green!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(0.0, 0.0, 0.0),
        &Vec3::new(0.0, h, 0.0),
        &Vec3::new(0.0, 0.0, h),
        red!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(113., 554., 127.),
        &Vec3::new(330., 0., 0.0),
        &Vec3::new(0.0, 0.0, 305.),
        light7,
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(0.0, h, 0.0),
        &Vec3::new(h, 0., 0.0),
        &Vec3::new(0.0, 0.0, h),
        white!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(0., 0., 0.),
        &Vec3::new(h, 0., 0.0),
        &Vec3::new(0.0, 0.0, h),
        white!(),
    )));
    world.add(Arc::new(Quad::new(
        &Vec3::new(0., 0., h),
        &Vec3::new(h, 0., 0.0),
        &Vec3::new(0.0, h, 0.),
        white!(),
    )));
    let mut box1: Arc<dyn Hittable> = Arc::new(new_box(
        Vec3::new(0., 0., 0.),
        Vec3::new(165., 330., 165.),
        white!(),
    ));
    let mut box2: Arc<dyn Hittable> = Arc::new(new_box(
        Vec3::new(0., 0., 0.),
        Vec3::new(165., 165., 165.),
        white!(),
    ));
    box1 = Arc::new(RotateY::new(box1, 15.0));
    box1 = Arc::new(Translate::new(box1, Vec3::new(265., 0., 295.)));
    box1 = Arc::new(ConstantMedium::new(
        box1,
        0.01,
        Arc::new(SolidColor {
            color_value: BLACK_COLOR,
        }),
    ));

    box2 = Arc::new(RotateY::new(box2, -18.0));
    box2 = Arc::new(Translate::new(box2, Vec3::new(130., 0., 65.)));
    box2 = Arc::new(ConstantMedium::new(
        box2,
        0.01,
        Arc::new(SolidColor {
            color_value: WHITE_COLOR,
        }),
    ));
    world.add(box1);
    world.add(box2);
    let look_from = Vec3::new(278.0, 278.0, -800.0);
    let look_at = Vec3::new(278.0, 278.0, 0.0);
    let defocus_angle = 0.0;
    let cam = Camera::new(
        look_from,
        look_at,
        WORLD_UP,
        40.0,
        defocus_angle,
        BLACK_COLOR,
        image_width,
        image_height,
        samples_per_pixel,
        max_depth,
    );
    (world, cam)
}

pub fn random_scene(
    image_width: usize,
    image_height: usize,
    samples_per_pixel: usize,
    max_depth: usize,
) -> (HittableList, Camera) {
    let mut world = HittableList::new();

    world.add(Arc::new(Sphere {
        center: Vec3::new(0.0, -1000.0, 0.0),
        radius: 1000.0,
        mat_ptr: Arc::new(Lambert {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.5, 0.5, 0.5),
            }),
        }),
    }));
    for a in -11..11 {
        let a = a as f64;
        for b in -11..11 {
            let b = b as f64;
            let choose_mat = random::<f64>();
            let center = Vec3::new(a + 0.9 * random::<f64>(), 0.2, b + 0.9 * random::<f64>());
            if length(&(center - Vec3::new(4.0, 0.2, 0.0))) > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = Arc::new(SolidColor {
                        color_value: random_vec3(),
                    });
                    world.add(Arc::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: Arc::new(Lambert { albedo }),
                    }))
                } else if choose_mat < 0.95 {
                    let albedo = random_vec3() / 2.0 + Vec3::new(0.5, 0.5, 0.5);
                    let albedo = Arc::new(SolidColor {
                        color_value: albedo,
                    });
                    let fuzz = random::<f64>() / 2.0;
                    world.add(Arc::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: Arc::new(Metal { albedo, fuzz }),
                    }))
                } else {
                    world.add(Arc::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: Arc::new(Dielectric { ref_idx: 1.5 }),
                    }))
                }
            }
        }
    }
    world.add(Arc::new(Sphere {
        center: Vec3::new(0.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: Arc::new(Dielectric { ref_idx: 1.5 }),
    }));
    world.add(Arc::new(Sphere {
        center: Vec3::new(-4.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: Arc::new(Lambert {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.4, 0.2, 0.1),
            }),
        }),
    }));
    world.add(Arc::new(Sphere {
        center: Vec3::new(4.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: Arc::new(Metal {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.7, 0.6, 0.5),
            }),
            fuzz: 0.0,
        }),
    }));

    let look_from = Vec3::new(13.0, 2.0, 3.0);
    let look_at = ZERO;
    let defocus_angle = 0.6;
    let cam = Camera::new(
        look_from,
        look_at,
        WORLD_UP,
        20.0,
        defocus_angle,
        Vec3::new(0.7, 0.7, 0.7),
        image_width,
        image_height,
        samples_per_pixel,
        max_depth,
    );
    (world, cam)
}

pub fn final_scene(
    image_width: usize,
    image_height: usize,
    samples_per_pixel: usize,
    max_depth: usize,
) -> (HittableList, Camera) {
    let mut boxes1 = HittableList::new();
    let ground = Arc::new(Lambert {
        albedo: Arc::new(SolidColor {
            color_value: Vec3::new(0.48, 0.83, 0.53),
        }),
    });
    let boxes_per_side = 20;
    for i in 0..boxes_per_side {
        for j in 0..boxes_per_side {
            let w = 100.0;
            let x0 = -1000.0 + i as f64 * w;
            let z0 = -1000.0 + j as f64 * w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let y1 = random::<f64>() * 100.0 + 1.0;
            let z1 = z0 + w;
            boxes1.add(Arc::new(new_box(
                Vec3::new(x0, y0, z0),
                Vec3::new(x1, y1, z1),
                ground.clone(),
            )));
        }
    }
    let mut world = HittableList::new();
    world.add(Arc::new(boxes1));
    let light: Arc<dyn Material> = Arc::new(Emit {
        texture: Arc::new(SolidColor {
            color_value: WHITE_COLOR * 7.0,
        }),
    });
    world.add(Arc::new(Quad::new(
        &Vec3::new(123.0, 554.0, 147.0),
        &Vec3::new(300.0, 0.0, 0.0),
        &Vec3::new(0.0, 0.0, 265.0),
        light.clone(),
    )));
    let center0 = Vec3::new(400., 400., 200.);
    let center1 = center0 + Vec3::new(30., 0., 0.);
    let moving_sphere = Arc::new(MovingSphere {
        center0,
        center1,
        time0: 0.0,
        time1: 1.0,
        radius: 50.0,
        mat_ptr: Arc::new(Lambert {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.7, 0.3, 0.1),
            }),
        }),
    });
    world.add(moving_sphere);
    world.add(Arc::new(Sphere {
        center: Vec3::new(260.0, 150.0, 45.0),
        radius: 50.0,
        mat_ptr: Arc::new(Dielectric { ref_idx: 1.5 }),
    }));
    world.add(Arc::new(Sphere {
        center: Vec3::new(0.0, 150.0, 145.0),
        radius: 50.0,
        mat_ptr: Arc::new(Metal {
            albedo: Arc::new(SolidColor {
                color_value: Vec3::new(0.8, 0.8, 0.9),
            }),
            fuzz: 1.0,
        }),
    }));
    let boundary = Arc::new(Sphere {
        center: Vec3::new(360.0, 150.0, 145.0),
        radius: 70.0,
        mat_ptr: Arc::new(Dielectric { ref_idx: 1.5 }),
    });
    world.add(boundary.clone());
    world.add(Arc::new(ConstantMedium::new(
        boundary,
        0.2,
        Arc::new(SolidColor {
            color_value: Vec3::new(0.2, 0.4, 0.9),
        }),
    )));
    let boundary2 = Arc::new(Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 5000.0,
        mat_ptr: Arc::new(Dielectric { ref_idx: 1.5 }),
    });
    world.add(Arc::new(ConstantMedium::new(
        boundary2,
        0.0001,
        Arc::new(SolidColor {
            color_value: WHITE_COLOR,
        }),
    )));
    let emat = Arc::new(Lambert {
        albedo: Arc::new(Picture::new("earthmap.jpg")),
    });
    world.add(Arc::new(Sphere {
        center: Vec3::new(400.0, 200.0, 400.0),
        radius: 100.0,
        mat_ptr: emat,
    }));
    let pertext = Arc::new(NoiseTexture::new(0.1));
    world.add(Arc::new(Sphere {
        center: Vec3::new(220.0, 280.0, 300.0),
        radius: 80.0,
        mat_ptr: Arc::new(Lambert { albedo: pertext }),
    }));
    let mut boxes2 = HittableList::new();
    let white = Arc::new(Lambert {
        albedo: Arc::new(SolidColor::new(Vec3::new(0.73, 0.73, 0.73))),
    });
    let ns = 1000;
    for _ in 0..ns {
        boxes2.add(Arc::new(Sphere {
            center: Vec3::new(
                random::<f64>() * 165.0,
                random::<f64>() * 165.0,
                random::<f64>() * 165.0,
            ),
            radius: 10.0,
            mat_ptr: white.clone(),
        }));
    }
    world.add(Arc::new(Translate::new(
        Arc::new(RotateY::new(Arc::new(boxes2), 15.0)),
        Vec3::new(-100., 270., 395.),
    )));
    let cam = Camera::new(
        Vec3::new(478., 278., -600.),
        Vec3::new(278., 278., 0.),
        WORLD_UP,
        40.0,
        0.0,
        BLACK_COLOR,
        image_width,
        image_height,
        samples_per_pixel,
        max_depth,
    );
    (world, cam)
}
