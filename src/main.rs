use indicatif::ProgressBar;
use rayon::prelude::*;
use raytracer::camera::Camera;
use raytracer::hittable::*;
use raytracer::material::BLACK_COLOR;
use raytracer::scene::*;
use raytracer::util::*;
use std::io::Write;
use std::sync::{Arc, Mutex};

static IMAGE_WIDTH: usize = 800;
static IMAGE_HEIGHT: usize = 800;
static CAMERA_SAMPLE: usize = 2000;
static RAY_DEPTH: usize = 40;
#[inline]
pub fn write_color(v: &Vec3) -> String {
    let v = v / (CAMERA_SAMPLE as f64);
    let r = clamp_one(v.x.sqrt());
    let g = clamp_one(v.y.sqrt());
    let b = clamp_one(v.z.sqrt());
    format!(
        "{} {} {}\n",
        (255.999 * r) as u8,
        (255.999 * g) as u8,
        (255.999 * b) as u8
    )
}
async fn progress_manager(progress: Arc<Mutex<usize>>) {
    let mut delay = tokio::time::interval(std::time::Duration::from_secs(1));
    let pb = ProgressBar::new(100);
    loop {
        delay.tick().await;

        let progress = *progress.lock().unwrap();

        if progress >= IMAGE_HEIGHT * IMAGE_WIDTH - 1 {
            pb.finish();
            break;
        } else {
            let pos = (progress as f64 / (IMAGE_HEIGHT * IMAGE_WIDTH) as f64 * 100.0) as u64;
            pb.set_position(pos);
        }
    }
}

#[tokio::main]
async fn main() {
    let mut file = std::fs::File::create("pic.ppm").expect("create failed");
    write!(file, "P3\n{IMAGE_WIDTH} {IMAGE_HEIGHT}\n255\n").expect("write failed");
    let aspect_ratio: f64 = IMAGE_WIDTH as f64 / IMAGE_HEIGHT as f64;
    let (world, cam) = final_scene(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_SAMPLE, RAY_DEPTH);
    let world = BVHNode::bvh_node_build_root(world);
    let mut output: Vec<Vec3> = vec![ZERO; IMAGE_HEIGHT * IMAGE_WIDTH];
    let progress = Arc::new(Mutex::new(0usize));
    let timer = tokio::spawn(progress_manager(progress.clone()));
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(x, new_color)| {
            let j = x / IMAGE_WIDTH;
            let i = x % IMAGE_WIDTH;
            cam.render(&world, new_color, i, j);
            *progress.lock().unwrap() += 1;
        });

    for j in 0..IMAGE_HEIGHT {
        for i in 0..IMAGE_WIDTH {
            write!(file, "{}", write_color(&output[j * IMAGE_WIDTH + i])).unwrap();
        }
    }
    timer.await.unwrap();
    println!("Finish!");
    /*
    //shutdown automatically
    use tokio::process::Command;
    Command::new("shutdown")
        .args(&["-s","-t","60"])
        .spawn().expect("shutdown command failed to start")
        .wait()
        .await
        .expect("shutdown command failed to run");

     */
}
