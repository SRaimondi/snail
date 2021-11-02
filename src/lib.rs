pub use complex::*;
pub use quaternion::*;
pub use vec::*;

pub mod complex;
pub mod quaternion;
pub mod vec;

#[cfg(test)]
mod tests {
    use super::*;

    fn check_vector(r: Vec3f32, e: Vec3f32) {
        float_cmp::assert_approx_eq!(f32, r.x(), e.x());
        float_cmp::assert_approx_eq!(f32, r.y(), e.y());
        float_cmp::assert_approx_eq!(f32, r.z(), e.z());
    }

    #[test]
    fn test_from_two_vectors() {
        // Rotation between two non parallel vectors
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let v1 = Vec3f32::new(0.0, 1.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v1);
        let v = q.rotate(v0);
        check_vector(v, v1);

        // Rotation between the same vector
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v0);
        let v = q.rotate(v0);
        check_vector(v, v0);

        // Rotation between opposite vectors
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let v1 = Vec3f32::new(-1.0, 0.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v1);
        let v = q.rotate(v0);
        check_vector(v, v1);
    }

    #[test]
    fn test_rotate() {
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        let q = Quaternionf32::from_rotation(PI, Vec3f32::new(0.0, 1.0, 0.0));
        check_vector(q.rotate(Vec3f32::EX), -Vec3f32::EX);

        check_vector(
            Quaternionf32::x_rotation(FRAC_PI_4).rotate(Vec3f32::EY),
            Vec3f32::new(0.0, 1.0, 1.0).normalised(),
        );
        check_vector(
            Quaternionf32::x_rotation(-FRAC_PI_4).rotate(Vec3f32::EY),
            Vec3f32::new(0.0, 1.0, -1.0).normalised(),
        );
        check_vector(
            Quaternionf32::x_rotation(PI).rotate(Vec3f32::EY),
            -Vec3f32::EY,
        );
        let v = Vec3f32::broadcast(1.0).normalised();
        check_vector(
            Quaternionf32::y_rotation(PI).rotate(v),
            Vec3f32::new(-1.0, 1.0, -1.0).normalised(),
        );
        check_vector(
            Quaternionf32::z_rotation(FRAC_PI_2).rotate(v),
            Vec3f32::new(-1.0, 1.0, 1.0).normalised(),
        );
        check_vector(
            Quaternionf32::z_rotation(-FRAC_PI_2).rotate(v),
            Vec3f32::new(1.0, -1.0, 1.0).normalised(),
        );
        check_vector(
            Quaternionf32::z_rotation(PI).rotate(v),
            Vec3f32::new(-1.0, -1.0, 1.0).normalised(),
        );
        check_vector(
            (Quaternionf32::x_rotation(FRAC_PI_2) * Quaternionf32::y_rotation(FRAC_PI_2))
                .rotate(Vec3f32::EX),
            Vec3f32::EY,
        );
        check_vector(
            (Quaternionf32::z_rotation(FRAC_PI_2) * Quaternionf32::y_rotation(-FRAC_PI_4))
                .rotate(Vec3f32::EX),
            Vec3f32::new(0.0, 1.0, 1.0).normalised(),
        );
    }
}
