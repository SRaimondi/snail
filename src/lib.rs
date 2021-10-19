pub use complex::*;
pub use quaternion::*;
pub use vec::*;

pub mod complex;
pub mod quaternion;
pub mod vec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_two_vectors() {
        // Rotation between two non parallel vectors
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let v1 = Vec3f32::new(0.0, 1.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v1);
        let v = q.rotate(v0);
        float_cmp::assert_approx_eq!(f32, v.x(), v1.x());
        float_cmp::assert_approx_eq!(f32, v.y(), v1.y());
        float_cmp::assert_approx_eq!(f32, v.z(), v1.z());
        // Rotation between the same vector
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v0);
        let v = q.rotate(v0);
        float_cmp::assert_approx_eq!(f32, v.x(), v0.x());
        float_cmp::assert_approx_eq!(f32, v.y(), v0.y());
        float_cmp::assert_approx_eq!(f32, v.z(), v0.z());
        // Rotation between opposite vectors
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let v1 = Vec3f32::new(-1.0, 0.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v1);
        let v = q.rotate(v0);
        float_cmp::assert_approx_eq!(f32, v.x(), v1.x());
        float_cmp::assert_approx_eq!(f32, v.y(), v1.y());
        float_cmp::assert_approx_eq!(f32, v.z(), v1.z());
    }

    #[test]
    fn test_rotate() {
        use std::f32::consts::PI;
        let q = Quaternionf32::from_rotation(PI, Vec3f32::new(0.0, 1.0, 0.0));
        let v = Vec3f32::new(1.0, 0.0, 0.0);
        let v_r = q.rotate(v);
        float_cmp::assert_approx_eq!(f32, v_r.x(), -1.0);
        float_cmp::assert_approx_eq!(f32, v_r.y(), 0.0);
        float_cmp::assert_approx_eq!(f32, v_r.z(), 0.0);
    }

    #[test]
    fn test_euler() {
        use std::f32::consts::FRAC_PI_4;
        let q = Quaternionf32::from_rotation(FRAC_PI_4, Vec3f32::new(1.2, -4.5, 8.0).normalised());
        // Rotate using quaternion
        let v = Vec3f32::new(10.0, 15.0, -4.0);
        let v_rotated = q.rotate(v);
        // Try to extract euler angles
        let angles = q.extract_euler();
        // Rotate v in order using the Euler angles
        let v_rot_z = Quaternionf32::from_rotation(angles.0, Vec3f32::new(0.0, 0.0, 1.0)).rotate(v);
        let v_rot_y =
            Quaternionf32::from_rotation(angles.1, Vec3f32::new(0.0, 1.0, 0.0)).rotate(v_rot_z);
        let v_rot_x =
            Quaternionf32::from_rotation(angles.2, Vec3f32::new(1.0, 0.0, 0.0)).rotate(v_rot_y);
        // Because of the operations, it's hard to get exactly the same values, we use a larger
        // tolerance compared to the default one
        float_cmp::assert_approx_eq!(f32, v_rotated.x(), v_rot_x.x(), epsilon = 0.00001);
        float_cmp::assert_approx_eq!(f32, v_rotated.y(), v_rot_x.y(), epsilon = 0.00001);
        float_cmp::assert_approx_eq!(f32, v_rotated.z(), v_rot_x.z(), epsilon = 0.00001);
    }
}
