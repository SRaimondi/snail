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
}
