pub use comparison::*;
pub use complex::*;
pub use quaternion::*;
pub use vec::*;

pub mod comparison;
pub mod complex;
pub mod quaternion;
pub mod vec;

#[cfg(test)]
mod tests {
    use super::*;

    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    use pcg32::Pcg32;

    fn sample_sphere(rng: &mut Pcg32) -> Vec3f32 {
        let y = 1.0 - 2.0 * rng.next_f32();
        let r = (1.0 - y * y).max(0.0).sqrt();
        let phi = TAU * rng.next_f32();
        let (s, c) = phi.sin_cos();
        Vec3f32::new(r * c, y, r * s)
    }

    #[test]
    fn test_angle_with() {
        let a = Vec2f32::unit_x();
        let b = Vec2f32::unit_y();
        assert!(a.angle_with(b).approx_eq(FRAC_PI_2));
        assert!(a.angle_with(-b).approx_eq(FRAC_PI_2));
        assert!(a.angle_with(Vec2f32::broadcast(10.0)).approx_eq(FRAC_PI_4));
        assert!(a
            .angle_with(Vec2f32::new(-5.0, 5.0))
            .approx_eq(3.0 * FRAC_PI_4));
        assert!(a.angle_with(-a).approx_eq(PI));
    }

    #[test]
    fn test_from_two_vectors() {
        // Rotation between two non parallel vectors
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let v1 = Vec3f32::new(0.0, 1.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v1);
        let v = q.rotate(v0);
        assert!(v.approx_eq(v1));

        // Rotation between the same vector
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v0);
        let v = q.rotate(v0);
        assert!(v.approx_eq(v0));

        // Rotation between opposite vectors
        let v0 = Vec3f32::new(1.0, 0.0, 0.0);
        let v1 = Vec3f32::new(-1.0, 0.0, 0.0);
        let q = Quaternionf32::from_two_vectors_normalised(v0, v1);
        let v = q.rotate(v0);
        assert!(v.approx_eq(v1));
    }

    #[test]
    fn test_rotate() {
        let q = Quaternionf32::from_rotation(PI, Vec3f32::new(0.0, 1.0, 0.0));
        q.rotate(Vec3f32::unit_x()).approx_eq(-Vec3f32::unit_x());

        assert!(Quaternionf32::x_rotation(FRAC_PI_4)
            .rotate(Vec3f32::unit_y())
            .approx_eq(Vec3f32::new(0.0, 1.0, 1.0).normalised()));
        assert!(Quaternionf32::x_rotation(-FRAC_PI_4)
            .rotate(Vec3f32::unit_y())
            .approx_eq(Vec3f32::new(0.0, 1.0, -1.0).normalised()));
        assert!(Quaternionf32::x_rotation(PI)
            .rotate(Vec3f32::unit_y())
            .approx_eq(-Vec3f32::unit_y()));

        let v = Vec3f32::broadcast(1.0).normalised();
        assert!(Quaternionf32::y_rotation(PI)
            .rotate(v)
            .approx_eq(Vec3f32::new(-1.0, 1.0, -1.0).normalised()));
        assert!(Quaternionf32::z_rotation(FRAC_PI_2)
            .rotate(v)
            .approx_eq(Vec3f32::new(-1.0, 1.0, 1.0).normalised()));
        assert!(Quaternionf32::z_rotation(-FRAC_PI_2)
            .rotate(v)
            .approx_eq(Vec3f32::new(1.0, -1.0, 1.0).normalised()));
        assert!(Quaternionf32::z_rotation(PI)
            .rotate(v)
            .approx_eq(Vec3f32::new(-1.0, -1.0, 1.0).normalised()));
        assert!(
            (Quaternionf32::x_rotation(FRAC_PI_2) * Quaternionf32::y_rotation(FRAC_PI_2))
                .rotate(Vec3f32::unit_x())
                .approx_eq(Vec3f32::unit_y())
        );
        assert!(
            (Quaternionf32::z_rotation(FRAC_PI_2) * Quaternionf32::y_rotation(-FRAC_PI_4))
                .rotate(Vec3f32::unit_x())
                .approx_eq(Vec3f32::new(0.0, 1.0, 1.0).normalised())
        );
    }

    #[test]
    fn test_euler() {
        let q = Quaternionf32::from_rotation(PI / 10.0, Vec3f32::new(1.0, 0.0, 1.0).normalised());
        let euler = q.extract_euler_zyx();
        assert!(euler.0.approx_eq(0.22035267));
        assert!(euler.1.approx_eq(0.02447425));
        assert!(euler.2.approx_eq(0.22035267));
    }

    #[test]
    fn test_rotation_to() {
        let q0 = Quaternionf32::IDENTITY;
        let q1 = -Quaternionf32::IDENTITY;
        assert!(q0.rotation_to(q1).approx_eq(-Quaternionf32::IDENTITY));

        let q0 = Quaternionf32::IDENTITY;
        let q1 = Quaternionf32::x_rotation(FRAC_PI_2 + FRAC_PI_4);
        assert!(q0.rotation_to(q1).approx_eq(q1));

        let q0 = Quaternionf32::x_rotation(FRAC_PI_4);
        let q1 = Quaternionf32::x_rotation(PI + FRAC_PI_4);
        assert!(q0.rotation_to(q1).approx_eq(Quaternionf32::x_rotation(PI)));
    }

    #[test]
    fn test_quaternion_from_matrix() {
        const TESTS: usize = 10_000;
        let mut rng = Pcg32::default();
        for _ in 0..TESTS {
            let s = sample_sphere(&mut rng);
            const ROTATIONS: usize = 1_000;
            for _ in 0..ROTATIONS {
                let angle = rng.next_f32() * TAU;
                let n = Quaternionf32::from_rotation(angle, s).rotate(s.compute_perpendicular());
                let t = s.cross(n);
                let q = Quaternionf32::from_frame(s, n, t);
                // Check we get the original axes
                assert!(q.rotate(Vec3f32::unit_x()).approx_eq(s));
                assert!(q.rotate(Vec3f32::unit_y()).approx_eq(n));
                assert!(q.rotate(Vec3f32::unit_z()).approx_eq(t));
            }
        }
    }
}
