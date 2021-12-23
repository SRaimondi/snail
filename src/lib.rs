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

    #[derive(Copy, Clone)]
    pub struct Pcg32 {
        state: u64,
        stream: u64,
    }

    impl Pcg32 {
        fn next_u32(&mut self) -> u32 {
            const PCG32_MULTIPLIER: u64 = 0x5851f42d4c957f2d;
            let old_state = self.state;
            self.state = old_state
                .wrapping_mul(PCG32_MULTIPLIER)
                .wrapping_add(self.stream);
            let xor_shifted = (((old_state >> 18u64) ^ old_state) >> 27u64) as u32;
            let rot = (old_state >> 59u64) as u32;
            (xor_shifted >> rot) | (xor_shifted << ((!rot).wrapping_add(1u32) & 31u32))
        }

        fn next_f32(&mut self) -> f32 {
            let u = (self.next_u32() >> 9u32) | 0x3f800000u32;
            f32::from_bits(u) - 1.0
        }
    }

    impl std::default::Default for Pcg32 {
        #[inline]
        fn default() -> Self {
            const PCG32_DEFAULT_STATE: u64 = 0x853c49e6748fea9b;
            const PCG32_DEFAULT_STREAM: u64 = 0xda3e39cb94b95bdb;
            Self {
                state: PCG32_DEFAULT_STATE,
                stream: PCG32_DEFAULT_STREAM,
            }
        }
    }

    fn sample_sphere(rng: &mut Pcg32) -> Vec3f32 {
        let y = 1.0 - 2.0 * rng.next_f32();
        let r = (1.0 - y * y).max(0.0).sqrt();
        let phi = TAU * rng.next_f32();
        let (s, c) = phi.sin_cos();
        Vec3f32::new(r * c, y, r * s)
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
        v.approx_eq(v1);
    }

    #[test]
    fn test_rotate() {
        let q = Quaternionf32::from_rotation(PI, Vec3f32::new(0.0, 1.0, 0.0));
        q.rotate(Vec3f32::UNIT_X).approx_eq(-Vec3f32::UNIT_X);

        assert!(Quaternionf32::x_rotation(FRAC_PI_4)
            .rotate(Vec3f32::UNIT_Y)
            .approx_eq(Vec3f32::new(0.0, 1.0, 1.0).normalised()),);
        assert!(Quaternionf32::x_rotation(-FRAC_PI_4)
            .rotate(Vec3f32::UNIT_Y)
            .approx_eq(Vec3f32::new(0.0, 1.0, -1.0).normalised()));
        assert!(Quaternionf32::x_rotation(PI)
            .rotate(Vec3f32::UNIT_Y)
            .approx_eq(-Vec3f32::UNIT_Y));

        let v = Vec3f32::broadcast(1.0).normalised();
        assert!(Quaternionf32::y_rotation(PI)
            .rotate(v)
            .approx_eq(Vec3f32::new(-1.0, 1.0, -1.0).normalised()),);
        assert!(Quaternionf32::z_rotation(FRAC_PI_2)
            .rotate(v)
            .approx_eq(Vec3f32::new(-1.0, 1.0, 1.0).normalised()),);
        assert!(Quaternionf32::z_rotation(-FRAC_PI_2)
            .rotate(v)
            .approx_eq(Vec3f32::new(1.0, -1.0, 1.0).normalised()),);
        assert!(Quaternionf32::z_rotation(PI)
            .rotate(v)
            .approx_eq(Vec3f32::new(-1.0, -1.0, 1.0).normalised()),);
        assert!(
            (Quaternionf32::x_rotation(FRAC_PI_2) * Quaternionf32::y_rotation(FRAC_PI_2))
                .rotate(Vec3f32::UNIT_X)
                .approx_eq(Vec3f32::UNIT_Y),
        );
        assert!(
            (Quaternionf32::z_rotation(FRAC_PI_2) * Quaternionf32::y_rotation(-FRAC_PI_4))
                .rotate(Vec3f32::UNIT_X)
                .approx_eq(Vec3f32::new(0.0, 1.0, 1.0).normalised()),
        );
    }

    #[test]
    fn test_euler_angles() {
        const TESTS: usize = 10000;

        let mut pcg32 = Pcg32::default();

        let generate_angles = |rng: &mut Pcg32| {
            (
                PI * rng.next_f32(),
                PI * rng.next_f32(),
                PI * rng.next_f32(),
            )
        };

        let check_dot = |a: Vec3f32, b: Vec3f32| {
            // The precision here is arbitrary, we want to make sure the two final vectors are close enough
            assert!(a.dot(b).approx_eq_rel_eps(1.0, 0.00005));
        };

        // XYZ order
        for _ in 0..TESTS {
            let (x_angle, y_angle, z_angle) = generate_angles(&mut pcg32);
            let x_q = Quaternionf32::x_rotation(x_angle);
            let y_q = Quaternionf32::y_rotation(y_angle);
            let z_q = Quaternionf32::z_rotation(z_angle);

            let q = (z_q * y_q * x_q).normalised();
            let v = sample_sphere(&mut pcg32);
            let r = q.rotate(v);

            let (rx, ry, rz) = q.extract_euler_angles(EulerOrder::XYZ).into();
            let res = (Quaternionf32::z_rotation(rz)
                * Quaternionf32::y_rotation(ry)
                * Quaternionf32::x_rotation(rx))
            .normalised()
            .rotate(v);
            check_dot(res, r);
        }

        // YZX order
        for _ in 0..TESTS {
            let (x_angle, y_angle, z_angle) = generate_angles(&mut pcg32);
            let x_q = Quaternionf32::x_rotation(x_angle);
            let y_q = Quaternionf32::y_rotation(y_angle);
            let z_q = Quaternionf32::z_rotation(z_angle);

            let q = (x_q * z_q * y_q).normalised();

            let v = sample_sphere(&mut pcg32);
            let r = q.rotate(v);

            let (ry, rz, rx) = q.extract_euler_angles(EulerOrder::YZX).into();
            let res = (Quaternionf32::x_rotation(rx)
                * Quaternionf32::z_rotation(rz)
                * Quaternionf32::y_rotation(ry))
            .normalised()
            .rotate(v);
            check_dot(res, r);
        }

        // ZXY order
        for _ in 0..TESTS {
            let (x_angle, y_angle, z_angle) = generate_angles(&mut pcg32);
            let x_q = Quaternionf32::x_rotation(x_angle);
            let y_q = Quaternionf32::y_rotation(y_angle);
            let z_q = Quaternionf32::z_rotation(z_angle);

            let q = (y_q * x_q * z_q).normalised();

            let v = sample_sphere(&mut pcg32);
            let r = q.rotate(v);

            let (rz, rx, ry) = q.extract_euler_angles(EulerOrder::ZXY).into();
            let res = (Quaternionf32::y_rotation(ry)
                * Quaternionf32::x_rotation(rx)
                * Quaternionf32::z_rotation(rz))
            .normalised()
            .rotate(v);
            check_dot(res, r);
        }

        // ZYX order
        for _ in 0..TESTS {
            let (x_angle, y_angle, z_angle) = generate_angles(&mut pcg32);
            let x_q = Quaternionf32::x_rotation(x_angle);
            let y_q = Quaternionf32::y_rotation(y_angle);
            let z_q = Quaternionf32::z_rotation(z_angle);

            let q = (x_q * y_q * z_q).normalised();

            let v = sample_sphere(&mut pcg32);
            let r = q.rotate(v);

            let (rz, ry, rx) = q.extract_euler_angles(EulerOrder::ZYX).into();
            let res = (Quaternionf32::x_rotation(rx)
                * Quaternionf32::y_rotation(ry)
                * Quaternionf32::z_rotation(rz))
            .normalised()
            .rotate(v);
            check_dot(res, r);
        }

        // XZY order
        for _ in 0..TESTS {
            let (x_angle, y_angle, z_angle) = generate_angles(&mut pcg32);
            let x_q = Quaternionf32::x_rotation(x_angle);
            let y_q = Quaternionf32::y_rotation(y_angle);
            let z_q = Quaternionf32::z_rotation(z_angle);

            let q = (y_q * z_q * x_q).normalised();

            let v = sample_sphere(&mut pcg32);
            let r = q.rotate(v);

            let (rx, rz, ry) = q.extract_euler_angles(EulerOrder::XZY).into();
            let res = (Quaternionf32::y_rotation(ry)
                * Quaternionf32::z_rotation(rz)
                * Quaternionf32::x_rotation(rx))
            .normalised()
            .rotate(v);
            check_dot(res, r);
        }

        // YXZ order
        for _ in 0..TESTS {
            let (x_angle, y_angle, z_angle) = generate_angles(&mut pcg32);
            let x_q = Quaternionf32::x_rotation(x_angle);
            let y_q = Quaternionf32::y_rotation(y_angle);
            let z_q = Quaternionf32::z_rotation(z_angle);

            let q = (z_q * x_q * y_q).normalised();

            let v = sample_sphere(&mut pcg32);
            let r = q.rotate(v);

            let (ry, rx, rz) = q.extract_euler_angles(EulerOrder::YXZ).into();
            let res = (Quaternionf32::z_rotation(rz)
                * Quaternionf32::x_rotation(rx)
                * Quaternionf32::y_rotation(ry))
            .normalised()
            .rotate(v);
            check_dot(res, r);
        }
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
                assert!(q.rotate(Vec3f32::UNIT_X).approx_eq(s));
                assert!(q.rotate(Vec3f32::UNIT_Y).approx_eq(n));
                assert!(q.rotate(Vec3f32::UNIT_Z).approx_eq(t));
            }
        }
    }
}
