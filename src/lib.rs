pub use complex::*;
pub use quaternion::*;
pub use vec::*;

pub mod complex;
pub mod quaternion;
pub mod vec;

const F32_EPS: f32 = 1e-5;
const F64_EPS: f64 = 1e-12;

#[cfg(test)]
mod tests {
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub struct Pcg32 {
        state: u64,
        stream: u64,
    }

    const PCG32_DEFAULT_STATE: u64 = 0x853c49e6748fea9b;
    const PCG32_DEFAULT_STREAM: u64 = 0xda3e39cb94b95bdb;
    const PCG32_MULTIPLIER: u64 = 0x5851f42d4c957f2d;

    impl Pcg32 {
        #[inline]
        #[must_use]
        pub fn next_u32(&mut self) -> u32 {
            let old_state = self.state;
            self.state = old_state
                .wrapping_mul(PCG32_MULTIPLIER)
                .wrapping_add(self.stream);
            let xor_shifted = (((old_state >> 18u64) ^ old_state) >> 27u64) as u32;
            let rot = (old_state >> 59u64) as u32;
            (xor_shifted >> rot) | (xor_shifted << ((!rot).wrapping_add(1u32) & 31u32))
        }

        #[inline]
        #[must_use]
        pub fn next_f32(&mut self) -> f32 {
            let u = (self.next_u32() >> 9u32) | 0x3f800000u32;
            f32::from_bits(u) - 1.0
        }
    }

    impl std::default::Default for Pcg32 {
        #[inline]
        fn default() -> Self {
            Self {
                state: PCG32_DEFAULT_STATE,
                stream: PCG32_DEFAULT_STREAM,
            }
        }
    }

    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn check_vector_eps(r: Vec3f32, e: Vec3f32, eps: f32) {
        float_cmp::assert_approx_eq!(f32, r.x(), e.x(), epsilon = eps);
        float_cmp::assert_approx_eq!(f32, r.y(), e.y(), epsilon = eps);
        float_cmp::assert_approx_eq!(f32, r.z(), e.z(), epsilon = eps);
    }

    fn check_vector(r: Vec3f32, e: Vec3f32) {
        check_vector_eps(r, e, 0.00001);
    }

    fn check_quaternion(r: Quaternionf32, e: Quaternionf32) {
        float_cmp::assert_approx_eq!(f32, r.scalar, e.scalar);
        check_vector_eps(
            r.complex,
            e.complex,
            float_cmp::F32Margin::default().epsilon,
        );
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

    #[test]
    fn test_euler_angles() {
        const TESTS: usize = 10000;

        let mut pcg32 = Pcg32::default();

        let generate_angles = |rng: &mut Pcg32| {
            use std::f32::consts::PI;
            (
                PI * rng.next_f32(),
                PI * rng.next_f32(),
                PI * rng.next_f32(),
            )
        };

        let sample_sphere = |rng: &mut Pcg32| {
            let y = 1.0 - 2.0 * rng.next_f32();
            let r = (1.0 - y * y).max(0.0).sqrt();
            let phi = std::f32::consts::TAU * rng.next_f32();
            let (s, c) = phi.sin_cos();
            Vec3f32::new(r * c, y, r * s)
        };

        let compare_dot = |dot: f32| {
            // The precision here is arbitrary, we want to make sure the two final vectors are close enough
            assert!((dot - 1.0).abs() <= 0.00005);
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

            let r_euler = match q.extract_euler_angles(EulerOrder::XYZ) {
                EulerDecompositionf32::Normal(t0, t1, t2) => (Quaternionf32::z_rotation(t2)
                    * Quaternionf32::y_rotation(t1)
                    * Quaternionf32::x_rotation(t0))
                .normalised()
                .rotate(v),
                EulerDecompositionf32::Singularity(t0, t1) => (Quaternionf32::y_rotation(t1)
                    * Quaternionf32::x_rotation(t0))
                .normalised()
                .rotate(v),
            };

            compare_dot(r_euler.dot(r));
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

            let r_euler = match q.extract_euler_angles(EulerOrder::YZX) {
                EulerDecompositionf32::Normal(t0, t1, t2) => (Quaternionf32::x_rotation(t2)
                    * Quaternionf32::z_rotation(t1)
                    * Quaternionf32::y_rotation(t0))
                .normalised()
                .rotate(v),
                EulerDecompositionf32::Singularity(t0, t1) => (Quaternionf32::z_rotation(t1)
                    * Quaternionf32::y_rotation(t0))
                .normalised()
                .rotate(v),
            };

            compare_dot(r_euler.dot(r));
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

            let r_euler = match q.extract_euler_angles(EulerOrder::ZXY) {
                EulerDecompositionf32::Normal(t0, t1, t2) => (Quaternionf32::y_rotation(t2)
                    * Quaternionf32::x_rotation(t1)
                    * Quaternionf32::z_rotation(t0))
                .normalised()
                .rotate(v),
                EulerDecompositionf32::Singularity(t0, t1) => (Quaternionf32::x_rotation(t1)
                    * Quaternionf32::z_rotation(t0))
                .normalised()
                .rotate(v),
            };

            compare_dot(r_euler.dot(r));
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

            let r_euler = match q.extract_euler_angles(EulerOrder::ZYX) {
                EulerDecompositionf32::Normal(t0, t1, t2) => (Quaternionf32::x_rotation(t2)
                    * Quaternionf32::y_rotation(t1)
                    * Quaternionf32::z_rotation(t0))
                .normalised()
                .rotate(v),
                EulerDecompositionf32::Singularity(t0, t1) => (Quaternionf32::y_rotation(t1)
                    * Quaternionf32::z_rotation(t0))
                .normalised()
                .rotate(v),
            };

            compare_dot(r_euler.dot(r));
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

            let r_euler = match q.extract_euler_angles(EulerOrder::XZY) {
                EulerDecompositionf32::Normal(t0, t1, t2) => (Quaternionf32::y_rotation(t2)
                    * Quaternionf32::z_rotation(t1)
                    * Quaternionf32::x_rotation(t0))
                .normalised()
                .rotate(v),
                EulerDecompositionf32::Singularity(t0, t1) => (Quaternionf32::z_rotation(t1)
                    * Quaternionf32::x_rotation(t0))
                .normalised()
                .rotate(v),
            };

            compare_dot(r_euler.dot(r));
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

            let r_euler = match q.extract_euler_angles(EulerOrder::YXZ) {
                EulerDecompositionf32::Normal(t0, t1, t2) => (Quaternionf32::z_rotation(t2)
                    * Quaternionf32::x_rotation(t1)
                    * Quaternionf32::y_rotation(t0))
                .normalised()
                .rotate(v),
                EulerDecompositionf32::Singularity(t0, t1) => (Quaternionf32::x_rotation(t1)
                    * Quaternionf32::y_rotation(t0))
                .normalised()
                .rotate(v),
            };

            compare_dot(r_euler.dot(r));
        }
    }

    #[test]
    fn test_rotation_to() {
        let q0 = Quaternionf32::identity_rotation();
        let q1 = -Quaternionf32::identity_rotation();
        check_quaternion(q0.rotation_to(q1), -Quaternionf32::identity_rotation());

        let q0 = Quaternionf32::identity_rotation();
        let q1 = Quaternionf32::x_rotation(FRAC_PI_2 + FRAC_PI_4);
        check_quaternion(q0.rotation_to(q1), q1);

        let q0 = Quaternionf32::x_rotation(FRAC_PI_4);
        let q1 = Quaternionf32::x_rotation(PI + FRAC_PI_4);
        check_quaternion(q0.rotation_to(q1), Quaternionf32::x_rotation(PI));
    }
}
