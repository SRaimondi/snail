use crate::{ApproxEq, Vec3f32, Vec3f64};

use std::{
    f32, f64,
    ops::{Add, Div, Mul, Neg, Sub},
};

macro_rules! generate_quaternion {
    ($name:ident, $vec_name:ident, $euler_name:ident, $t:ty, $pi_2:expr, $eps:expr) => {
        /// Helper type for the euler decomposition.
        pub type $euler_name = ($t, $t, $t);

        /// q = q_scalar + complex.x * i + complex.y * j + complex.z * k.
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(C)]
        pub struct $name {
            pub scalar: $t,
            pub complex: $vec_name,
        }

        impl $name {
            /// Identity rotation quaternion.
            pub const IDENTITY: Self = Self::from_components(1.0, 0.0, 0.0, 0.0);

            /// Create new quaternion from the given components.
            #[inline(always)]
            pub const fn from_components(
                scalar: $t,
                complex_x: $t,
                complex_y: $t,
                complex_z: $t,
            ) -> Self {
                Self {
                    scalar,
                    complex: $vec_name::new(complex_x, complex_y, complex_z),
                }
            }

            /// Create new quaternion from the given scalar and complex parts.
            #[inline(always)]
            pub const fn new(scalar: $t, complex: $vec_name) -> Self {
                Self { scalar, complex }
            }

            /// Create pure quaternion from the given vector.
            #[inline(always)]
            pub const fn from_vector(v: $vec_name) -> Self {
                Self::new(0.0, v)
            }

            /// Create quaternion as rotation between two vectors.
            /// Panics in debug if the norm of either vector is not close enough to 1.0.
            #[inline(always)]
            pub fn from_two_vectors_normalised(from: $vec_name, to: $vec_name) -> Self {
                debug_assert!(from.norm().approx_eq(1.0));
                debug_assert!(to.norm().approx_eq(1.0));

                let c = from.dot(to);
                let q = if c < -1.0 + <$t as ApproxEq>::DEFAULT_ABSOLUTE_EPS {
                    Self::from_vector(from.compute_perpendicular())
                } else {
                    let axis = from.cross(to);
                    let s = (2.0 * (1.0 + c)).sqrt();
                    let inv_s = 1.0 / s;
                    Self::new(0.5 * s, inv_s * axis)
                };
                q.normalised()
            }

            /// Create quaternion as rotation between two vectors.
            #[inline(always)]
            pub fn from_two_vectors(from: $vec_name, to: $vec_name) -> Self {
                Self::from_two_vectors_normalised(from.normalised(), to.normalised())
            }

            /// Create rotation versor from the given angle and axis.
            /// Panics in debug if the norm of the axis is not 1.0.
            #[inline(always)]
            pub fn from_rotation_normalised(angle: $t, axis: $vec_name) -> Self {
                debug_assert!(axis.norm().approx_eq(1.0));
                let (s, c) = (angle / 2.0).sin_cos();
                Self::new(c, s * axis)
            }

            /// Create rotation versor for the given angle and axis.
            #[inline(always)]
            pub fn from_rotation(angle: $t, axis: $vec_name) -> Self {
                Self::from_rotation_normalised(angle, axis.normalised())
            }

            /// Create quaternion from given rotation matrix columns.
            /// Panics in debug if the length of the given columns is not one or they are
            /// not perpendicular.
            #[inline]
            pub fn from_frame(s: $vec_name, n: $vec_name, t: $vec_name) -> Self {
                debug_assert!(s.norm().approx_eq(1.0));
                debug_assert!(n.norm().approx_eq(1.0));
                debug_assert!(t.norm().approx_eq(1.0));
                debug_assert!(s.dot(n).approx_zero());
                debug_assert!(s.dot(t).approx_zero());

                // Naming similar to http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
                let (m00, m10, m20) = s.into();
                let (m01, m11, m21) = n.into();
                let (m02, m12, m22) = t.into();

                let trace = m00 + m11 + m22;
                let (w, x, y, z) = if trace > 0.0 {
                    let s = 2.0 * (1.0 + trace).sqrt();
                    (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
                } else if m00 > m11 && m00 > m22 {
                    let s = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
                    ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
                } else if m11 > m22 {
                    let s = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
                    ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
                } else {
                    let s = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
                    ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
                };
                Self::from_components(w, x, y, z)
            }

            /// Check if this quaternion and other are approximate equal.
            #[inline(always)]
            pub fn approx_eq(self, other: Self) -> bool {
                self.scalar.approx_eq(other.scalar) && self.complex.approx_eq(other.complex)
            }

            /// Create rotation around the x axis for the given angle.
            #[inline(always)]
            pub fn x_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vec_name::unit_x())
            }

            /// Create rotation around the y axis for the given angle.
            #[inline(always)]
            pub fn y_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vec_name::unit_y())
            }

            /// Create rotation around the z axis for the given angle.
            #[inline(always)]
            pub fn z_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vec_name::unit_z())
            }

            /// Check if it's a unit quaternion.
            #[inline(always)]
            pub fn is_unit(self) -> bool {
                self.norm().approx_eq(1.0)
            }

            /// Compute squared norm of the quaternion.
            #[inline(always)]
            pub fn norm_squared(self) -> $t {
                self.dot(self)
            }

            /// Compute norm of the quaternion.
            #[inline(always)]
            pub fn norm(self) -> $t {
                self.norm_squared().sqrt()
            }

            /// Compute normalised quaternion.
            #[inline(always)]
            pub fn normalised(self) -> Self {
                let n = self.norm();
                debug_assert!(n > 0.0);
                Self::new(self.scalar / n, self.complex / n)
            }

            /// Compute dot product with another quaternion.
            #[inline(always)]
            pub fn dot(self, other: Self) -> $t {
                self.scalar * other.scalar + self.complex.dot(other.complex)
            }

            /// Compute conjugate quaternion.
            #[inline(always)]
            pub fn conjugate(self) -> Self {
                Self::new(self.scalar, -self.complex)
            }

            /// Compute reciprocal quaternion.
            #[inline(always)]
            pub fn recip(self) -> Self {
                self.conjugate() / self.norm_squared()
            }

            /// Extract angle of the quaternion.
            /// Assumes the quaternion has been normalised so the scalar part should be at most 1.0,
            /// panics in debug if that's not the case.
            #[inline(always)]
            pub fn angle(self) -> $t {
                // If quaternion is normalised, this should not happen
                debug_assert!(self.scalar.abs() <= 1.0);
                2.0 * self.scalar.acos()
            }

            /// Apply quaternion as rotation to the given vector.
            #[inline(always)]
            pub fn rotate(self, v: $vec_name) -> $vec_name {
                // Naming
                let qw = self.scalar;
                let qx = self.complex.x;
                let qy = self.complex.y;
                let qz = self.complex.z;

                let qx_2 = qx * qx;
                let qy_2 = qy * qy;
                let qz_2 = qz * qz;

                $vec_name::new(
                    v.x * (1.0 - 2.0 * (qy_2 + qz_2))
                        + 2.0 * v.y * (qx * qy - qw * qz)
                        + 2.0 * v.z * (qx * qz + qw * qy),
                    2.0 * v.x * (qw * qz + qx * qy)
                        + v.y * (1.0 - 2.0 * (qx_2 + qz_2))
                        + 2.0 * v.z * (qy * qz - qw * qx),
                    2.0 * v.x * (qx * qz - qw * qy)
                        + 2.0 * v.y * (qy * qz + qw * qx)
                        + v.z * (1.0 - 2.0 * (qx_2 + qy_2)),
                )
            }

            /// Extract euler angles for the classical rotation order ZYX.
            #[inline]
            pub fn extract_euler_zyx(self) -> $euler_name {
                debug_assert!(self.is_unit());

                // Tolerances used in the computation
                const TOL_A: $t = $pi_2 - 10.0 * $eps;
                const TOL_B: $t = -$pi_2 + 10.0 * $eps;

                let qa = self.scalar;
                let (qb, qc, qd) = (-self.complex.x, -self.complex.y, -self.complex.z);
                let tmp = (2.0 * qb * qd - 2.0 * qa * qc).clamp(-1.0, 1.0);

                let b = -tmp.asin();
                let (a, c) = if b >= TOL_A {
                    (-2.0 * qb.atan2(qa), 0.0)
                } else if b <= TOL_B {
                    (2.0 * qb.atan2(qa), 0.0)
                } else {
                    (
                        (2.0 * qa * qd + 2.0 * qb * qc).atan2(2.0 * qa * qa - 1.0 + 2.0 * qb * qb),
                        (2.0 * qa * qb + 2.0 * qc * qd).atan2(2.0 * qa * qa - 1.0 + 2.0 * qd * qd),
                    )
                };

                (-a, -b, -c)
            }

            /// Compute a quaternion q such that q * self = target.
            #[inline]
            pub fn rotation_to(self, target: Self) -> Self {
                debug_assert!(self.is_unit());
                debug_assert!(target.is_unit());
                // The right part should be the inverse but as the squared norm is assumed to be 1,
                // we can simply multiply by the conjugate
                target * self.conjugate()
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                Self::Output::new(self.scalar + rhs.scalar, self.complex + rhs.complex)
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::Output::new(self.scalar - rhs.scalar, self.complex - rhs.complex)
            }
        }

        impl Div<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.scalar / rhs, self.complex / rhs)
            }
        }

        impl Mul for $name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self::Output {
                Self::Output::new(
                    self.scalar * rhs.scalar - self.complex.dot(rhs.complex),
                    self.scalar * rhs.complex
                        + rhs.scalar * self.complex
                        + self.complex.cross(rhs.complex),
                )
            }
        }

        impl Neg for $name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.scalar, -self.complex)
            }
        }
    };
}

generate_quaternion!(
    Quaternionf32,
    Vec3f32,
    EulerDecompositionf32,
    f32,
    f32::consts::FRAC_PI_2,
    f32::EPSILON
);
generate_quaternion!(
    Quaternionf64,
    Vec3f64,
    EulerDecompositionf64,
    f64,
    f64::consts::FRAC_PI_2,
    f64::EPSILON
);
