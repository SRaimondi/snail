use crate::vec3::{Vec3f32, Vec3f64};
use std::{
    f32, f64,
    ops::{Add, Div, Mul, Sub},
};

macro_rules! generate_quaternion {
    ($name:ident, $vname:ident, $t:ty, $pi:expr, $pi_2:expr) => {
        /// q = q_scalar + complex.x() * i + complex.y() * j + complex.z() * k.
        #[derive(Copy, Clone, Debug, Default, PartialEq)]
        #[repr(C)]
        pub struct $name {
            pub scalar: $t,
            pub complex: $vname,
        }

        impl $name {
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
                    complex: $vname::new(complex_x, complex_y, complex_z),
                }
            }

            /// Create new quaternion from the given scalar and complex parts.
            #[inline(always)]
            pub const fn new(scalar: $t, complex: $vname) -> Self {
                Self { scalar, complex }
            }

            /// Create pure quaternion from the given vector.
            #[inline(always)]
            pub const fn from_vector(v: $vname) -> Self {
                Self::new(0.0, v)
            }

            /// Create quaternion as rotation between two vectors.
            /// Vector are expected to be normalised.
            #[inline(always)]
            pub fn from_two_vectors_normalised(v1: $vname, v2: $vname) -> Self {
                debug_assert!(float_cmp::approx_eq!($t, v1.norm(), 1.0));
                debug_assert!(float_cmp::approx_eq!($t, v2.norm(), 1.0));
                let c = v1.dot(v2);
                if float_cmp::approx_eq!($t, c, -1.0) {
                    Self::new(0.0, v1.compute_perpendicular())
                } else {
                    let axis = v1.cross(v2);
                    let s = (2.0 * (1.0 + c)).sqrt();
                    let inv_s = 1.0 / s;
                    Self::new(0.5 * s, inv_s * axis)
                }
            }

            /// Create quaternion as rotation between two vectors.
            /// Vector are not expected to be normalised.
            #[inline(always)]
            pub fn from_two_vectors(v1: $vname, v2: $vname) -> Self {
                Self::from_two_vectors_normalised(v1.normalised(), v2.normalised())
            }

            /// Create rotation versor. Assumes axis has unit length.
            #[inline(always)]
            pub fn from_rotation(angle: $t, axis: $vname) -> Self {
                debug_assert!(float_cmp::approx_eq!($t, axis.norm(), 1.0));
                let (s, c) = (angle / 2.0).sin_cos();
                Self::new(c, s * axis)
            }

            /// Compute squared norm of the quaternion.
            #[inline(always)]
            pub fn norm_squared(self) -> $t {
                self.scalar * self.scalar + self.complex.norm_squared()
            }

            /// Compute norm of the quaternion.
            #[inline(always)]
            pub fn norm(self) -> $t {
                self.norm_squared().sqrt()
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

            /// Apply quaternion as rotation to the given vector.
            #[inline(always)]
            pub fn rotate(self, v: $vname) -> $vname {
                debug_assert!(float_cmp::approx_eq!($t, self.norm(), 1.0));
                // Naming
                let qw = self.scalar;
                let qx = self.complex.x();
                let qy = self.complex.y();
                let qz = self.complex.z();

                let qx_2 = qx * qx;
                let qy_2 = qy * qy;
                let qz_2 = qz * qz;

                $vname::new(
                    v.x() * (1.0 - 2.0 * (qy_2 + qz_2))
                        + 2.0 * v.y() * (qx * qy - qw * qz)
                        + 2.0 * v.z() * (qx * qz + qw * qy),
                    2.0 * v.x() * (qw * qz + qx * qy)
                        + v.y() * (1.0 - 2.0 * (qx_2 + qz_2))
                        + 2.0 * v.z() * (qy * qz - qw * qx),
                    2.0 * v.x() * (qx * qz - qw * qy)
                        + 2.0 * v.y() * (qy * qz + qw * qx)
                        + v.z() * (1.0 - 2.0 * (qx_2 + qy_2)),
                )
            }

            /// Extract Euler angles from the quaternion in x, y, z order, assumed to be rotated
            /// first on z, then on y and last on x.
            /// http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/Quaternions.pdf
            pub fn extract_euler(self) -> ($t, $t, $t) {
                debug_assert!(float_cmp::approx_eq!($t, self.norm(), 1.0));
                // Check for singularity
                let s_test = self.scalar * self.complex.y() + self.complex.z() * self.complex.x();
                if float_cmp::approx_eq!($t, s_test.abs(), 0.5) {
                    (
                        self.complex.z().atan2(self.scalar),
                        $pi_2.copysign(s_test),
                        0.0,
                    )
                } else {
                    let theta_z = (2.0
                        * (self.scalar * self.complex.z() - self.complex.y() * self.complex.x()))
                    .atan2(
                        1.0 - 2.0
                            * (self.complex.z() * self.complex.z()
                                + self.complex.y() * self.complex.y()),
                    );
                    let theta_y = (2.0 * s_test).asin();
                    let theta_x = (2.0
                        * (self.scalar * self.complex.x() - self.complex.z() * self.complex.y()))
                    .atan2(
                        1.0 - 2.0
                            * (self.complex.y() * self.complex.y()
                                + self.complex.x() * self.complex.x()),
                    );
                    (theta_z, theta_y, theta_x)
                }
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
    };
}

generate_quaternion!(
    Quaternionf32,
    Vec3f32,
    f32,
    f32::consts::PI,
    f32::consts::FRAC_PI_2
);
generate_quaternion!(
    Quaternionf64,
    Vec3f64,
    f64,
    f64::consts::PI,
    f64::consts::FRAC_PI_2
);
