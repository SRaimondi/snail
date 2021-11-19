use crate::{Axis3, Vec3f32, Vec3f64};
use std::ops::{Add, Div, Mul, Sub, Neg};

macro_rules! generate_quaternion {
    ($name:ident, $euler_name:ident, $vname:ident, $t:ty, $pi_2:expr, $eps:expr) => {
        #[derive(Copy, Clone)]
        pub enum $euler_name {
            Normal($t, $t, $t),
            Singularity($t, $t),
        }

        impl std::convert::Into<($t, $t, $t)> for $euler_name {
            #[inline(always)]
            fn into(self) -> ($t, $t, $t) {
                match self {
                    Self::Normal(t0, t1, t2) => (t0, t1, t2),
                    Self::Singularity(t0, t1) => (t0, t1, 0.0),
                }
            }
        }

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
            pub fn from_two_vectors_normalised(from: $vname, to: $vname) -> Self {
                debug_assert!(float_cmp::approx_eq!($t, from.norm_squared(), 1.0));
                debug_assert!(float_cmp::approx_eq!($t, to.norm_squared(), 1.0));
                let c = from.dot(to);
                if c < -1.0 + $eps {
                    Self::from_vector(from.compute_perpendicular())
                } else {
                    let axis = from.cross(to);
                    let s = (2.0 * (1.0 + c)).sqrt();
                    let inv_s = 1.0 / s;
                    Self::new(0.5 * s, inv_s * axis)
                }
            }

            /// Create quaternion as rotation between two vectors.
            /// Vector are not expected to be normalised.
            #[inline(always)]
            pub fn from_two_vectors(from: $vname, to: $vname) -> Self {
                Self::from_two_vectors_normalised(from.normalised(), to.normalised())
            }

            /// Create rotation versor. Assumes axis has unit length.
            #[inline(always)]
            pub fn from_rotation(angle: $t, axis: $vname) -> Self {
                debug_assert!(float_cmp::approx_eq!($t, axis.norm_squared(), 1.0));
                let (s, c) = (angle / 2.0).sin_cos();
                Self::new(c, s * axis)
            }

            /// Create rotation around the x axis for the given angle.
            #[inline(always)]
            pub fn x_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vname::EX)
            }

            /// Create rotation around the y axis for the given angle.
            #[inline(always)]
            pub fn y_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vname::EY)
            }

            /// Create rotation around the z axis for the given angle.
            #[inline(always)]
            pub fn z_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vname::EZ)
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

            /// Extract angle of the quaternion
            #[inline(always)]
            pub fn angle(self) -> $t {
                debug_assert!(float_cmp::approx_eq!($t, axis.norm_squared(), 1.0));
                2.0 * self.scalar.acos()
            }

            /// Apply quaternion as rotation to the given vector.
            #[inline(always)]
            pub fn rotate(self, v: $vname) -> $vname {
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

            /// Extract the Euler angles for the given order from the quaternion.
            /// The output angles are ordered the same way rotations should be applied .0, .1 and .2
            /// See http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/Quaternions.pdf
            #[inline]
            pub fn extract_euler_angles(self, order: EulerOrder) -> $euler_name {
                debug_assert!(float_cmp::approx_eq!($t, self.norm_squared(), 1.0));
                // Naming
                let p0 = self.scalar;
                let (p1, p2, p3) = self.complex.permute_with_array(order.permutation()).into();
                // Compute e for sign swap
                let e = if order.is_e_positive() { 1.0 } else { -1.0 };

                // Test if we are at a singularity
                let s_test = p0 * p2 + e * p1 * p3;
                if s_test.abs() > 0.5 - $eps {
                    $euler_name::Singularity(2.0 * p1.atan2(p0), $pi_2.copysign(s_test))
                } else {
                    $euler_name::Normal(
                        (2.0 * (p0 * p1 - e * p2 * p3)).atan2(1.0 - 2.0 * (p1 * p1 + p2 * p2)),
                        (2.0 * s_test).asin(),
                        (2.0 * (p0 * p3 - e * p1 * p2)).atan2(1.0 - 2.0 * (p2 * p2 + p3 * p3)),
                    )
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

        impl Neg for $name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.scalar, -self.complex)
            }
        }
    };
}

#[derive(Copy, Clone)]
pub enum EulerOrder {
    XYZ,
    YZX,
    ZXY,
    ZYX,
    XZY,
    YXZ,
}

impl EulerOrder {
    #[inline(always)]
    fn is_e_positive(self) -> bool {
        match self {
            Self::XYZ | Self::YZX | Self::ZXY => false,
            Self::ZYX | Self::XZY | Self::YXZ => true,
        }
    }

    #[inline(always)]
    fn permutation(self) -> [Axis3; 3] {
        match self {
            Self::XYZ => [Axis3::X, Axis3::Y, Axis3::Z],
            Self::YZX => [Axis3::Y, Axis3::Z, Axis3::X],
            Self::ZXY => [Axis3::Z, Axis3::X, Axis3::Y],
            Self::ZYX => [Axis3::Z, Axis3::Y, Axis3::X],
            Self::XZY => [Axis3::X, Axis3::Z, Axis3::Y],
            Self::YXZ => [Axis3::Y, Axis3::X, Axis3::Z],
        }
    }
}

generate_quaternion!(
    Quaternionf32,
    EulerDecompositionf32,
    Vec3f32,
    f32,
    std::f32::consts::FRAC_PI_2,
    1e-5
);
generate_quaternion!(
    Quaternionf64,
    EulerDecompositionf64,
    Vec3f64,
    f64,
    std::f64::consts::FRAC_PI_2,
    1e-12
);
