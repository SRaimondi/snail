use crate::{ApproxEq, Axis3, Vec3f32, Vec3f64};
use std::{
    convert::Into,
    ops::{Add, Div, Mul, Neg, Sub},
};

macro_rules! generate_quaternion {
    ($name:ident, $euler_name:ident, $vec_name:ident, $t:ty, $pi_2:expr) => {
        /// Enum representing the result of the decomposition in Euler angles.
        #[derive(Copy, Clone)]
        pub enum $euler_name {
            /// All angles are used for the rotation.
            Normal($t, $t, $t),
            /// We are at a singularity, the last angle has value 0.
            Singularity($t, $t),
        }

        #[allow(clippy::from_over_into)]
        impl Into<($t, $t, $t)> for $euler_name {
            #[inline(always)]
            fn into(self) -> ($t, $t, $t) {
                match self {
                    Self::Normal(t0, t1, t2) => (t0, t1, t2),
                    Self::Singularity(t0, t1) => (t0, t1, 0.0),
                }
            }
        }

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
            /// Vector are expected to be normalised.
            #[inline(always)]
            pub fn from_two_vectors_normalised(from: $vec_name, to: $vec_name) -> Self {
                debug_assert!(from.norm().approx_eq(1.0));
                debug_assert!(to.norm().approx_eq(1.0));

                let c = from.dot(to);
                if c < -1.0 + <$t as ApproxEq>::DEFAULT_ABSOLUTE_EPS {
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
            pub fn from_two_vectors(from: $vec_name, to: $vec_name) -> Self {
                Self::from_two_vectors_normalised(from.normalised(), to.normalised())
            }

            /// Create rotation versor. Assumes axis has unit length.
            #[inline(always)]
            pub fn from_rotation(angle: $t, axis: $vec_name) -> Self {
                debug_assert!(axis.norm().approx_eq(1.0));
                let (s, c) = (angle / 2.0).sin_cos();
                Self::new(c, s * axis)
            }

            /// Check if this quaternion and other are approximate equal.
            #[inline(always)]
            pub fn approx_eq(self, other: Self) -> bool {
                self.scalar.approx_eq(other.scalar) && self.complex.approx_eq(other.complex)
            }

            /// Create rotation around the x axis for the given angle.
            #[inline(always)]
            pub fn x_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vec_name::UNIT_X)
            }

            /// Create rotation around the y axis for the given angle.
            #[inline(always)]
            pub fn y_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vec_name::UNIT_Y)
            }

            /// Create rotation around the z axis for the given angle.
            #[inline(always)]
            pub fn z_rotation(angle: $t) -> Self {
                Self::from_rotation(angle, $vec_name::UNIT_Z)
            }

            /// Check if it's a unit quaternion.
            #[inline(always)]
            pub fn is_unit(self) -> bool {
                self.norm().approx_eq_abs_eps(1.0, 0.0001)
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
            #[inline(always)]
            pub fn angle(self) -> $t {
                debug_assert!(self.scalar <= 1.0);
                2.0 * self.scalar.clamp(-1.0, 1.0).acos()
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

            /// Extract the Euler angles for the given order from the quaternion.
            /// The output angles are ordered the same way rotations should be applied .0, .1 and .2
            /// See http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/Quaternions.pdf
            #[inline]
            pub fn extract_euler_angles(self, order: EulerOrder) -> $euler_name {
                debug_assert!(self.is_unit());
                // Naming
                let p0 = self.scalar;
                let (p1, p2, p3) = self.complex.permute_with_array(order.permutation()).into();
                // Compute e for sign swap
                let e = if order.is_e_positive() { 1.0 } else { -1.0 };

                // Test if we are at a singularity
                let s_test = p0 * p2 + e * p1 * p3;
                if s_test.abs() > 0.5 - <$t as ApproxEq>::DEFAULT_ABSOLUTE_EPS {
                    $euler_name::Singularity(2.0 * p1.atan2(p0), $pi_2.copysign(s_test))
                } else {
                    $euler_name::Normal(
                        (2.0 * (p0 * p1 - e * p2 * p3)).atan2(1.0 - 2.0 * (p1 * p1 + p2 * p2)),
                        (2.0 * s_test).asin(),
                        (2.0 * (p0 * p3 - e * p1 * p2)).atan2(1.0 - 2.0 * (p2 * p2 + p3 * p3)),
                    )
                }
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
    std::f32::consts::FRAC_PI_2
);
generate_quaternion!(
    Quaternionf64,
    EulerDecompositionf64,
    Vec3f64,
    f64,
    std::f64::consts::FRAC_PI_2
);
