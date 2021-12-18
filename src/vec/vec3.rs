use crate::ApproxEq;
use std::{
    convert::From,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Enum used to represent the axes for Vec3.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis3 {
    X,
    Y,
    Z,
}

impl Axis3 {
    /// Get the value of the next axis looping back from z to x.
    #[inline(always)]
    pub const fn next(self) -> Self {
        match self {
            Self::X => Self::Y,
            Self::Y => Self::Z,
            Self::Z => Self::X,
        }
    }
}

macro_rules! generate_vec3 {
    ($name:ident, $t:ty) => {
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(C)]
        pub struct $name {
            pub x: $t,
            pub y: $t,
            pub z: $t,
        }

        impl $name {
            /// Associated constant representing the zero vector.
            pub const ZERO: Self = Self::broadcast(0.0);
            /// Associated constant representing the x axis.
            pub const UNIT_X: Self = Self::new(1.0, 0.0, 0.0);
            /// Associated constant representing the y axis.
            pub const UNIT_Y: Self = Self::new(0.0, 1.0, 0.0);
            /// Associated constant representing the z axis.
            pub const UNIT_Z: Self = Self::new(0.0, 0.0, 1.0);

            /// Create new vector from the given coordinates.
            #[inline(always)]
            pub const fn new(x: $t, y: $t, z: $t) -> Self {
                Self { x, y, z }
            }

            /// Create new vector with all components set to the given value.
            #[inline(always)]
            pub const fn broadcast(v: $t) -> Self {
                Self::new(v, v, v)
            }

            /// Create new vector from the given polar representation.
            /// phi is the angle with respect to the x axis and theta is
            /// the angle with respect to the y axis.
            #[inline(always)]
            pub fn from_polar(radius: $t, phi: $t, theta: $t) -> Self {
                let (s_theta, c_theta) = theta.sin_cos();
                let r_xz = s_theta * radius;
                let (s_phi, c_phi) = phi.sin_cos();
                Self::new(r_xz * c_phi, radius * c_theta, r_xz * s_phi)
            }

            /// Check if this vector and other are approximate equal.
            #[inline(always)]
            pub fn approx_eq(self, other: Self) -> bool {
                self.x.approx_eq(other.x) && self.y.approx_eq(other.y) && self.z.approx_eq(other.z)
            }

            /// Compute minimum value for each component.
            #[inline(always)]
            pub fn min(self, rhs: Self) -> Self {
                Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
            }

            /// Compute minimum value for each component assuming there are no NaNs.
            #[inline(always)]
            pub fn min_fast(self, rhs: Self) -> Self {
                Self::new(
                    super::min_helper(self.x, rhs.x),
                    super::min_helper(self.y, rhs.y),
                    super::min_helper(self.z, rhs.z),
                )
            }

            /// Compute maximum value for each component.
            #[inline(always)]
            pub fn max(self, rhs: Self) -> Self {
                Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
            }

            /// Compute maximum value for each component assuming there are no NaNs.
            #[inline(always)]
            pub fn max_fast(self, rhs: Self) -> Self {
                Self::new(
                    super::max_helper(self.x, rhs.x),
                    super::max_helper(self.y, rhs.y),
                    super::max_helper(self.z, rhs.z),
                )
            }

            /// Compute product for each component.
            #[inline(always)]
            pub fn product(self, rhs: Self) -> Self {
                Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
            }

            /// Compute quotient for each component.
            #[inline(always)]
            pub fn quotient(self, rhs: Self) -> Self {
                Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
            }

            /// Compute absolute value for each component.
            #[inline(always)]
            pub fn abs(self) -> Self {
                Self::new(self.x.abs(), self.y.abs(), self.z.abs())
            }

            /// Compute reciprocal value for each component.
            #[inline(always)]
            pub fn recip(self) -> Self {
                Self::new(self.x.recip(), self.y.recip(), self.z.recip())
            }

            /// Clamp each value between the given min and max.
            #[inline(always)]
            pub fn clamp(self, min: $t, max: $t) -> Self {
                Self::new(
                    self.x.clamp(min, max),
                    self.y.clamp(min, max),
                    self.z.clamp(min, max),
                )
            }

            /// Return a new normalised vector.
            #[inline(always)]
            pub fn normalised(self) -> Self {
                let n = self.norm();
                debug_assert!(n > 0.0);
                self / n
            }

            /// Return a new normalised vector, uses multiplication instead of division on the components.
            #[inline(always)]
            pub fn normalised_fast(self) -> Self {
                let n = self.norm();
                debug_assert!(n > 0.0);
                (1.0 / n) * self
            }

            /// Normalise vector in place.
            #[inline(always)]
            pub fn normalise(&mut self) {
                let n = self.norm();
                debug_assert!(n > 0.0);
                self.x /= n;
                self.y /= n;
                self.z /= n;
            }

            /// Normalise vector in place using multiplication.
            #[inline(always)]
            pub fn normalise_fast(&mut self) {
                let n = self.norm();
                debug_assert!(n > 0.0);
                let inv_n = 1.0 / n;
                self.x *= inv_n;
                self.y *= inv_n;
                self.z *= inv_n;
            }

            /// Linearly interpolate for each component.
            #[inline(always)]
            pub fn lerp(self, t: $t, end: Self) -> Self {
                (1.0 - t) * self + t * end
            }

            /// Compute cross product.
            #[inline(always)]
            pub fn cross(self, rhs: Self) -> Self {
                Self::new(
                    self.y * rhs.z - self.z * rhs.y,
                    self.z * rhs.x - self.x * rhs.z,
                    self.x * rhs.y - self.y * rhs.x,
                )
            }

            /// Compute dot product.
            #[inline(always)]
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
            }

            /// Compute squared norm of the vector.
            #[inline(always)]
            pub fn norm_squared(self) -> $t {
                self.dot(self)
            }

            /// Compute norm of the vector.
            #[inline(always)]
            pub fn norm(self) -> $t {
                self.norm_squared().sqrt()
            }

            /// Compute minimum element.
            #[inline(always)]
            pub fn min_element(self) -> $t {
                self.x.min(self.y.min(self.z))
            }

            /// Compute minimum element assuming there are no NaNs.
            #[inline(always)]
            pub fn min_element_fast(self) -> $t {
                super::min3_helper(self.x, self.y, self.z)
            }

            /// Compute maximum element.
            #[inline(always)]
            pub fn max_element(self) -> $t {
                self.x.max(self.y.max(self.z))
            }

            /// Compute maximum element assuming there are no NaNs.
            #[inline(always)]
            pub fn max_element_fast(self) -> $t {
                super::max3_helper(self.x, self.y, self.z)
            }

            /// Get element for the given axis.
            #[inline(always)]
            pub fn axis(self, axis: Axis3) -> $t {
                match axis {
                    Axis3::X => self.x,
                    Axis3::Y => self.y,
                    Axis3::Z => self.z,
                }
            }

            /// Get mutable reference to element for the given axis.
            #[inline(always)]
            pub fn axis_mut(&mut self, axis: Axis3) -> &mut $t {
                match axis {
                    Axis3::X => &mut self.x,
                    Axis3::Y => &mut self.y,
                    Axis3::Z => &mut self.z,
                }
            }

            /// Permute components for the given new axes.
            #[inline(always)]
            pub fn permute(self, x_axis: Axis3, y_axis: Axis3, z_axis: Axis3) -> Self {
                Self::new(self.axis(x_axis), self.axis(y_axis), self.axis(z_axis))
            }

            /// Permute components for the given new axes as array.
            #[inline(always)]
            pub fn permute_with_array(self, axes: [Axis3; 3]) -> Self {
                self.permute(axes[0], axes[1], axes[2])
            }

            /// Compute largest axis of the vector.
            #[inline(always)]
            pub fn largest_axis(self) -> Axis3 {
                if self.x >= self.y && self.x >= self.z {
                    Axis3::X
                } else if self.y >= self.z {
                    Axis3::Y
                } else {
                    Axis3::Z
                }
            }

            /// Compute perpendicular vector.
            #[inline(always)]
            pub fn compute_perpendicular(self) -> Self {
                if self.x.abs() > self.y.abs() {
                    let n = (self.x * self.x + self.z * self.z).sqrt();
                    Self::new(-self.z / n, 0.0, self.x / n)
                } else {
                    let n = (self.y * self.y + self.z * self.z).sqrt();
                    Self::new(0.0, self.z / n, -self.y / n)
                }
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
            }
        }

        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.x += rhs.x;
                self.y += rhs.y;
                self.z += rhs.z;
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
            }
        }

        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.x -= rhs.x;
                self.y -= rhs.y;
                self.z -= rhs.z;
            }
        }

        impl Mul<$name> for $t {
            type Output = $name;

            #[inline(always)]
            fn mul(self, rhs: $name) -> Self::Output {
                Self::Output::new(self * rhs.x, self * rhs.y, self * rhs.z)
            }
        }

        impl Mul<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x * rhs, self.y * rhs, self.z * rhs)
            }
        }

        impl MulAssign<$t> for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: $t) {
                self.x *= rhs;
                self.y *= rhs;
                self.z *= rhs;
            }
        }

        impl Neg for $name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.x, -self.y, -self.z)
            }
        }

        impl Div<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x / rhs, self.y / rhs, self.z / rhs)
            }
        }

        impl DivAssign<$t> for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: $t) {
                self.x /= rhs;
                self.y /= rhs;
                self.z /= rhs;
            }
        }

        impl From<$name> for ($t, $t, $t) {
            #[inline(always)]
            fn from(v: $name) -> ($t, $t, $t) {
                (v.x, v.y, v.z)
            }
        }

        impl From<$name> for [$t; 3] {
            #[inline(always)]
            fn from(v: $name) -> [$t; 3] {
                [v.x, v.y, v.z]
            }
        }
    };
}

generate_vec3!(Vec3f32, f32);
generate_vec3!(Vec3f64, f64);
