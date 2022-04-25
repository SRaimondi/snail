use crate::ApproxEq;
use std::{
    convert::From,
    ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Enum used to represent the axes for Vec2.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis2 {
    X,
    Y,
}

impl Axis2 {
    /// Get the value of the next axis looping back from y to x.
    #[inline(always)]
    pub const fn next(self) -> Self {
        match self {
            Self::X => Self::Y,
            Self::Y => Self::X,
        }
    }
}

/// Boolean vector used in element wise logical operations.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct Vec2bool {
    pub x: bool,
    pub y: bool,
}

impl Vec2bool {
    /// Create new boolean vector.
    #[inline(always)]
    pub fn new(x: bool, y: bool) -> Self {
        Self { x, y }
    }

    /// Check if all elements are true.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.x && self.y
    }

    /// Check if any element is true.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.x || self.y
    }
}

macro_rules! generate_vec2 {
    ($name:ident, $t:ty) => {
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(C)]
        pub struct $name {
            pub x: $t,
            pub y: $t,
        }

        impl $name {
            /// Associated constant representing the zero vector.
            pub const ZERO: Self = Self::broadcast(0.0);
            /// Associated constant representing the x axis.
            pub const UNIT_X: Self = Self::new(1.0, 0.0);
            /// Associated constant representing the y axis.
            pub const UNIT_Y: Self = Self::new(0.0, 1.0);

            /// Create new vector from the given coordinates.
            #[inline(always)]
            pub const fn new(x: $t, y: $t) -> Self {
                Self { x, y }
            }

            /// Create new vector with all components set to the given value.
            #[inline(always)]
            pub const fn broadcast(v: $t) -> Self {
                Self::new(v, v)
            }

            /// Create new vector from the given polar representation.
            #[inline(always)]
            pub fn from_polar(radius: $t, angle: $t) -> Self {
                let (s, c) = angle.sin_cos();
                Self::new(radius * c, radius * s)
            }

            /// Create new vector from the given angle.
            #[inline(always)]
            pub fn unit_polar(angle: $t) -> Self {
                Self::from_polar(1.0, angle)
            }

            /// Check if this vector and other are equal for the default crate tolerance.
            #[inline(always)]
            pub fn approx_eq(self, other: Self) -> Vec2bool {
                Vec2bool::new(self.x.approx_eq(other.x), self.y.approx_eq(other.y))
            }

            /// Check if components are approximately zero.
            #[inline(always)]
            pub fn approx_zero(self) -> Vec2bool {
                Vec2bool::new(self.x.approx_zero(), self.y.approx_zero())
            }

            /// Check if each component is less than the other.
            #[inline(always)]
            pub fn lt(self, other: Self) -> Vec2bool {
                Vec2bool::new(self.x < other.x, self.y < other.y)
            }

            /// Check if each component is less or equal then the other.
            #[inline(always)]
            pub fn le(self, other: Self) -> Vec2bool {
                Vec2bool::new(self.x <= other.x, self.y <= other.y)
            }

            /// Check if each component is larger than the other.
            #[inline(always)]
            pub fn gt(self, other: Self) -> Vec2bool {
                Vec2bool::new(self.x > other.x, self.y > other.y)
            }

            /// Check if each component is larger or equal then the other.
            #[inline(always)]
            pub fn ge(self, other: Self) -> Vec2bool {
                Vec2bool::new(self.x >= other.x, self.y >= other.y)
            }

            /// Compute minimum value for each component.
            #[inline(always)]
            pub fn ewise_min(self, rhs: Self) -> Self {
                Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
            }

            /// Compute maximum value for each component.
            #[inline(always)]
            pub fn ewise_max(self, rhs: Self) -> Self {
                Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
            }

            /// Compute product for each component.
            #[inline(always)]
            pub fn ewise_product(self, rhs: Self) -> Self {
                Self::new(self.x * rhs.x, self.y * rhs.y)
            }

            /// Compute quotient for each component.
            #[inline(always)]
            pub fn ewise_quotient(self, rhs: Self) -> Self {
                Self::new(self.x / rhs.x, self.y / rhs.y)
            }

            /// Compute absolute value for each component.
            #[inline(always)]
            pub fn ewise_abs(self) -> Self {
                Self::new(self.x.abs(), self.y.abs())
            }

            /// Compute reciprocal value for each component.
            #[inline(always)]
            pub fn ewise_recip(self) -> Self {
                Self::new(self.x.recip(), self.y.recip())
            }

            /// Clamp each value between the given min and max.
            #[inline(always)]
            pub fn ewise_clamp(self, min: $t, max: $t) -> Self {
                Self::new(self.x.clamp(min, max), self.y.clamp(min, max))
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
            }

            /// Normalise vector in place using multiplication.
            #[inline(always)]
            pub fn normalise_fast(&mut self) {
                let n = self.norm();
                debug_assert!(n > 0.0);
                let inv_n = 1.0 / n;
                self.x *= inv_n;
                self.y *= inv_n;
            }

            /// Linearly interpolate for each component.
            #[inline(always)]
            pub fn ewise_lerp(self, t: $t, end: Self) -> Self {
                (1.0 - t) * self + t * end
            }

            /// Return angle between self and other in radians.
            #[inline(always)]
            pub fn angle_with(self, other: Self) -> $t {
                let dot = self.dot(other);
                let norm_prod = self.norm() * other.norm();
                (dot / norm_prod).clamp(-1.0, 1.0).acos()
            }

            /// Return angle between self and other in radians, assumes vectors are unit length.
            #[inline(always)]
            pub fn unit_angle_with(self, other: Self) -> $t {
                debug_assert!(self.norm().approx_eq(1.0));
                debug_assert!(other.norm().approx_eq(1.0));
                self.dot(other).clamp(-1.0, 1.0).acos()
            }

            /// Compute dot product.
            #[inline(always)]
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y
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
                self.x.min(self.y)
            }

            /// Compute maximum element.
            #[inline(always)]
            pub fn max_element(self) -> $t {
                self.x.max(self.y)
            }

            /// Get element for the given axis.
            #[inline(always)]
            pub fn axis(self, axis: Axis2) -> $t {
                match axis {
                    Axis2::X => self.x,
                    Axis2::Y => self.y,
                }
            }

            /// Get mutable reference to element for the given axis.
            #[inline(always)]
            pub fn axis_mut(&mut self, axis: Axis2) -> &mut $t {
                match axis {
                    Axis2::X => &mut self.x,
                    Axis2::Y => &mut self.y,
                }
            }

            /// Permute components for the given new axes.
            #[inline(always)]
            pub fn permute(self, x_axis: Axis2, y_axis: Axis2) -> Self {
                Self::new(self.axis(x_axis), self.axis(y_axis))
            }

            /// Permute components for the given new axes as array.
            #[inline(always)]
            pub fn permute_with_array(self, axes: [Axis2; 2]) -> Self {
                self.permute(axes[0], axes[1])
            }

            /// Compute largest axis of the vector.
            #[inline(always)]
            pub fn largest_axis(self) -> Axis2 {
                if self.x >= self.y {
                    Axis2::X
                } else {
                    Axis2::Y
                }
            }

            /// Rotate vector around origin for the given angle in radians.
            #[inline(always)]
            pub fn rotate(self, angle_rad: $t) -> Self {
                let (s, c) = angle_rad.sin_cos();
                Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self::new(self.x + rhs.x, self.y + rhs.y)
            }
        }

        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.x += rhs.x;
                self.y += rhs.y;
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self::new(self.x - rhs.x, self.y - rhs.y)
            }
        }

        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.x -= rhs.x;
                self.y -= rhs.y;
            }
        }

        impl Mul<$name> for $t {
            type Output = $name;

            #[inline(always)]
            fn mul(self, rhs: $name) -> Self::Output {
                Self::Output::new(self * rhs.x, self * rhs.y)
            }
        }

        impl Mul<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x * rhs, self.y * rhs)
            }
        }

        impl MulAssign<$t> for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: $t) {
                self.x *= rhs;
                self.y *= rhs;
            }
        }

        impl Neg for $name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.x, -self.y)
            }
        }

        impl Div<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x / rhs, self.y / rhs)
            }
        }

        impl DivAssign<$t> for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: $t) {
                self.x /= rhs;
                self.y /= rhs;
            }
        }

        impl Index<Axis2> for $name {
            type Output = $t;

            #[inline(always)]
            fn index(&self, index: Axis2) -> &Self::Output {
                match index {
                    Axis2::X => &self.x,
                    Axis2::Y => &self.y,
                }
            }
        }

        impl From<$name> for ($t, $t) {
            #[inline(always)]
            fn from(v: $name) -> ($t, $t) {
                (v.x, v.y)
            }
        }

        impl From<$name> for [$t; 2] {
            #[inline(always)]
            fn from(v: $name) -> [$t; 2] {
                [v.x, v.y]
            }
        }
    };
}

generate_vec2!(Vec2f32, f32);
generate_vec2!(Vec2f64, f64);
