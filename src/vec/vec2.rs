use crate::{Abs, ApproxEq, Clamp, Float, MinMax, One, Zero};

use std::{
    convert::From,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
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

/// Helper class representing boolean operations on vector
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Vec2bool {
    pub x: bool,
    pub y: bool,
}

impl Vec2bool {
    /// Create new boolean vector from the given values.
    #[inline(always)]
    pub const fn new(x: bool, y: bool) -> Self {
        Self { x, y }
    }

    /// Check if all elements are true.
    #[inline(always)]
    pub const fn all(self) -> bool {
        self.x && self.y
    }

    /// Check if any element is true.
    #[inline(always)]
    pub const fn any(self) -> bool {
        self.x || self.y
    }

    /// Select elements from the two given vectors using the boolean vector as mask.
    #[inline(always)]
    pub fn select<T>(a: Vector2<T>, b: Vector2<T>, mask: Self) -> Vector2<T> {
        Vector2::new(
            if mask.x { a.x } else { b.x },
            if mask.y { a.y } else { b.y },
        )
    }
}

impl Index<Axis2> for Vec2bool {
    type Output = bool;

    #[inline(always)]
    fn index(&self, axis: Axis2) -> &Self::Output {
        match axis {
            Axis2::X => &self.x,
            Axis2::Y => &self.y,
        }
    }
}

/// Generic vector in 2D.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vector2<T> {
    /// Create new vector from the given components.
    #[inline(always)]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Access element for the given axis.
    #[inline(always)]
    pub fn axis(self, axis: Axis2) -> T {
        match axis {
            Axis2::X => self.x,
            Axis2::Y => self.y,
        }
    }
}

impl<T> Vector2<T>
where
    T: Copy,
{
    /// Create new vector with all components set to the same value.
    #[inline(always)]
    pub fn broadcast(v: T) -> Self {
        Self::new(v, v)
    }

    /// Permute the vector components.
    #[inline(always)]
    pub fn permute(self, x_axis: Axis2, y_axis: Axis2) -> Self {
        Self::new(self.axis(x_axis), self.axis(y_axis))
    }

    /// Permute the vector components with the given array of axes.
    #[inline(always)]
    pub fn permute_with_array(self, axes: [Axis2; 2]) -> Self {
        self.permute(axes[0], axes[1])
    }
}

impl<T> Vector2<T>
where
    T: PartialOrd,
{
    /// Check if the elements are less than the one of the other.
    #[inline(always)]
    pub fn ewise_lt(self, other: Self) -> Vec2bool {
        Vec2bool::new(self.x < other.x, self.y < other.y)
    }

    /// Check if the elements are less or equal than the one of the other.
    #[inline(always)]
    pub fn ewise_le(self, other: Self) -> Vec2bool {
        Vec2bool::new(self.x <= other.x, self.y <= other.y)
    }

    /// Check if the elements are greater than the one of the other.
    #[inline(always)]
    pub fn ewise_gt(self, other: Self) -> Vec2bool {
        Vec2bool::new(self.x > other.x, self.y > other.y)
    }

    /// Check if the elements are greater or equal than the one of the other.
    #[inline(always)]
    pub fn ewise_ge(self, other: Self) -> Vec2bool {
        Vec2bool::new(self.x >= other.x, self.y >= other.y)
    }

    /// Check if the elements are equal to the one of the other.
    #[inline(always)]
    pub fn ewise_eq(self, other: Self) -> Vec2bool {
        Vec2bool::new(self.x == other.x, self.y == other.y)
    }

    /// Check if the elements are different to the one of the other.
    #[inline(always)]
    pub fn ewise_neq(self, other: Self) -> Vec2bool {
        Vec2bool::new(self.x != other.x, self.y != other.y)
    }

    /// Find largest axis.
    #[inline(always)]
    pub fn largest_axis(self) -> Axis2 {
        if self.x >= self.y {
            Axis2::X
        } else {
            Axis2::Y
        }
    }
}

impl<T> Vector2<T>
where
    T: Mul<Output = T>,
{
    /// Compute element-wise product.
    #[inline(always)]
    pub fn ewise_product(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y)
    }
}

impl<T> Vector2<T>
where
    T: Div<Output = T>,
{
    /// Compute element wise quotient.
    #[inline(always)]
    pub fn ewise_quotient(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y)
    }
}

impl<T> Vector2<T>
where
    T: Add<Output = T> + Mul<Output = T>,
{
    /// Compute dot product.
    #[inline(always)]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }
}

impl<T> Vector2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Compute squared norm.
    #[inline(always)]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }
}

impl<T> Vector2<T>
where
    T: Zero,
{
    /// Return a vector with all components zero.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO)
    }
}

impl<T> Vector2<T>
where
    T: Zero + One,
{
    /// Return unit x axis.
    #[inline(always)]
    pub fn unit_x() -> Self {
        Self::new(T::ONE, T::ZERO)
    }

    /// Return unit y axis.
    #[inline(always)]
    pub fn unit_y() -> Self {
        Self::new(T::ZERO, T::ONE)
    }

    /// Return the unit axis for the given axis.
    #[inline(always)]
    pub fn unit_for_axis(axis: Axis2) -> Self {
        match axis {
            Axis2::X => Self::unit_x(),
            Axis2::Y => Self::unit_y(),
        }
    }
}

impl<T> Vector2<T>
where
    T: Abs,
{
    /// Compute element wise absolute value.
    #[inline(always)]
    pub fn ewise_abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }
}

impl<T> Vector2<T>
where
    T: MinMax,
{
    /// Compute element wise minimum.
    #[inline(always)]
    pub fn ewise_min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// Compute element wise maximum.
    #[inline(always)]
    pub fn ewise_max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    /// Return the smallest value.
    #[inline(always)]
    pub fn min_element(self) -> T {
        self.x.min(self.y)
    }

    /// Return the largest value.
    #[inline(always)]
    pub fn max_element(self) -> T {
        self.x.max(self.y)
    }
}

impl<T> Vector2<T>
where
    T: Copy + Clamp + PartialOrd,
{
    /// Clamp each element of the vector with the given values.
    /// Panics if min > max.
    #[inline(always)]
    pub fn ewise_clamp(self, min: T, max: T) -> Self {
        assert!(min <= max);
        Self::new(self.x.clamp(min, max), self.y.clamp(min, max))
    }
}

impl<T> Vector2<T>
where
    T: Float,
{
    /// Create new vector from the given polar representation.
    #[inline(always)]
    pub fn from_polar(radius: T, angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(radius * c, radius * s)
    }

    /// Create new vector from the given angle.
    #[inline(always)]
    pub fn unit_polar(angle: T) -> Self {
        Self::from_polar(T::ONE, angle)
    }

    /// Compute norm of the vector.
    #[inline(always)]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Compute element wise reciprocal.
    #[inline(always)]
    pub fn ewise_recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip())
    }

    /// Return a new vector after normalising.
    #[inline(always)]
    pub fn normalised(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        Self::new(self.x / n, self.y / n)
    }

    /// Return a new vector after normalising, uses multiplication instead
    /// of division.
    #[inline(always)]
    pub fn normalised_fast(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        let inv_n = T::ONE / n;
        Self::new(self.x * inv_n, self.y * inv_n)
    }

    /// Normalise vector in place.
    #[inline(always)]
    pub fn normalise(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        self.x /= n;
        self.y /= n;
    }

    /// Normalise vector in place using multiplication.
    #[inline(always)]
    pub fn normalise_fast(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        let inv_n = T::ONE / n;
        self.x *= inv_n;
        self.y *= inv_n;
    }

    /// Linearly interpolate each element.
    #[inline(always)]
    pub fn ewise_lerp(self, t: T, end: Self) -> Self {
        Self::new(self.x.lerp(t, end.x), self.y.lerp(t, end.y))
    }

    /// Compute angle with the other vector.
    #[inline(always)]
    pub fn angle_with(self, other: Self) -> T {
        let dot = self.dot(other);
        let norm_prod = self.norm() * other.norm();
        (dot / norm_prod).clamp(-T::ONE, T::ONE).acos()
    }

    /// Compute angle with the other vector assuming they are both normalised.
    /// # Safety
    /// The function expects both the vector to norm one, using vectors that don't have unit
    /// length will produce incorrect results.
    #[inline(always)]
    pub unsafe fn unit_angle_with(self, other: Self) -> T {
        self.dot(other).clamp(-T::ONE, T::ONE).acos()
    }

    /// Rotate vector around origin for the given radiant angle counter-clockwise.
    #[inline(always)]
    pub fn rotate(self, angle_rad: T) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }
}

impl<T> Vector2<T>
where
    T: ApproxEq,
{
    /// Check if each component is approximately equal to the one of the other.
    #[inline(always)]
    pub fn ewise_approx_eq(self, other: Self) -> Vec2bool {
        Vec2bool::new(self.x.approx_eq(other.x), self.y.approx_eq(other.y))
    }

    /// Check if all components are approximately equal to the one of the other.
    #[inline(always)]
    pub fn approx_eq(self, other: Self) -> bool {
        self.ewise_approx_eq(other).all()
    }

    /// Check if each component is approximately zero.
    #[inline(always)]
    pub fn ewise_approx_zero(self) -> Vec2bool {
        Vec2bool::new(self.x.approx_zero(), self.y.approx_zero())
    }

    /// Check if all components are approximately zero.
    #[inline(always)]
    pub fn approx_zero(self) -> bool {
        self.ewise_approx_zero().all()
    }
}

impl<T> Add for Vector2<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<T> AddAssign for Vector2<T>
where
    T: AddAssign,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T> Sub for Vector2<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<T> SubAssign for Vector2<T>
where
    T: SubAssign,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T> Mul<T> for Vector2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::new(self.x * rhs, self.y * rhs)
    }
}

macro_rules! impl_mul {
    ($t:ty) => {
        impl Mul<Vector2<$t>> for $t {
            type Output = Vector2<$t>;

            #[inline(always)]
            fn mul(self, rhs: Vector2<$t>) -> Self::Output {
                Self::Output::new(self * rhs.x, self * rhs.y)
            }
        }
    };
}

impl_mul!(u8);
impl_mul!(u16);
impl_mul!(u32);
impl_mul!(u64);
impl_mul!(u128);
impl_mul!(usize);

impl_mul!(i8);
impl_mul!(i16);
impl_mul!(i32);
impl_mul!(i64);
impl_mul!(i128);
impl_mul!(isize);

impl_mul!(f32);
impl_mul!(f64);

impl<T> MulAssign<T> for Vector2<T>
where
    T: Copy + MulAssign,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T> Neg for Vector2<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl<T> Div<T> for Vector2<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl<T> DivAssign<T> for Vector2<T>
where
    T: Copy + DivAssign,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T> Index<Axis2> for Vector2<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: Axis2) -> &Self::Output {
        match index {
            Axis2::X => &self.x,
            Axis2::Y => &self.y,
        }
    }
}

impl<T> IndexMut<Axis2> for Vector2<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: Axis2) -> &mut Self::Output {
        match index {
            Axis2::X => &mut self.x,
            Axis2::Y => &mut self.y,
        }
    }
}

impl<T> From<Vector2<T>> for (T, T) {
    #[inline(always)]
    fn from(v: Vector2<T>) -> Self {
        (v.x, v.y)
    }
}
