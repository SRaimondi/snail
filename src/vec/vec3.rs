use crate::{Abs, ApproxEq, Clamp, Float, MinMax, One, Zero};

use std::{
    convert::From,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
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

/// Helper class representing boolean operations on vector
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Vec3bool {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl Vec3bool {
    /// Create new boolean vector from the given values.
    #[inline(always)]
    fn new(x: bool, y: bool, z: bool) -> Self {
        Self { x, y, z }
    }

    /// Check if all elements are true.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.x && self.y && self.z
    }

    /// Check if any element is true.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.x || self.y || self.z
    }
}

/// Generic vector in 3D.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector3<T> {
    /// Create new vector from the given components.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Access element for the given axis.
    #[inline(always)]
    pub fn axis(self, axis: Axis3) -> T {
        match axis {
            Axis3::X => self.x,
            Axis3::Y => self.y,
            Axis3::Z => self.z,
        }
    }
}

impl<T> Vector3<T>
where
    T: Copy,
{
    /// Create new vector with all components set to the same value.
    #[inline(always)]
    pub fn broadcast(v: T) -> Self {
        Self::new(v, v, v)
    }

    /// Permute the vector components.
    #[inline(always)]
    pub fn permute(self, x_axis: Axis3, y_axis: Axis3, z_axis: Axis3) -> Self {
        Self::new(self.axis(x_axis), self.axis(y_axis), self.axis(z_axis))
    }

    /// Permute the vector components with the given array of axes.
    #[inline(always)]
    pub fn permute_with_array(self, axes: [Axis3; 3]) -> Self {
        self.permute(axes[0], axes[1], axes[2])
    }
}

impl<T> Vector3<T>
where
    T: PartialOrd,
{
    /// Check if the elements are less than the one of the other.
    #[inline(always)]
    pub fn ewise_lt(self, other: Self) -> Vec3bool {
        Vec3bool::new(self.x < other.x, self.y < other.y, self.z < other.z)
    }

    /// Check if the elements are less or equal than the one of the other.
    #[inline(always)]
    pub fn ewise_le(self, other: Self) -> Vec3bool {
        Vec3bool::new(self.x <= other.x, self.y <= other.y, self.z <= other.z)
    }

    /// Check if the elements are greater than the one of the other.
    #[inline(always)]
    pub fn ewise_gt(self, other: Self) -> Vec3bool {
        Vec3bool::new(self.x > other.x, self.y > other.y, self.z > other.z)
    }

    /// Check if the elements are greater or equal than the one of the other.
    #[inline(always)]
    pub fn ewise_ge(self, other: Self) -> Vec3bool {
        Vec3bool::new(self.x >= other.x, self.y >= other.y, self.z >= other.z)
    }

    /// Find largest axis.
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
}

impl<T> Vector3<T>
where
    T: Mul<Output = T>,
{
    /// Compute element-wise product.
    #[inline(always)]
    pub fn ewise_product(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl<T> Vector3<T>
where
    T: Div<Output = T>,
{
    /// Compute element wise quotient.
    #[inline(always)]
    pub fn ewise_quotient(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

impl<T> Vector3<T>
where
    T: Add<Output = T> + Mul<Output = T>,
{
    /// Compute dot product.
    #[inline(always)]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<T> Vector3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    /// Compute cross product.
    #[inline(always)]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl<T> Vector3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    /// Compute squared norm.
    #[inline(always)]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }
}

impl<T> Vector3<T>
where
    T: Zero,
{
    /// Return a vector with all components zero.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(T::ZERO, T::ZERO, T::ZERO)
    }
}

impl<T> Vector3<T>
where
    T: Zero + One,
{
    /// Return unit x axis.
    #[inline(always)]
    pub fn unit_x() -> Self {
        Self::new(T::ONE, T::ZERO, T::ZERO)
    }

    /// Return unit y axis.
    #[inline(always)]
    pub fn unit_y() -> Self {
        Self::new(T::ZERO, T::ONE, T::ZERO)
    }

    /// Return unit z axis.
    #[inline(always)]
    pub fn unit_z() -> Self {
        Self::new(T::ZERO, T::ZERO, T::ONE)
    }
}

impl<T> Vector3<T>
where
    T: Abs,
{
    /// Compute element wise absolute value.
    #[inline(always)]
    pub fn ewise_abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

impl<T> Vector3<T>
where
    T: MinMax,
{
    /// Compute element wise minimum.
    #[inline(always)]
    pub fn ewise_min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Compute element wise maximum.
    #[inline(always)]
    pub fn ewise_max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    /// Return the smallest value.
    #[inline(always)]
    pub fn min_element(self) -> T {
        self.x.min(self.y).min(self.z)
    }

    /// Return the largest value.
    #[inline(always)]
    pub fn max_element(self) -> T {
        self.x.max(self.y).max(self.z)
    }
}

impl<T> Vector3<T>
where
    T: Copy + Clamp + PartialOrd,
{
    /// Clamp each element of the vector with the given values.
    /// Panics if min > max.
    #[inline(always)]
    pub fn ewise_clamp(self, min: T, max: T) -> Self {
        assert!(min <= max);
        Self::new(
            self.x.clamp(min, max),
            self.y.clamp(min, max),
            self.z.clamp(min, max),
        )
    }
}

impl<T> Vector3<T>
where
    T: Float,
{
    /// Create new vector from the given polar representation.
    #[inline(always)]
    pub fn from_polar(radius: T, phi: T, theta: T) -> Self {
        let (s_theta, c_theta) = theta.sin_cos();
        let r_xz = s_theta * radius;
        let (s_phi, c_phi) = phi.sin_cos();
        Self::new(r_xz * c_phi, radius * c_theta, r_xz * s_phi)
    }

    /// Create new vector from the given angles.
    #[inline(always)]
    pub fn unit_polar(phi: T, theta: T) -> Self {
        Self::from_polar(T::ONE, phi, theta)
    }

    /// Compute norm of the vector.
    #[inline(always)]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Compute element wise reciprocal.
    #[inline(always)]
    pub fn ewise_recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip(), self.z.recip())
    }

    /// Return a new vector after normalising.
    #[inline(always)]
    pub fn normalised(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        Self::new(self.x / n, self.y / n, self.z / n)
    }

    /// Return a new vector after normalising, uses multiplication instead
    /// of division.
    #[inline(always)]
    pub fn normalised_fast(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        let inv_n = T::ONE / n;
        Self::new(self.x * inv_n, self.y * inv_n, self.z * inv_n)
    }

    /// Normalise vector in place.
    #[inline(always)]
    pub fn normalise(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        self.x /= n;
        self.y /= n;
        self.z /= n;
    }

    /// Normalise vector in place using multiplication.
    #[inline(always)]
    pub fn normalise_fast(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::ZERO);
        let inv_n = T::ONE / n;
        self.x *= inv_n;
        self.y *= inv_n;
        self.z *= inv_n;
    }

    /// Linearly interpolate each element.
    #[inline(always)]
    pub fn ewise_lerp(self, t: T, end: Self) -> Self {
        Self::new(
            self.x.lerp(t, end.x),
            self.y.lerp(t, end.y),
            self.z.lerp(t, end.z),
        )
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
}

impl<T> Vector3<T>
where
    T: Abs + Float,
{
    /// Compute perpendicular vector.
    #[inline(always)]
    pub fn compute_perpendicular(self) -> Self {
        if self.x.abs() > self.y.abs() {
            let n = (self.x * self.x + self.z * self.z).sqrt();
            Self::new(-self.z / n, T::ZERO, self.x / n)
        } else {
            let n = (self.y * self.y + self.z * self.z).sqrt();
            Self::new(T::ZERO, self.z / n, -self.y / n)
        }
    }
}

impl<T> Vector3<T>
where
    T: ApproxEq,
{
    /// Check if each component is approximately equal to the one of the other.
    #[inline(always)]
    pub fn ewise_approx_eq(self, other: Self) -> Vec3bool {
        Vec3bool::new(
            self.x.approx_eq(other.x),
            self.y.approx_eq(other.y),
            self.z.approx_eq(other.z),
        )
    }

    /// Check if all components are approximately equal to the one of the other.
    #[inline(always)]
    pub fn approx_eq(self, other: Self) -> bool {
        self.ewise_approx_eq(other).all()
    }

    /// Check if each component is approximately zero.
    #[inline(always)]
    pub fn ewise_approx_zero(self) -> Vec3bool {
        Vec3bool::new(
            self.x.approx_zero(),
            self.y.approx_zero(),
            self.z.approx_zero(),
        )
    }

    /// Check if all components are approximately zero.
    #[inline(always)]
    pub fn approx_zero(self) -> bool {
        self.ewise_approx_zero().all()
    }
}

impl<T> Add for Vector3<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T> AddAssign for Vector3<T>
where
    T: AddAssign,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl<T> Sub for Vector3<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<T> SubAssign for Vector3<T>
where
    T: SubAssign,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl<T> Mul<T> for Vector3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

macro_rules! impl_mul {
    ($t:ty) => {
        impl Mul<Vector3<$t>> for $t {
            type Output = Vector3<$t>;

            #[inline(always)]
            fn mul(self, rhs: Vector3<$t>) -> Self::Output {
                Self::Output::new(self * rhs.x, self * rhs.y, self * rhs.z)
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

impl<T> MulAssign<T> for Vector3<T>
where
    T: Copy + MulAssign,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T> Neg for Vector3<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<T> Div<T> for Vector3<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<T> DivAssign<T> for Vector3<T>
where
    T: Copy + DivAssign,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl<T> Index<Axis3> for Vector3<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: Axis3) -> &Self::Output {
        match index {
            Axis3::X => &self.x,
            Axis3::Y => &self.y,
            Axis3::Z => &self.z,
        }
    }
}

impl<T> IndexMut<Axis3> for Vector3<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: Axis3) -> &mut Self::Output {
        match index {
            Axis3::X => &mut self.x,
            Axis3::Y => &mut self.y,
            Axis3::Z => &mut self.z,
        }
    }
}

impl<T> From<Vector3<T>> for (T, T, T) {
    #[inline(always)]
    fn from(v: Vector3<T>) -> Self {
        (v.x, v.y, v.z)
    }
}
