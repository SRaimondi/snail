use crate::{Abs, ApproxEq, Float, One, Zero};

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
pub struct Vec3Bool {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl Vec3Bool {
    #[inline]
    fn new(x: bool, y: bool, z: bool) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn all(self) -> bool {
        self.x && self.y && self.z
    }

    #[inline]
    pub fn any(self) -> bool {
        self.x || self.y || self.z
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector3<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    #[inline]
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
    #[inline]
    pub fn broadcast(v: T) -> Self {
        Self::new(v, v, v)
    }

    #[inline]
    pub fn permute(self, x_axis: Axis3, y_axis: Axis3, z_axis: Axis3) -> Self {
        Self::new(self.axis(x_axis), self.axis(y_axis), self.axis(z_axis))
    }

    #[inline]
    pub fn permute_with_array(self, axes: [Axis3; 3]) -> Self {
        self.permute(axes[0], axes[1], axes[2])
    }
}

impl<T> Vector3<T>
where
    T: PartialOrd,
{
    #[inline]
    pub fn ewise_lt(self, other: Self) -> Vec3Bool {
        Vec3Bool::new(self.x < other.x, self.y < other.y, self.z < other.z)
    }

    #[inline]
    pub fn ewise_le(self, other: Self) -> Vec3Bool {
        Vec3Bool::new(self.x <= other.x, self.y <= other.y, self.z <= other.z)
    }

    #[inline]
    pub fn ewise_gt(self, other: Self) -> Vec3Bool {
        Vec3Bool::new(self.x > other.x, self.y > other.y, self.z > other.z)
    }

    #[inline]
    pub fn ewise_ge(self, other: Self) -> Vec3Bool {
        Vec3Bool::new(self.x >= other.x, self.y >= other.y, self.z >= other.z)
    }

    #[inline]
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
    #[inline]
    pub fn ewise_product(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl<T> Vector3<T>
where
    T: Div<Output = T>,
{
    #[inline]
    pub fn ewise_quotient(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

impl<T> Vector3<T>
where
    T: Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<T> Vector3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    #[inline]
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
    #[inline]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }
}

impl<T> Vector3<T>
where
    T: Zero,
{
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }
}

impl<T> Vector3<T>
where
    T: Zero + One,
{
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero(), T::zero())
    }

    #[inline]
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one(), T::zero())
    }

    #[inline]
    pub fn unit_z() -> Self {
        Self::new(T::zero(), T::zero(), T::one())
    }
}

impl<T> Vector3<T>
where
    T: Abs,
{
    #[inline]
    pub fn ewise_abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

impl<T> Vector3<T>
where
    T: Ord,
{
    #[inline]
    pub fn ewise_min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    #[inline]
    pub fn ewise_max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    #[inline]
    pub fn min_element(self) -> T {
        self.x.min(self.y).min(self.z)
    }

    #[inline]
    pub fn max_element(self) -> T {
        self.x.max(self.y).max(self.z)
    }
}

impl<T> Vector3<T>
where
    T: Copy + Ord,
{
    #[inline]
    pub fn ewise_clamp(self, min: T, max: T) -> Self {
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
    #[inline]
    pub fn from_polar(radius: T, phi: T, theta: T) -> Self {
        let (s_theta, c_theta) = theta.sin_cos();
        let r_xz = s_theta * radius;
        let (s_phi, c_phi) = phi.sin_cos();
        Self::new(r_xz * c_phi, radius * c_theta, r_xz * s_phi)
    }

    #[inline]
    pub fn unit_polar(phi: T, theta: T) -> Self {
        Self::from_polar(T::one(), phi, theta)
    }

    #[inline]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    #[inline]
    pub fn ewise_recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip(), self.z.recip())
    }

    #[inline]
    pub fn normalised(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::zero());
        Self::new(self.x / n, self.y / n, self.z / n)
    }

    #[inline]
    pub fn normalised_fast(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::zero());
        let inv_n = T::one() / n;
        Self::new(self.x * inv_n, self.y * inv_n, self.z * inv_n)
    }

    #[inline]
    pub fn normalise(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::zero());
        self.x /= n;
        self.y /= n;
        self.z /= n;
    }

    #[inline]
    pub fn normalise_fast(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::zero());
        let inv_n = T::one() / n;
        self.x *= inv_n;
        self.y *= inv_n;
        self.z *= inv_n;
    }

    #[inline]
    pub fn ewise_lerp(self, t: T, end: Self) -> Self {
        Self::new(
            self.x.lerp(t, end.x),
            self.y.lerp(t, end.y),
            self.z.lerp(t, end.z),
        )
    }

    #[inline]
    pub fn angle_with(self, other: Self) -> T {
        let dot = self.dot(other);
        let norm_prod = self.norm() * other.norm();
        (dot / norm_prod).clamp(-T::one(), T::one()).acos()
    }
}

impl<T> Vector3<T>
where
    T: Abs + Float,
{
    #[inline]
    pub fn compute_perpendicular(self) -> Self {
        if self.x.abs() > self.y.abs() {
            let n = (self.x * self.x + self.z * self.z).sqrt();
            Self::new(-self.z / n, T::zero(), self.x / n)
        } else {
            let n = (self.y * self.y + self.z * self.z).sqrt();
            Self::new(T::zero(), self.z / n, -self.y / n)
        }
    }
}

impl<T> Vector3<T>
where
    T: ApproxEq,
{
    #[inline]
    pub fn ewise_approx_eq(self, other: Self) -> Vec3Bool {
        Vec3Bool::new(
            self.x.approx_eq(other.x),
            self.y.approx_eq(other.y),
            self.z.approx_eq(other.z),
        )
    }

    #[inline]
    pub fn approx_eq(self, other: Self) -> bool {
        self.ewise_approx_eq(other).all()
    }

    #[inline]
    pub fn ewise_approx_zero(self) -> Vec3Bool {
        Vec3Bool::new(
            self.x.approx_zero(),
            self.y.approx_zero(),
            self.z.approx_zero(),
        )
    }

    #[inline]
    pub fn approx_zero(self) -> bool {
        self.ewise_approx_zero().all()
    }
}

impl<T> Add for Vector3<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T> AddAssign for Vector3<T>
where
    T: AddAssign,
{
    #[inline]
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

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<T> SubAssign for Vector3<T>
where
    T: SubAssign,
{
    #[inline]
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

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

macro_rules! impl_mul {
    ($t:ty) => {
        impl Mul<Vector3<$t>> for $t {
            type Output = Vector3<$t>;

            #[inline]
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
    #[inline]
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

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<T> Div<T> for Vector3<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<T> DivAssign<T> for Vector3<T>
where
    T: Copy + DivAssign,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl<T> Index<Axis3> for Vector3<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Axis3) -> &Self::Output {
        match index {
            Axis3::X => &self.x,
            Axis3::Y => &self.y,
            Axis3::Z => &self.z,
        }
    }
}

impl<T> IndexMut<Axis3> for Vector3<T> {
    #[inline]
    fn index_mut(&mut self, index: Axis3) -> &mut Self::Output {
        match index {
            Axis3::X => &mut self.x,
            Axis3::Y => &mut self.y,
            Axis3::Z => &mut self.z,
        }
    }
}

impl<T> From<Vector3<T>> for (T, T, T) {
    #[inline]
    fn from(v: Vector3<T>) -> Self {
        (v.x, v.y, v.z)
    }
}
