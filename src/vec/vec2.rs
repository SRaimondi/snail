use crate::{Abs, ApproxEq, Float, One, Zero};

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
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
pub struct BoolVector2 {
    pub x: bool,
    pub y: bool,
}

impl BoolVector2 {
    #[inline]
    fn new(x: bool, y: bool) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn all(self) -> bool {
        self.x && self.y
    }

    #[inline]
    pub fn any(self) -> bool {
        self.x || self.y
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vector2<T> {
    #[inline]
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    #[inline]
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
    #[inline]
    pub fn broadcast(v: T) -> Self {
        Self::new(v, v)
    }

    #[inline]
    pub fn permute(self, x_axis: Axis2, y_axis: Axis2) -> Self {
        Self::new(self.axis(x_axis), self.axis(y_axis))
    }

    #[inline]
    pub fn permute_with_array(self, axes: [Axis2; 2]) -> Self {
        self.permute(axes[0], axes[1])
    }
}

impl<T> Vector2<T>
where
    T: PartialOrd,
{
    #[inline]
    pub fn ewise_lt(self, other: Self) -> BoolVector2 {
        BoolVector2::new(self.x < other.x, self.y < other.y)
    }

    #[inline]
    pub fn ewise_le(self, other: Self) -> BoolVector2 {
        BoolVector2::new(self.x <= other.x, self.y <= other.y)
    }

    #[inline]
    pub fn ewise_gt(self, other: Self) -> BoolVector2 {
        BoolVector2::new(self.x > other.x, self.y > other.y)
    }

    #[inline]
    pub fn ewise_ge(self, other: Self) -> BoolVector2 {
        BoolVector2::new(self.x >= other.x, self.y >= other.y)
    }

    #[inline]
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
    #[inline]
    pub fn ewise_product(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y)
    }
}

impl<T> Vector2<T>
where
    T: Div<Output = T>,
{
    #[inline]
    pub fn ewise_quotient(self, other: Self) -> Self {
        Self::new(self.x / other.x, self.y / other.y)
    }
}

impl<T> Vector2<T>
where
    T: Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }
}

impl<T> Vector2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }
}

impl<T> Vector2<T>
where
    T: Zero,
{
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
}

impl<T> Vector2<T>
where
    T: Zero + One,
{
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero())
    }

    #[inline]
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one())
    }
}

impl<T> Vector2<T>
where
    T: Abs,
{
    #[inline]
    pub fn ewise_abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }
}

impl<T> Vector2<T>
where
    T: Ord,
{
    #[inline]
    pub fn ewise_min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    #[inline]
    pub fn ewise_max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    #[inline]
    pub fn min_element(self) -> T {
        self.x.min(self.y)
    }

    #[inline]
    pub fn max_element(self) -> T {
        self.x.max(self.y)
    }
}

impl<T> Vector2<T>
where
    T: Copy + Ord,
{
    #[inline]
    pub fn ewise_clamp(self, min: T, max: T) -> Self {
        Self::new(self.x.clamp(min, max), self.y.clamp(min, max))
    }
}

impl<T> Vector2<T>
where
    T: Float,
{
    #[inline]
    pub fn from_polar(radius: T, angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(radius * c, radius * s)
    }

    #[inline]
    pub fn unit_polar(angle: T) -> Self {
        Self::from_polar(T::one(), angle)
    }

    #[inline]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    #[inline]
    pub fn ewise_recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip())
    }

    #[inline]
    pub fn normalised(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::zero());
        Self::new(self.x / n, self.y / n)
    }

    #[inline]
    pub fn normalised_fast(self) -> Self {
        let n = self.norm();
        debug_assert!(n > T::zero());
        let inv_n = T::one() / n;
        Self::new(self.x * inv_n, self.y * inv_n)
    }

    #[inline]
    pub fn normalise(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::zero());
        self.x /= n;
        self.y /= n;
    }

    #[inline]
    pub fn normalise_fast(&mut self) {
        let n = self.norm();
        debug_assert!(n > T::zero());
        let inv_n = T::one() / n;
        self.x *= inv_n;
        self.y *= inv_n;
    }

    #[inline]
    pub fn ewise_lerp(self, t: T, end: Self) -> Self {
        Self::new(self.x.lerp(t, end.x), self.y.lerp(t, end.y))
    }

    #[inline]
    pub fn angle_with(self, other: Self) -> T {
        let dot = self.dot(other);
        let norm_prod = self.norm() * other.norm();
        (dot / norm_prod).clamp(-T::one(), T::one()).acos()
    }

    #[inline]
    pub fn rotate(self, angle_rad: T) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }
}

impl<T> Vector2<T>
where
    T: ApproxEq,
{
    #[inline]
    pub fn ewise_approx_eq(self, other: Self) -> BoolVector2 {
        BoolVector2::new(self.x.approx_eq(other.x), self.y.approx_eq(other.y))
    }

    #[inline]
    pub fn approx_eq(self, other: Self) -> bool {
        self.ewise_approx_eq(other).all()
    }

    #[inline]
    pub fn ewise_approx_zero(self) -> BoolVector2 {
        BoolVector2::new(self.x.approx_zero(), self.y.approx_zero())
    }

    #[inline]
    pub fn approx_zero(self) -> bool {
        self.ewise_approx_zero().all()
    }
}

impl<T> Add for Vector2<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<T> AddAssign for Vector2<T>
where
    T: AddAssign,
{
    #[inline]
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

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<T> SubAssign for Vector2<T>
where
    T: SubAssign,
{
    #[inline]
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

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::new(self.x * rhs, self.y * rhs)
    }
}

macro_rules! impl_mul {
    ($t:ty) => {
        impl Mul<Vector2<$t>> for $t {
            type Output = Vector2<$t>;

            #[inline]
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
    #[inline]
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

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl<T> Div<T> for Vector2<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl<T> DivAssign<T> for Vector2<T>
where
    T: Copy + DivAssign,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T> Index<Axis2> for Vector2<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Axis2) -> &Self::Output {
        match index {
            Axis2::X => &self.x,
            Axis2::Y => &self.y,
        }
    }
}

impl<T> IndexMut<Axis2> for Vector2<T> {
    #[inline]
    fn index_mut(&mut self, index: Axis2) -> &mut Self::Output {
        match index {
            Axis2::X => &mut self.x,
            Axis2::Y => &mut self.y,
        }
    }
}
