pub use vec2::*;
pub use vec3::*;

pub mod vec2;
pub mod vec3;

pub type Vec2u8 = Vector2<u8>;
pub type Vec2u16 = Vector2<u16>;
pub type Vec2u32 = Vector2<u32>;
pub type Vec2u64 = Vector2<u64>;
pub type Vec2u128 = Vector2<u128>;
pub type Vec2usize = Vector2<usize>;

pub type Vec2i8 = Vector2<i8>;
pub type Vec2i16 = Vector2<i16>;
pub type Vec2i32 = Vector2<i32>;
pub type Vec2i64 = Vector2<i64>;
pub type Vec2i128 = Vector2<i128>;
pub type Vec2iSize = Vector2<isize>;

pub type Vec2f32 = Vector2<f32>;
pub type Vec2f64 = Vector2<f64>;

pub type Vec3u8 = Vector3<u8>;
pub type Vec3u16 = Vector3<u16>;
pub type Vec3u32 = Vector3<u32>;
pub type Vec3u64 = Vector3<u64>;
pub type Vec3u128 = Vector3<u128>;
pub type Vec3usize = Vector3<usize>;

pub type Vec3i8 = Vector3<i8>;
pub type Vec3i16 = Vector3<i16>;
pub type Vec3i32 = Vector3<i32>;
pub type Vec3i64 = Vector3<i64>;
pub type Vec3i128 = Vector3<i128>;
pub type Vec3iSize = Vector3<isize>;

pub type Vec3f32 = Vector3<f32>;
pub type Vec3f64 = Vector3<f64>;

/// Trait used for types that can return the value 0.
pub trait Zero {
    const ZERO: Self;
}

macro_rules! impl_zero {
    ($t:ty, $zero:expr) => {
        impl Zero for $t {
            const ZERO: Self = $zero;
        }
    };
}
impl_zero!(u8, 0);
impl_zero!(u16, 0);
impl_zero!(u32, 0);
impl_zero!(u64, 0);
impl_zero!(u128, 0);
impl_zero!(usize, 0);

impl_zero!(i8, 0);
impl_zero!(i16, 0);
impl_zero!(i32, 0);
impl_zero!(i64, 0);
impl_zero!(i128, 0);
impl_zero!(isize, 0);

impl_zero!(f32, 0.0);
impl_zero!(f64, 0.0);

/// Trait used for types that can return the value 1.
pub trait One {
    const ONE: Self;
}

macro_rules! impl_one {
    ($t:ty, $one:expr) => {
        impl One for $t {
            const ONE: Self = $one;
        }
    };
}

impl_one!(u8, 1);
impl_one!(u16, 1);
impl_one!(u32, 1);
impl_one!(u64, 1);
impl_one!(u128, 1);
impl_one!(usize, 1);

impl_one!(i8, 1);
impl_one!(i16, 1);
impl_one!(i32, 1);
impl_one!(i64, 1);
impl_one!(i128, 1);
impl_one!(isize, 1);

impl_one!(f32, 1.0);
impl_one!(f64, 1.0);

/// Trait used for types that can compute the absolute value.
pub trait Abs {
    fn abs(self) -> Self;
}

macro_rules! impl_abs {
    ($t:ty) => {
        impl Abs for $t {
            #[inline(always)]
            fn abs(self) -> Self {
                self.abs()
            }
        }
    };
}

impl_abs!(i8);
impl_abs!(i16);
impl_abs!(i32);
impl_abs!(i64);
impl_abs!(i128);

impl_abs!(f32);
impl_abs!(f64);

use std::ops::{Add, Div, DivAssign, Mul, MulAssign, Neg, Sub};

/// Trait used for types that can do floating point operations (f32 and f64).
pub trait Float:
    Zero
    + One
    + Copy
    + PartialOrd
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + MulAssign
    + DivAssign
{
    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn acos(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn lerp(self, t: Self, end: Self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            #[inline(always)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn recip(self) -> Self {
                self.recip()
            }

            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                self.sin_cos()
            }

            #[inline(always)]
            fn acos(self) -> Self {
                self.acos()
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                self.min(other)
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                self.max(other)
            }

            #[inline(always)]
            fn clamp(self, min: Self, max: Self) -> Self {
                self.clamp(min, max)
            }

            #[inline(always)]
            fn lerp(self, t: Self, end: Self) -> Self {
                (1.0 - t) * self + t * end
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
