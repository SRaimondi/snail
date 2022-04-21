pub use vec2::*;
pub use vec3::*;

pub mod vec2;
pub mod vec3;

pub type Vector2U8 = Vector2<u8>;
pub type Vector2U16 = Vector2<u16>;
pub type Vector2U32 = Vector2<u32>;
pub type Vector2U64 = Vector2<u64>;
pub type Vector2U128 = Vector2<u128>;
pub type Vector2USize = Vector2<usize>;

pub type Vector2I8 = Vector2<i8>;
pub type Vector2I16 = Vector2<i16>;
pub type Vector2I32 = Vector2<i32>;
pub type Vector2I64 = Vector2<i64>;
pub type Vector2I128 = Vector2<i128>;
pub type Vector2ISize = Vector2<isize>;

pub type Vector2F32 = Vector2<f32>;
pub type Vector2F64 = Vector2<f64>;

pub trait Zero {
    fn zero() -> Self;
}

macro_rules! impl_zero {
    ($t:ty, $z:expr) => {
        impl Zero for $t {
            #[inline]
            fn zero() -> Self {
                $z
            }
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

pub trait One {
    fn one() -> Self;
}

macro_rules! impl_one {
    ($t:ty, $o:expr) => {
        impl One for $t {
            #[inline]
            fn one() -> Self {
                $o
            }
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

pub trait Abs {
    fn abs(self) -> Self;
}

macro_rules! impl_abs {
    ($t:ty) => {
        impl Abs for $t {
            #[inline]
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
            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline]
            fn recip(self) -> Self {
                self.recip()
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                self.sin_cos()
            }

            #[inline]
            fn acos(self) -> Self {
                self.acos()
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                self.min(other)
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                self.max(other)
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                self.clamp(min, max)
            }

            #[inline]
            fn lerp(self, t: Self, end: Self) -> Self {
                (1.0 - t) * self + t * end
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
