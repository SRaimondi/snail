use std::{
    f32, f64,
    ops::{Add, AddAssign, Div, DivAssign, Mul, Neg, Sub, SubAssign},
};

macro_rules! generate_complex {
    ($name:ident, $t:ty, $pi:expr) => {
        #[derive(Copy, Clone, Debug, Default, PartialEq)]
        #[repr(C)]
        pub struct $name {
            pub real: $t,
            pub im: $t,
        }

        impl $name {
            /// Create complex number from the given values.
            #[inline(always)]
            pub const fn new(real: $t, im: $t) -> Self {
                Self { real, im }
            }

            /// Create complex number from polar representation.
            #[inline(always)]
            pub fn from_polar(radius: $t, angle: $t) -> Self {
                let (s, c) = angle.sin_cos();
                Self::new(c * radius, s * radius)
            }

            /// Create complex number with radius 1 and for the given angle.
            #[inline(always)]
            pub fn from_angle(angle: $t) -> Self {
                Self::from_polar(1.0, angle)
            }

            /// Compute squared norm.
            #[inline(alwawys)]
            pub fn norm_squared(self) -> $t {
                self.real * self.real + self.im * self.im
            }

            /// Compute norm.
            #[inline(alwawys)]
            pub fn norm(self) -> $t {
                self.norm_squared().sqrt()
            }

            /// Compute conjugate.
            #[inline(always)]
            pub fn conjugate(self) -> Self {
                Self::new(self.real, -self.im)
            }

            /// Extract angle.
            #[inline(always)]
            pub fn angle(self) -> $t {
                let a = self.im.atan2(self.real);
                if a < 0.0 {
                    a + 2.0 * $pi
                } else {
                    a
                }
            }

            /// Compute reciprocal.
            #[inline(always)]
            pub fn recip(self) -> Self {
                let n2 = self.norm_squared();
                assert!(!float_cmp::approx_eq!($t, n2, 0.0));
                Self::new(self.real / n2, -self.im / n2)
            }

            /// Extract polar representation.
            #[inline(always)]
            pub fn polar(self) -> ($t, $t) {
                (self.norm(), self.angle())
            }

            /// Normalise complex number to have norm 1.
            #[inline(always)]
            pub fn normalise(&mut self) {
                let n = self.norm();
                assert!(n > 0.0);
                self.real /= n;
                self.im /= n;
            }

            /// Return a new Complex number with norm 1.
            #[inline(always)]
            pub fn normalised(self) -> Self {
                let n = self.norm();
                assert!(n > 0.0);
                Self::new(self.real / n, self.im / n)
            }
        }

        impl Add for Complex {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                Self::Output::new(self.real + rhs.real, self.im + rhs.im)
            }
        }

        impl AddAssign for Complex {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.real += rhs.real;
                self.im += rhs.im;
            }
        }

        impl Div for Complex {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self::Output {
                let n2 = rhs.norm_squared();
                let r = (self.real * rhs.real + self.im * rhs.im);
                let im = (self.im * rhs.real - self.real * rhs.im);
                Self::Output::new(r / n2, im / n2)
            }
        }

        impl Div<$t> for Complex {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.real / rhs, self.im / rhs)
            }
        }

        impl DivAssign<$t> for Complex {
            #[inline(always)]
            fn div_assign(&mut self, rhs: $t) {
                self.real /= rhs;
                self.im /= rhs;
            }
        }

        impl Mul for Complex {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self::Output {
                Self::Output::new(
                    self.real * rhs.real - self.im * rhs.im,
                    self.real * rhs.im + self.im * rhs.real,
                )
            }
        }

        impl Mul<$t> for Complex {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: f32) -> Self::Output {
                Self::Output::new(self.real * rhs, self.im * rhs)
            }
        }

        impl Mul<Complex> for $t {
            type Output = Complex;

            #[inline(always)]
            fn mul(self, rhs: Complex) -> Self::Output {
                Self::Output::new(self * rhs.real, self * rhs.im)
            }
        }

        impl Neg for Complex {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.real, -self.im)
            }
        }

        impl Sub for Complex {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::Output::new(self.real - rhs.real, self.im - rhs.im)
            }
        }

        impl SubAssign for Complex {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.real -= rhs.real;
                self.im -= rhs.im;
            }
        }
    };
}

generate_complex!(Complexf32, f32, f32::consts::PI);
generate_complex!(Complexf64, f64, f64::consts::PI);