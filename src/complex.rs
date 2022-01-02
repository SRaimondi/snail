use crate::ApproxEq;
use std::{
    f32, f64,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

macro_rules! generate_complex {
    ($cartesian_name:ident, $polar_name:ident, $t:ty, $pi:expr, $tau:expr) => {
        /// Cartesian complex number representation.
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(C)]
        pub struct $cartesian_name {
            pub real: $t,
            pub im: $t,
        }

        impl $cartesian_name {
            /// Create complex number from the given values.
            #[inline(always)]
            pub const fn new(real: $t, im: $t) -> Self {
                Self { real, im }
            }

            /// Convert to polar form with angle in the range [0, tau)
            #[inline(always)]
            pub fn to_polar(self) -> $polar_name {
                let a = self.im.atan2(self.real);
                let angle = if a < 0.0 { a + $tau } else { a };
                $polar_name::new(self.norm(), angle)
            }

            /// Check for equality with other.
            #[inline(always)]
            pub fn approx_eq(self, other: Self) -> bool {
                self.real.approx_eq(other.real) && self.im.approx_eq(other.im)
            }

            /// Compute squared norm.
            #[inline(always)]
            pub fn norm_squared(self) -> $t {
                self.real * self.real + self.im * self.im
            }

            /// Compute norm.
            #[inline(always)]
            pub fn norm(self) -> $t {
                self.norm_squared().sqrt()
            }

            /// Compute conjugate.
            #[inline(always)]
            pub fn conjugate(self) -> Self {
                Self::new(self.real, -self.im)
            }

            /// Compute reciprocal.
            #[inline(always)]
            pub fn recip(self) -> Self {
                let n2 = self.norm_squared();
                debug_assert!(n2 > 0.0);
                Self::new(self.real / n2, -self.im / n2)
            }

            /// Normalise complex number to have norm 1.
            #[inline(always)]
            pub fn normalise(&mut self) {
                let n = self.norm();
                debug_assert!(n > 0.0);
                self.real /= n;
                self.im /= n;
            }

            /// Normalise complex number to have norm 1 using multiplication by inverse.
            #[inline(always)]
            pub fn normalise_fast(&mut self) {
                let n = self.norm();
                debug_assert!(n > 0.0);
                let recip_n = n.recip();
                self.real *= recip_n;
                self.im *= recip_n;
            }

            /// Returns the complex number after being normalised.
            #[inline(always)]
            pub fn normalised(self) -> Self {
                let n = self.norm();
                debug_assert!(n > 0.0);
                self / n
            }

            /// Return the complex number after being normalised, uses multiplication instead of division.
            #[inline(always)]
            pub fn normalised_fast(self) -> Self {
                let n = self.norm();
                debug_assert!(n > 0.0);
                n.recip() * self
            }
        }

        impl Add for $cartesian_name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                Self::Output::new(self.real + rhs.real, self.im + rhs.im)
            }
        }

        impl AddAssign for $cartesian_name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.real += rhs.real;
                self.im += rhs.im;
            }
        }

        impl Sub for $cartesian_name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::Output::new(self.real - rhs.real, self.im - rhs.im)
            }
        }

        impl SubAssign for $cartesian_name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.real -= rhs.real;
                self.im -= rhs.im;
            }
        }

        impl Mul for $cartesian_name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self::Output {
                Self::Output::new(
                    self.real * rhs.real - self.im * rhs.im,
                    self.real * rhs.im + self.im * rhs.real,
                )
            }
        }

        impl Mul<$t> for $cartesian_name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.real * rhs, self.im * rhs)
            }
        }

        impl Mul<$cartesian_name> for $t {
            type Output = $cartesian_name;

            #[inline(always)]
            fn mul(self, rhs: $cartesian_name) -> Self::Output {
                Self::Output::new(self * rhs.real, self * rhs.im)
            }
        }

        impl Div for $cartesian_name {
            type Output = Self;

            #[inline(always)]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn div(self, rhs: Self) -> Self::Output {
                self * rhs.recip()
            }
        }

        impl Div<$t> for $cartesian_name {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.real / rhs, self.im / rhs)
            }
        }

        impl Neg for $cartesian_name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.real, -self.im)
            }
        }

        /// Polar complex number representation.
        #[derive(Copy, Clone, Debug, Default)]
        #[repr(C)]
        pub struct $polar_name {
            pub radius: $t,
            pub angle: $t,
        }

        impl $polar_name {
            /// Create polar complex number from the given values.
            #[inline(always)]
            pub fn new(radius: $t, angle: $t) -> Self {
                debug_assert!(radius >= 0.0);
                Self { radius, angle }
            }

            /// Create polar complex number of radius 1 for the given angle.
            #[inline(always)]
            pub fn unit_from_angle(angle: $t) -> Self {
                Self::new(1.0, angle)
            }

            /// Normalise complex number to have norm 1.
            #[inline(always)]
            pub fn normalise(&mut self) {
                self.radius = 1.0;
            }

            /// Return a new Complex number with norm 1.
            #[inline(always)]
            pub fn normalised(self) -> Self {
                Self::new(1.0, self.angle)
            }

            /// Compute nth root. Returns the first root and the angle period as second element of the tuple.
            #[inline(always)]
            pub fn nth_root(self, n: u32) -> (Self, $t) {
                let n = n as $t;
                let radius = self.radius.powf(1.0 / n);
                let angle = self.angle / n;
                let period = $tau / n;
                (Self { radius, angle }, period)
            }

            /// Create cartesian complex number.
            #[inline(always)]
            pub fn to_cartesian(self) -> $cartesian_name {
                let (s, c) = self.angle.sin_cos();
                $cartesian_name::new(c * self.radius, s * self.radius)
            }
        }

        impl Mul for $polar_name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self::Output {
                Self::Output::new(self.radius * rhs.radius, self.angle + rhs.angle)
            }
        }

        impl Mul<$t> for $polar_name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.radius * rhs, self.angle)
            }
        }

        impl Mul<$polar_name> for $t {
            type Output = $polar_name;

            #[inline(always)]
            fn mul(self, rhs: $polar_name) -> Self::Output {
                Self::Output::new(self * rhs.radius, rhs.angle)
            }
        }

        impl Div for $polar_name {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self::Output {
                debug_assert!(rhs.radius > 0.0);
                Self::Output::new(self.radius / rhs.radius, self.angle - rhs.angle)
            }
        }

        impl Neg for $polar_name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(self.radius, self.angle + $pi)
            }
        }
    };
}

generate_complex!(
    ComplexCartesianf32,
    ComplexPolarf32,
    f32,
    f32::consts::PI,
    f32::consts::TAU
);
generate_complex!(
    ComplexCartesianf64,
    ComplexPolarf64,
    f64,
    f64::consts::PI,
    f64::consts::TAU
);
