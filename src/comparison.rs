/// Trait to compare the type with itself for some given epsilons.
pub trait ApproxEq: Sized {
    /// Default epsilon for relative comparison.
    const DEFAULT_RELATIVE_EPS: Self;
    /// Default epsilon for absolute comparison.
    const DEFAULT_ABSOLUTE_EPS: Self;

    /// Compute with the other value using the given epsilon values.
    fn approx_eq_eps(self, other: Self, rel_eps: Self, abs_eps: Self) -> bool;

    /// Compare with the other value using the given relative epsilon and the default absolute one.
    #[inline]
    fn approx_eq_rel_eps(self, other: Self, rel_eps: Self) -> bool {
        self.approx_eq_eps(other, rel_eps, Self::DEFAULT_ABSOLUTE_EPS)
    }

    /// Compare with the other value using the given absolute epsilon and the default relative one.
    #[inline]
    fn approx_eq_abs_eps(self, other: Self, abs_eps: Self) -> bool {
        self.approx_eq_eps(other, Self::DEFAULT_RELATIVE_EPS, abs_eps)
    }

    /// Compare with the other value using the default epsilon values.
    #[inline]
    fn approx_eq(self, other: Self) -> bool {
        self.approx_eq_eps(
            other,
            Self::DEFAULT_RELATIVE_EPS,
            Self::DEFAULT_ABSOLUTE_EPS,
        )
    }

    /// Check if the value is zero for the given epsilon.
    fn approx_zero_eps(self, eps: Self) -> bool;

    /// Check if the value is zero for the default absolute epsilon.
    #[inline]
    fn approx_zero(self) -> bool {
        self.approx_zero_eps(Self::DEFAULT_ABSOLUTE_EPS)
    }
}

macro_rules! generate_approx_eq {
    ($t:ty, $def_rel_eps:expr, $def_abs_eps:expr) => {
        impl ApproxEq for $t {
            const DEFAULT_RELATIVE_EPS: Self = $def_rel_eps;
            const DEFAULT_ABSOLUTE_EPS: Self = $def_abs_eps;

            #[inline]
            fn approx_eq_eps(self, other: Self, rel_eps: Self, abs_eps: Self) -> bool {
                let diff_abs = (self - other).abs();
                if diff_abs <= abs_eps {
                    return true;
                }
                let self_abs = self.abs();
                let other_abs = other.abs();
                let largest = self_abs.max(other_abs);
                diff_abs <= largest * rel_eps
            }

            #[inline]
            fn approx_zero_eps(self, eps: Self) -> bool {
                self.abs() <= eps
            }
        }
    };
}

generate_approx_eq!(f32, f32::EPSILON, 1e-5);
generate_approx_eq!(f64, f64::EPSILON, 1e-12);
