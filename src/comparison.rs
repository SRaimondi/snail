pub trait ApproxEq: Sized {
    const DEFAULT_RELATIVE_EPS: Self;
    const DEFAULT_ABSOLUTE_EPS: Self;

    fn approx_eq_eps(self, other: Self, rel_eps: Self, abs_eps: Self) -> bool;

    #[inline(always)]
    fn approx_eq_rel_eps(self, other: Self, rel_eps: Self) -> bool {
        self.approx_eq_eps(other, rel_eps, Self::DEFAULT_ABSOLUTE_EPS)
    }

    #[inline(always)]
    fn approx_eq_abs_eps(self, other: Self, abs_eps: Self) -> bool {
        self.approx_eq_eps(other, Self::DEFAULT_RELATIVE_EPS, abs_eps)
    }

    #[inline(always)]
    fn approx_eq(self, other: Self) -> bool {
        self.approx_eq_eps(
            other,
            Self::DEFAULT_RELATIVE_EPS,
            Self::DEFAULT_ABSOLUTE_EPS,
        )
    }

    fn approx_zero_eps(self, eps: Self) -> bool;

    #[inline(always)]
    fn approx_zero(self) -> bool {
        self.approx_zero_eps(Self::DEFAULT_ABSOLUTE_EPS)
    }
}

macro_rules! generate_approx_eq {
    ($t:ty, $def_rel_eps:expr, $def_abs_eps:expr) => {
        impl ApproxEq for $t {
            const DEFAULT_RELATIVE_EPS: Self = $def_rel_eps;
            const DEFAULT_ABSOLUTE_EPS: Self = $def_abs_eps;

            #[inline(always)]
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

            #[inline(always)]
            fn approx_zero_eps(self, eps: Self) -> bool {
                self.abs() <= eps
            }
        }
    };
}

generate_approx_eq!(f32, f32::EPSILON, 1e-5);
generate_approx_eq!(f64, f64::EPSILON, 1e-12);
