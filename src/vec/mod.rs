pub use vec2::*;
pub use vec3::*;

pub mod vec2;
pub mod vec3;

use std::cmp::PartialOrd;

/// Helper function to compute the minimum of two values assuming there are no NaNs involved.
fn min_helper<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

/// Helper function to compute the minimum of 3 values assuming there are no NaNs involved.
fn min3_helper<T: PartialOrd>(a: T, b: T, c: T) -> T {
    min_helper(a, min_helper(b, c))
}

/// Helper function to compute the maximum of two values assuming there are no NaNs involved.
fn max_helper<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

/// Helper function to compute the maximum of 3 values assuming there are no NaNs involved.
fn max3_helper<T: PartialOrd>(a: T, b: T, c: T) -> T {
    max_helper(a, max_helper(b, c))
}
