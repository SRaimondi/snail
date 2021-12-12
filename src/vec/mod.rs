pub use vec2::*;
pub use vec3::*;

pub mod vec2;
pub mod vec3;

use std::cmp::PartialOrd;

fn min_helper<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

fn min3_helper<T: PartialOrd>(a: T, b: T, c: T) -> T {
    min_helper(a, min_helper(b, c))
}

fn max_helper<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

fn max3_helper<T: PartialOrd>(a: T, b: T, c: T) -> T {
    max_helper(a, max_helper(b, c))
}
