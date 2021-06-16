use std::ops::{
    Add, AddAssign, Deref, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Axis3 {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis3 {
    #[inline]
    pub const fn next(self) -> Self {
        match self {
            Self::X => Self::Y,
            Self::Y => Self::Z,
            Self::Z => Self::X,
        }
    }
}

macro_rules! generate_vec3 {
    ($name:ident, $t:ty) => {
        #[derive(Copy, Clone, Debug, Default, PartialEq)]
        #[repr(C)]
        pub struct $name {
            elements: [$t; 3],
        }

        impl $name {
            // Vector construction
            #[inline]
            pub const fn new(x: $t, y: $t, z: $t) -> Self {
                Self {
                    elements: [x, y, z],
                }
            }

            #[inline]
            pub const fn broadcast(v: $t) -> Self {
                Self::new(v, v, v)
            }

            // Element access
            #[inline]
            pub const fn x(self) -> $t {
                self.elements[Axis3::X as usize]
            }

            #[inline]
            pub const fn y(self) -> $t {
                self.elements[Axis3::Y as usize]
            }

            #[inline]
            pub const fn z(self) -> $t {
                self.elements[Axis3::Z as usize]
            }

            #[inline]
            pub fn x_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis3::X as usize]
            }

            #[inline]
            pub fn y_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis3::Y as usize]
            }

            #[inline]
            pub fn z_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis3::Z as usize]
            }

            #[inline]
            pub const fn size(self) -> usize {
                3
            }

            /// # Safety
            /// The index should either be 0 or 1, otherwise we get undefined behaviour
            #[inline]
            pub unsafe fn get_unchecked(self, i: usize) -> $t {
                debug_assert!(i < self.size());
                *self.elements.get_unchecked(i)
            }

            /// # Safety
            /// The index should either be 0 or 1, otherwise we get undefined behaviour
            #[inline]
            pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut $t {
                debug_assert!(i < self.size());
                self.elements.get_unchecked_mut(i)
            }

            #[inline]
            pub fn get(self, i: usize) -> Option<$t> {
                if i < self.size() {
                    Some(unsafe { self.get_unchecked(i) })
                } else {
                    None
                }
            }

            #[inline]
            pub fn get_mut(&mut self, i: usize) -> Option<&mut $t> {
                if i < self.size() {
                    Some(unsafe { self.get_unchecked_mut(i) })
                } else {
                    None
                }
            }

            // Element wise operations
            #[inline]
            pub fn element_wise_min(self, rhs: Self) -> Self {
                Self::new(
                    self.x().min(rhs.x()),
                    self.y().min(rhs.y()),
                    self.z().min(rhs.z()),
                )
            }

            #[inline]
            pub fn element_wise_max(self, rhs: Self) -> Self {
                Self::new(
                    self.x().max(rhs.x()),
                    self.y().max(rhs.y()),
                    self.z().max(rhs.z()),
                )
            }

            #[inline]
            pub fn element_wise_product(self, rhs: Self) -> Self {
                Self::new(self.x() * rhs.x(), self.y() * rhs.y(), self.z() * rhs.z())
            }

            #[inline]
            pub fn element_wise_quotient(self, rhs: Self) -> Self {
                Self::new(self.x() / rhs.x(), self.y() / rhs.y(), self.z() / rhs.z())
            }

            #[inline]
            pub fn element_wise_abs(self) -> Self {
                Self::new(self.x().abs(), self.y().abs(), self.z().abs())
            }

            #[inline]
            pub fn element_wise_recip(self) -> Self {
                Self::new(self.x().recip(), self.y().recip(), self.z().recip())
            }

            #[inline]
            pub fn normalised(self) -> Self {
                let l = self.length();
                self / l
            }

            #[inline]
            pub fn normalised_fast(self) -> Self {
                let inv_l = 1.0 / self.length();
                inv_l * self
            }

            #[inline]
            pub fn normalise(&mut self) {
                let l = self.length();
                *self.x_mut() /= l;
                *self.y_mut() /= l;
                *self.z_mut() /= l;
            }

            #[inline]
            pub fn normalise_fast(&mut self) {
                let inv_l = 1.0 / self.length();
                *self.x_mut() *= inv_l;
                *self.y_mut() *= inv_l;
                *self.z_mut() *= inv_l;
            }

            #[inline]
            pub fn lerp(self, t: $t, end: Self) -> Self {
                self + t * (end - self)
            }

            #[inline]
            pub fn cross(self, rhs: Self) -> Self {
                Self::new(
                    self.y() * rhs.z() - self.z() * rhs.y(),
                    self.z() * rhs.x() - self.x() * rhs.z(),
                    self.x() * rhs.y() - self.y() * rhs.x(),
                )
            }

            // Scalar result operations
            #[inline]
            pub fn dot(self, rhs: Self) -> $t {
                self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z()
            }

            #[inline]
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }

            #[inline]
            pub fn length(self) -> $t {
                self.length_squared().sqrt()
            }

            #[inline]
            pub fn min_element(self) -> $t {
                self.x().min(self.y().min(self.z()))
            }

            #[inline]
            pub fn max_element(self) -> $t {
                self.x().max(self.y().max(self.z()))
            }

            // Permute vector elements
            #[inline]
            pub fn permute(self, x_axis: Axis3, y_axis: Axis3, z_axis: Axis3) -> Self {
                Self::new(self[x_axis], self[y_axis], self[z_axis])
            }

            #[inline]
            pub fn permute_with_array(self, axes: [Axis3; 3]) -> Self {
                self.permute(axes[0], axes[1], axes[2])
            }

            // Find largest axis in the vector
            #[inline]
            pub fn largest_axis(self) -> Axis3 {
                if self.x() >= self.y() && self.x() >= self.z() {
                    Axis3::X
                } else if self.y() >= self.z() {
                    Axis3::Y
                } else {
                    Axis3::Z
                }
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self::new(self.x() + rhs.x(), self.y() + rhs.y(), self.z() + rhs.z())
            }
        }

        impl AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self.x_mut() += rhs.x();
                *self.y_mut() += rhs.y();
                *self.z_mut() += rhs.z();
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
            }
        }

        impl SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self.x_mut() -= rhs.x();
                *self.y_mut() -= rhs.y();
                *self.z_mut() -= rhs.z();
            }
        }

        impl Mul<$name> for $t {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                Self::Output::new(self * rhs.x(), self * rhs.y(), self * rhs.z())
            }
        }

        impl Mul<$t> for $name {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x() * rhs, self.y() * rhs, self.z() * rhs)
            }
        }

        impl MulAssign<$t> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                *self.x_mut() *= rhs;
                *self.y_mut() *= rhs;
                *self.z_mut() *= rhs;
            }
        }

        impl Div<$t> for $name {
            type Output = Self;

            #[inline]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x() / rhs, self.y() / rhs, self.z() / rhs)
            }
        }

        impl DivAssign<$t> for $name {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                *self.x_mut() /= rhs;
                *self.y_mut() /= rhs;
                *self.z_mut() /= rhs;
            }
        }

        impl Deref for $name {
            type Target = [$t];

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.elements
            }
        }

        impl Index<Axis3> for $name {
            type Output = $t;

            #[inline]
            fn index(&self, index: Axis3) -> &Self::Output {
                &self.elements[index as usize]
            }
        }

        impl IndexMut<Axis3> for $name {
            #[inline]
            fn index_mut(&mut self, index: Axis3) -> &mut Self::Output {
                &mut self.elements[index as usize]
            }
        }
    };
}

generate_vec3!(Vec3f32, f32);
generate_vec3!(Vec3f64, f64);
