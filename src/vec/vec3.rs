use std::ops::{
    Add, AddAssign, Deref, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
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
            /// Create new vector from the given coordinates.
            #[inline(always)]
            pub const fn new(x: $t, y: $t, z: $t) -> Self {
                Self {
                    elements: [x, y, z],
                }
            }

            /// Create new vector with all components set to the given value.
            #[inline(always)]
            pub const fn broadcast(v: $t) -> Self {
                Self::new(v, v, v)
            }

            /// Create new vector from the given array.
            #[inline(always)]
            pub const fn new_from_array(components: [$t; 3]) -> Self {
                Self {
                    elements: components,
                }
            }

            /// Access x components by value.
            #[inline(always)]
            pub const fn x(self) -> $t {
                self.elements[Axis3::X as usize]
            }

            /// Access y components by value.
            #[inline(always)]
            pub const fn y(self) -> $t {
                self.elements[Axis3::Y as usize]
            }

            /// Access z components by value.
            #[inline(always)]
            pub const fn z(self) -> $t {
                self.elements[Axis3::Z as usize]
            }

            /// Access x components as mutable reference.
            #[inline(always)]
            pub fn x_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis3::X as usize]
            }

            /// Access y components as mutable reference.
            #[inline(always)]
            pub fn y_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis3::Y as usize]
            }

            /// Access z components as mutable reference.
            #[inline(always)]
            pub fn z_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis3::Z as usize]
            }

            /// Number of elements in the vector.
            #[inline(always)]
            pub const fn len(self) -> usize {
                3
            }

            /// Access a component by value using an index.
            /// # Safety
            /// The index should either be 0, 1 or 2, otherwise we get undefined behaviour.
            #[inline(always)]
            pub unsafe fn get_unchecked(self, i: usize) -> $t {
                debug_assert!(i < self.len());
                *self.elements.get_unchecked(i)
            }

            /// Access a component by index and returns a mutable reference to it.
            /// # Safety
            /// The index should either be 0, 1 or 2, otherwise we get undefined behaviour.
            #[inline(always)]
            pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut $t {
                debug_assert!(i < self.len());
                self.elements.get_unchecked_mut(i)
            }

            /// Access a component by value using an index.
            /// Returns None if the index is out of range.
            #[inline(always)]
            pub fn get(self, i: usize) -> Option<$t> {
                if i < self.len() {
                    Some(unsafe { self.get_unchecked(i) })
                } else {
                    None
                }
            }

            /// Access a component by mutable reference using an index.
            /// Returns None if the index is out of range.
            #[inline(always)]
            pub fn get_mut(&mut self, i: usize) -> Option<&mut $t> {
                if i < self.len() {
                    Some(unsafe { self.get_unchecked_mut(i) })
                } else {
                    None
                }
            }

            /// Compute minimum value for each component.
            #[inline(always)]
            pub fn element_wise_min(self, rhs: Self) -> Self {
                Self::new(
                    self.x().min(rhs.x()),
                    self.y().min(rhs.y()),
                    self.z().min(rhs.z()),
                )
            }

            /// Compute maximum value for each component.
            #[inline(always)]
            pub fn element_wise_max(self, rhs: Self) -> Self {
                Self::new(
                    self.x().max(rhs.x()),
                    self.y().max(rhs.y()),
                    self.z().max(rhs.z()),
                )
            }

            /// Compute product for each component.
            #[inline(always)]
            pub fn element_wise_product(self, rhs: Self) -> Self {
                Self::new(self.x() * rhs.x(), self.y() * rhs.y(), self.z() * rhs.z())
            }

            /// Compute quotient for each component.
            #[inline(always)]
            pub fn element_wise_quotient(self, rhs: Self) -> Self {
                Self::new(self.x() / rhs.x(), self.y() / rhs.y(), self.z() / rhs.z())
            }

            /// Compute absolute value for each component.
            #[inline(always)]
            pub fn element_wise_abs(self) -> Self {
                Self::new(self.x().abs(), self.y().abs(), self.z().abs())
            }

            /// Compute reciprocal value for each component.
            #[inline(always)]
            pub fn element_wise_recip(self) -> Self {
                Self::new(self.x().recip(), self.y().recip(), self.z().recip())
            }

            /// Clamp each value between the given min and max.
            #[inline(always)]
            pub fn element_wise_clamp(self, min: $t, max: $t) -> Self {
                Self::new(
                    self.x().clamp(min, max),
                    self.y().clamp(min, max),
                    self.z().clamp(min, max),
                )
            }

            /// Return a new normalised vector.
            #[inline(always)]
            pub fn normalised(self) -> Self {
                let l = self.length();
                self / l
            }

            /// Return a new normalised vector, uses multiplication instead of division on the components.
            #[inline(always)]
            pub fn normalised_fast(self) -> Self {
                let inv_l = 1.0 / self.length();
                inv_l * self
            }

            /// Normalise vector in place.
            #[inline(always)]
            pub fn normalise(&mut self) {
                let l = self.length();
                *self.x_mut() /= l;
                *self.y_mut() /= l;
                *self.z_mut() /= l;
            }

            /// Normalise vector in place using multiplication.
            #[inline(always)]
            pub fn normalise_fast(&mut self) {
                let inv_l = 1.0 / self.length();
                *self.x_mut() *= inv_l;
                *self.y_mut() *= inv_l;
                *self.z_mut() *= inv_l;
            }

            /// Linearly interpolate for each component.
            #[inline(always)]
            pub fn lerp(self, t: $t, end: Self) -> Self {
                (1.0 - t) * self + t * end
            }

            /// Compute cross product.
            #[inline(always)]
            pub fn cross(self, rhs: Self) -> Self {
                Self::new(
                    self.y() * rhs.z() - self.z() * rhs.y(),
                    self.z() * rhs.x() - self.x() * rhs.z(),
                    self.x() * rhs.y() - self.y() * rhs.x(),
                )
            }

            /// Compute dot product.
            #[inline(always)]
            pub fn dot(self, rhs: Self) -> $t {
                self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z()
            }

            /// Compute squared length of the vector.
            #[inline(always)]
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }

            /// Compute length of the vector.
            #[inline(always)]
            pub fn length(self) -> $t {
                self.length_squared().sqrt()
            }

            /// Compute minimum element.
            #[inline(always)]
            pub fn min_element(self) -> $t {
                self.x().min(self.y().min(self.z()))
            }

            /// Compute maximum element.
            #[inline(always)]
            pub fn max_element(self) -> $t {
                self.x().max(self.y().max(self.z()))
            }

            /// Permute components for the given new axes.
            #[inline(always)]
            pub fn permute(self, x_axis: Axis3, y_axis: Axis3, z_axis: Axis3) -> Self {
                Self::new(self[x_axis], self[y_axis], self[z_axis])
            }

            /// Permute components for the given new axes as array.
            #[inline(always)]
            pub fn permute_with_array(self, axes: [Axis3; 3]) -> Self {
                self.permute(axes[0], axes[1], axes[2])
            }

            /// Compute largest axis of the vector.
            #[inline(always)]
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

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self::new(self.x() + rhs.x(), self.y() + rhs.y(), self.z() + rhs.z())
            }
        }

        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self.x_mut() += rhs.x();
                *self.y_mut() += rhs.y();
                *self.z_mut() += rhs.z();
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
            }
        }

        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self.x_mut() -= rhs.x();
                *self.y_mut() -= rhs.y();
                *self.z_mut() -= rhs.z();
            }
        }

        impl Mul<$name> for $t {
            type Output = $name;

            #[inline(always)]
            fn mul(self, rhs: $name) -> Self::Output {
                Self::Output::new(self * rhs.x(), self * rhs.y(), self * rhs.z())
            }
        }

        impl Mul<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x() * rhs, self.y() * rhs, self.z() * rhs)
            }
        }

        impl MulAssign<$t> for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: $t) {
                *self.x_mut() *= rhs;
                *self.y_mut() *= rhs;
                *self.z_mut() *= rhs;
            }
        }

        impl Neg for $name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.x(), -self.y(), -self.z())
            }
        }

        impl Div<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x() / rhs, self.y() / rhs, self.z() / rhs)
            }
        }

        impl DivAssign<$t> for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: $t) {
                *self.x_mut() /= rhs;
                *self.y_mut() /= rhs;
                *self.z_mut() /= rhs;
            }
        }

        impl Deref for $name {
            type Target = [$t];

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.elements
            }
        }

        impl Index<Axis3> for $name {
            type Output = $t;

            #[inline(always)]
            fn index(&self, index: Axis3) -> &Self::Output {
                &self.elements[index as usize]
            }
        }

        impl IndexMut<Axis3> for $name {
            #[inline(always)]
            fn index_mut(&mut self, index: Axis3) -> &mut Self::Output {
                &mut self.elements[index as usize]
            }
        }
    };
}

generate_vec3!(Vec3f32, f32);
generate_vec3!(Vec3f64, f64);
