use std::ops::{
    Add, AddAssign, Deref, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Axis2 {
    X = 0,
    Y = 1,
}

impl Axis2 {
    #[inline]
    pub const fn next(self) -> Self {
        match self {
            Self::X => Self::Y,
            Self::Y => Self::X,
        }
    }
}

macro_rules! generate_vec2 {
    ($name:ident, $t:ty) => {
        #[derive(Copy, Clone, Debug, Default, PartialEq)]
        #[repr(C)]
        pub struct $name {
            elements: [$t; 2],
        }

        impl $name {
            /// Create new vector from the given coordinates.
            #[inline(always)]
            pub const fn new(x: $t, y: $t) -> Self {
                Self { elements: [x, y] }
            }

            /// Create new vector with all components set to the given value.
            #[inline(always)]
            pub const fn broadcast(v: $t) -> Self {
                Self::new(v, v)
            }

            /// Create new vector from the given array.
            #[inline(always)]
            pub const fn from_array(components: [$t; 2]) -> Self {
                Self {
                    elements: components,
                }
            }

            /// Create new vector from the given polar representation.
            #[inline(always)]
            pub fn from_polar(radius: $t, angle: $t) -> Self {
                let (s, c) = angle.sin_cos();
                Self::new(radius * c, radius * s)
            }

            /// Access x components by value.
            #[inline(always)]
            pub const fn x(self) -> $t {
                self.elements[Axis2::X as usize]
            }

            /// Access x components by value.
            #[inline(always)]
            pub const fn y(self) -> $t {
                self.elements[Axis2::Y as usize]
            }

            /// Access x components as mutable reference.
            #[inline(always)]
            pub fn x_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis2::X as usize]
            }

            /// Access y components as mutable reference.
            #[inline(always)]
            pub fn y_mut(&mut self) -> &mut $t {
                &mut self.elements[Axis2::Y as usize]
            }

            /// Number of elements in the vector.
            #[inline(always)]
            pub const fn len(self) -> usize {
                2
            }

            /// Access a component by value using an index.
            /// # Safety
            /// The index should either be 0 or 1, otherwise we get undefined behaviour
            #[inline(always)]
            pub unsafe fn get_unchecked(self, i: usize) -> $t {
                debug_assert!(i < self.len());
                *self.elements.get_unchecked(i)
            }

            /// Access a component by index and returns a mutable reference to it.
            /// # Safety
            /// The index should either be 0 or 1, otherwise we get undefined behaviour
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
                Self::new(self.x().min(rhs.x()), self.y().min(rhs.y()))
            }

            /// Compute maximum value for each component.
            #[inline(always)]
            pub fn element_wise_max(self, rhs: Self) -> Self {
                Self::new(self.x().max(rhs.x()), self.y().max(rhs.y()))
            }

            /// Compute product for each component.
            #[inline(always)]
            pub fn element_wise_product(self, rhs: Self) -> Self {
                Self::new(self.x() * rhs.x(), self.y() * rhs.y())
            }

            /// Compute quotient for each component.
            #[inline(always)]
            pub fn element_wise_quotient(self, rhs: Self) -> Self {
                Self::new(self.x() / rhs.x(), self.y() / rhs.y())
            }

            /// Compute absolute value for each component.
            #[inline(always)]
            pub fn element_wise_abs(self) -> Self {
                Self::new(self.x().abs(), self.y().abs())
            }

            /// Compute reciprocal value for each component.
            #[inline(always)]
            pub fn element_wise_recip(self) -> Self {
                Self::new(self.x().recip(), self.y().recip())
            }

            /// Clamp each value between the given min and max.
            #[inline(always)]
            pub fn element_wise_clamp(self, min: $t, max: $t) -> Self {
                Self::new(self.x().clamp(min, max), self.y().clamp(min, max))
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
            }

            /// Normalise vector in place using multiplication.
            #[inline(always)]
            pub fn normalise_fast(&mut self) {
                let inv_l = 1.0 / self.length();
                *self.x_mut() *= inv_l;
                *self.y_mut() *= inv_l;
            }

            /// Linearly interpolate for each component.
            #[inline(always)]
            pub fn lerp(self, t: $t, end: Self) -> Self {
                (1.0 - t) * self + t * end
            }

            /// Compute dot product.
            #[inline(always)]
            pub fn dot(self, rhs: Self) -> $t {
                self.x() * rhs.x() + self.y() * rhs.y()
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
                self.x().min(self.y())
            }

            /// Compute maximum element.
            #[inline(always)]
            pub fn max_element(self) -> $t {
                self.x().max(self.y())
            }

            /// Permute components for the given new axes.
            #[inline(always)]
            pub fn permute(self, x_axis: Axis2, y_axis: Axis2) -> Self {
                Self::new(self[x_axis], self[y_axis])
            }

            /// Permute components for the given new axes as array.
            #[inline(always)]
            pub fn permute_with_array(self, axes: [Axis2; 2]) -> Self {
                self.permute(axes[0], axes[1])
            }

            /// Compute largest axis of the vector.
            #[inline(always)]
            pub fn largest_axis(self) -> Axis2 {
                if self.x() >= self.y() {
                    Axis2::X
                } else {
                    Axis2::Y
                }
            }

            /// Rotate vector around origin for the given angle in radians.
            #[inline(always)]
            pub fn rotate(self, angle_rad: $t) -> Self {
                let (s, c) = angle_rad.sin_cos();
                Self::new(self.x() * c - self.y() * s, self.x() * s + self.y() * c)
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self::new(self.x() + rhs.x(), self.y() + rhs.y())
            }
        }

        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self.x_mut() += rhs.x();
                *self.y_mut() += rhs.y();
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self::new(self.x() - rhs.x(), self.y() - rhs.y())
            }
        }

        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self.x_mut() -= rhs.x();
                *self.y_mut() -= rhs.y();
            }
        }

        impl Mul<$name> for $t {
            type Output = $name;

            #[inline(always)]
            fn mul(self, rhs: $name) -> Self::Output {
                Self::Output::new(self * rhs.x(), self * rhs.y())
            }
        }

        impl Mul<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x() * rhs, self.y() * rhs)
            }
        }

        impl MulAssign<$t> for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: $t) {
                *self.x_mut() *= rhs;
                *self.y_mut() *= rhs;
            }
        }

        impl Neg for $name {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                Self::Output::new(-self.x(), -self.y())
            }
        }

        impl Div<$t> for $name {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: $t) -> Self::Output {
                Self::Output::new(self.x() / rhs, self.y() / rhs)
            }
        }

        impl DivAssign<$t> for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: $t) {
                *self.x_mut() /= rhs;
                *self.y_mut() /= rhs;
            }
        }

        impl Deref for $name {
            type Target = [$t];

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.elements
            }
        }

        impl Index<Axis2> for $name {
            type Output = $t;

            #[inline(always)]
            fn index(&self, index: Axis2) -> &Self::Output {
                &self.elements[index as usize]
            }
        }

        impl IndexMut<Axis2> for $name {
            #[inline(always)]
            fn index_mut(&mut self, index: Axis2) -> &mut Self::Output {
                &mut self.elements[index as usize]
            }
        }
    };
}

generate_vec2!(Vec2f32, f32);
generate_vec2!(Vec2f64, f64);
