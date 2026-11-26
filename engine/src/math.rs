//! The f32 functions core lacks, from libm without `std`. With `std` (and in
//! tests, which link std) the inherent methods are used and this trait is not
//! imported.

#![cfg_attr(any(feature = "std", test), allow(dead_code))]

pub(crate) trait Float: Sized {
    fn exp(self) -> Self;
    fn ln_1p(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn round(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn tanh(self) -> Self;
    fn ln(self) -> Self;
}

impl Float for f32 {
    #[inline]
    fn exp(self) -> f32 {
        libm::expf(self)
    }
    #[inline]
    fn ln_1p(self) -> f32 {
        libm::log1pf(self)
    }
    #[inline]
    fn sqrt(self) -> f32 {
        libm::sqrtf(self)
    }
    #[inline]
    fn powf(self, n: f32) -> f32 {
        libm::powf(self, n)
    }
    #[inline]
    fn round(self) -> f32 {
        libm::roundf(self)
    }
    #[inline]
    fn sin_cos(self) -> (f32, f32) {
        libm::sincosf(self)
    }
    #[inline]
    fn tanh(self) -> f32 {
        libm::tanhf(self)
    }
    #[inline]
    fn ln(self) -> f32 {
        libm::logf(self)
    }
}
