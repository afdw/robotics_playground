use approx::{relative_eq, AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, DimName, Field, Matrix, RealField, Scalar, SimdValue};
use num_traits::{Float, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
use simba::scalar::SubsetOf;
use std::{
    array,
    cmp::Ordering,
    fmt::Display,
    num::FpCategory,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

pub trait FloatScalar: Float {}

impl FloatScalar for f32 {}

impl FloatScalar for f64 {}

pub trait Deriv {
    type Base;

    fn r#const(_: Self::Base) -> Self;

    fn base(self) -> Self::Base;
}

#[derive(Clone, Copy, Debug)]
pub struct Dual<T>(pub T, pub T);

impl<T> PartialEq for Dual<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Dual<T> where T: Eq {}

impl<T> PartialOrd for Dual<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Ord for Dual<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> Display for Dual<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) + ({}) ε", self.0, self.1)
    }
}

impl<F> Deriv for F
where
    F: FloatScalar,
{
    type Base = F;

    fn r#const(x: F) -> F {
        x
    }

    fn base(self) -> F {
        self
    }
}

impl<F, T> Deriv for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F>,
{
    type Base = T::Base;

    fn r#const(x: F) -> Dual<T> {
        Dual(T::r#const(x), T::r#const(F::zero()))
    }

    fn base(self) -> F {
        self.0.base()
    }
}

impl<F, T> SubsetOf<Dual<T>> for f32
where
    F: FloatScalar,
    T: Deriv<Base = F> + Clone,
{
    fn to_superset(&self) -> Dual<T> {
        Dual::r#const(F::from(*self).unwrap())
    }

    fn from_superset_unchecked(element: &Dual<T>) -> f32 {
        element.clone().base().to_f32().unwrap()
    }

    fn is_in_subset(_element: &Dual<T>) -> bool {
        unimplemented!()
    }
}

impl<F, T> SubsetOf<Dual<T>> for f64
where
    F: FloatScalar,
    T: Deriv<Base = F> + Clone,
{
    fn to_superset(&self) -> Dual<T> {
        Dual::r#const(F::from(*self).unwrap())
    }

    fn from_superset_unchecked(element: &Dual<T>) -> f64 {
        element.clone().base().to_f64().unwrap()
    }

    fn is_in_subset(_element: &Dual<T>) -> bool {
        unimplemented!()
    }
}

impl<T> SubsetOf<Dual<T>> for Dual<T>
where
    T: Clone,
{
    fn to_superset(&self) -> Dual<T> {
        self.clone()
    }

    fn from_superset_unchecked(element: &Dual<T>) -> Dual<T> {
        element.clone()
    }

    fn is_in_subset(_element: &Dual<T>) -> bool {
        true
    }
}

impl<T> SimdValue for Dual<T>
where
    T: Clone,
{
    type Element = Dual<T>;
    type SimdBool = bool;

    fn lanes() -> usize {
        1
    }

    fn splat(val: Self) -> Self {
        val
    }

    fn extract(&self, _i: usize) -> Self {
        self.clone()
    }

    unsafe fn extract_unchecked(&self, _i: usize) -> Self {
        self.clone()
    }

    fn replace(&mut self, _i: usize, val: Self) {
        *self = val;
    }

    unsafe fn replace_unchecked(&mut self, _i: usize, val: Self::Element) {
        *self = val;
    }

    fn select(self, cond: bool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl<F, T> Zero for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Zero,
{
    fn zero() -> Dual<T> {
        Dual(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<F, T> One for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Zero + One + Add<T, Output = T> + Clone,
{
    fn one() -> Self {
        Dual(T::one(), T::zero())
    }
}

impl<F, T> Add for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Add<T, Output = T>,
{
    type Output = Dual<T>;

    fn add(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0 + other.0, self.1 + other.1)
    }
}

impl<F, T> AddAssign for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Clone,
    Dual<T>: Add<Dual<T>, Output = Dual<T>>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F, T> Sub for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Sub<T, Output = T>,
{
    type Output = Dual<T>;

    fn sub(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0 - other.0, self.1 - other.1)
    }
}

impl<F, T> SubAssign for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Clone,
    Dual<T>: Sub<Dual<T>, Output = Dual<T>>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F, T> Mul for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Add<T, Output = T> + Mul<T, Output = T> + Clone,
{
    type Output = Dual<T>;

    fn mul(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0.clone() * other.0.clone(), self.0 * other.1 + other.0 * self.1)
    }
}

impl<F, T> MulAssign for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Clone,
    Dual<T>: Mul<Dual<T>, Output = Dual<T>>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, T> Div for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T> + Clone,
{
    type Output = Dual<T>;

    fn div(self, other: Dual<T>) -> Dual<T> {
        Dual(
            self.0.clone() / other.0.clone(),
            (self.0 * other.1 - other.0.clone() * self.1) / (other.0.clone() * other.0),
        )
    }
}

impl<F, T> DivAssign for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Clone,
    Dual<T>: Div<Dual<T>, Output = Dual<T>>,
{
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl<F, T> Rem for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Rem<T, Output = T> + Clone,
{
    type Output = Dual<T>;

    fn rem(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0 % other.0, self.1)
    }
}

impl<F, T> RemAssign for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Clone,
    Dual<T>: Rem<Dual<T>, Output = Dual<T>>,
{
    fn rem_assign(&mut self, rhs: Self) {
        *self = self.clone() % rhs;
    }
}

impl<F, T> Neg for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Neg<Output = T> + Clone,
{
    type Output = Dual<T>;

    fn neg(self) -> Dual<T> {
        Dual(-self.0, -self.1)
    }
}

impl<F, T> Field for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Field + Clone,
{
}

impl<F, T> Num for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Num + Clone,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str, radix).map(|x| Dual(x, T::zero()))
    }
}

impl<F, T> ToPrimitive for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + ToPrimitive,
{
    fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.0.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.0.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_i128(&self) -> Option<i128> {
        self.0.to_i128()
    }

    fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.0.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.0.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_u128(&self) -> Option<u128> {
        self.0.to_u128()
    }

    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

impl<F, T> FromPrimitive for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + FromPrimitive + Zero,
{
    fn from_isize(n: isize) -> Option<Dual<T>> {
        T::from_isize(n).map(|x| Dual(x, T::zero()))
    }

    fn from_i8(n: i8) -> Option<Dual<T>> {
        T::from_i8(n).map(|x| Dual(x, T::zero()))
    }

    fn from_i16(n: i16) -> Option<Dual<T>> {
        T::from_i16(n).map(|x| Dual(x, T::zero()))
    }

    fn from_i32(n: i32) -> Option<Dual<T>> {
        T::from_i32(n).map(|x| Dual(x, T::zero()))
    }

    fn from_i64(n: i64) -> Option<Dual<T>> {
        T::from_i64(n).map(|x| Dual(x, T::zero()))
    }

    fn from_i128(n: i128) -> Option<Dual<T>> {
        T::from_i128(n).map(|x| Dual(x, T::zero()))
    }

    fn from_usize(n: usize) -> Option<Dual<T>> {
        T::from_usize(n).map(|x| Dual(x, T::zero()))
    }

    fn from_u8(n: u8) -> Option<Dual<T>> {
        T::from_u8(n).map(|x| Dual(x, T::zero()))
    }

    fn from_u16(n: u16) -> Option<Dual<T>> {
        T::from_u16(n).map(|x| Dual(x, T::zero()))
    }

    fn from_u32(n: u32) -> Option<Dual<T>> {
        T::from_u32(n).map(|x| Dual(x, T::zero()))
    }

    fn from_u64(n: u64) -> Option<Dual<T>> {
        T::from_u64(n).map(|x| Dual(x, T::zero()))
    }

    fn from_u128(n: u128) -> Option<Dual<T>> {
        T::from_u128(n).map(|x| Dual(x, T::zero()))
    }

    fn from_f32(n: f32) -> Option<Dual<T>> {
        T::from_f32(n).map(|x| Dual(x, T::zero()))
    }

    fn from_f64(n: f64) -> Option<Dual<T>> {
        T::from_f64(n).map(|x| Dual(x, T::zero()))
    }
}

impl<F, T> NumCast for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + NumCast,
{
    fn from<S: ToPrimitive>(n: S) -> Option<Self> {
        T::from(n).map(|x| Dual(x, T::r#const(F::zero())))
    }
}

impl<F, T> Float for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + Float,
{
    fn nan() -> Self {
        Self::r#const(F::nan())
    }

    fn infinity() -> Self {
        Self::r#const(F::infinity())
    }

    fn neg_infinity() -> Self {
        Self::r#const(F::neg_infinity())
    }

    fn neg_zero() -> Self {
        Self::r#const(F::neg_zero())
    }

    fn min_value() -> Self {
        Self::r#const(F::min_value())
    }

    fn min_positive_value() -> Self {
        Self::r#const(F::min_positive_value())
    }

    fn max_value() -> Self {
        Self::r#const(F::max_value())
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> FpCategory {
        self.0.classify()
    }

    fn floor(self) -> Dual<T> {
        Dual(self.0.floor(), T::zero())
    }

    fn ceil(self) -> Dual<T> {
        Dual(self.0.ceil(), T::zero())
    }

    fn round(self) -> Dual<T> {
        Dual(self.0.round(), T::zero())
    }

    fn trunc(self) -> Dual<T> {
        Dual(self.0.trunc(), T::zero())
    }

    fn fract(self) -> Dual<T> {
        Dual(self.0.fract(), self.1)
    }

    fn abs(self) -> Dual<T> {
        Dual(self.0.abs(), if self.0.is_sign_positive() { self.1 } else { -self.1 })
    }

    fn signum(self) -> Dual<T> {
        Dual(self.0.signum(), T::zero())
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    fn recip(self) -> Self {
        Self::one() / self
    }

    fn powi(self, n: i32) -> Dual<T> {
        Dual(self.0.powi(n), T::r#const(F::from(n).unwrap()) * self.0.powi(n - 1) * self.1)
    }

    fn powf(self, n: Dual<T>) -> Dual<T> {
        Dual(
            self.0.powf(n.0),
            n.0 * self.0.powf(n.0 - T::one()) * self.1 + self.0.powf(n.0) * self.0.ln() * n.1,
        )
    }

    fn sqrt(self) -> Dual<T> {
        Dual(self.0.sqrt(), self.1 / (T::r#const(F::from(2.0).unwrap()) * self.0.sqrt()))
    }

    fn exp(self) -> Dual<T> {
        Dual(self.0.exp(), self.0.exp() * self.1)
    }

    fn exp2(self) -> Dual<T> {
        Dual(self.0.exp2(), T::r#const(F::from(2.0).unwrap().ln()) * self.0.exp2() * self.1)
    }

    fn ln(self) -> Dual<T> {
        Dual(self.0.ln(), self.1 / self.0)
    }

    fn log(self, base: Dual<T>) -> Dual<T> {
        Dual(
            self.0.log(base.0),
            self.1 / (base.0.ln() * self.0) - (self.0.ln() * base.1) / (base.0.ln() * base.0.ln() * base.0),
        )
    }

    fn log2(self) -> Dual<T> {
        Dual(self.0.log2(), self.1 / (T::r#const(F::from(2.0).unwrap().ln()) * self.0))
    }

    fn log10(self) -> Dual<T> {
        Dual(self.0.log2(), self.1 / (T::r#const(F::from(10.0).unwrap().ln()) * self.0))
    }

    fn max(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0.max(self.1), if self.0 <= other.0 { other.1 } else { self.1 })
    }

    fn min(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0.min(self.1), if self.0 <= other.0 { self.1 } else { other.1 })
    }

    fn abs_sub(self, other: Self) -> Self {
        if self <= other {
            Self::zero()
        } else {
            self - other
        }
    }

    fn cbrt(self) -> Dual<T> {
        Dual(self.0.cbrt(), self.1 / (T::r#const(F::from(3.0).unwrap()) * self.0.cbrt() * self.0.cbrt()))
    }

    fn hypot(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0.hypot(other.0), (self.0 * self.1 + other.0 * other.1) / self.0.hypot(other.0))
    }

    fn sin(self) -> Dual<T> {
        Dual(self.0.sin(), self.0.cos() * self.1)
    }

    fn cos(self) -> Dual<T> {
        Dual(self.0.cos(), -self.0.sin() * self.1)
    }

    fn tan(self) -> Dual<T> {
        Dual(self.0.tan(), self.1 / (self.0.cos() * self.0.cos()))
    }

    fn asin(self) -> Dual<T> {
        Dual(self.0.asin(), self.1 / (T::one() - self.0 * self.0).sqrt())
    }

    fn acos(self) -> Dual<T> {
        Dual(self.0.acos(), -self.1 / (T::one() - self.0 * self.0).sqrt())
    }

    fn atan(self) -> Dual<T> {
        Dual(self.0.atan(), self.1 / (T::one() + self.0 * self.0))
    }

    fn atan2(self, other: Dual<T>) -> Dual<T> {
        Dual(
            self.0.atan2(other.0),
            (self.0 * other.1 - other.0 * self.1) / (self.0 * self.0 + other.0 * other.0),
        )
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn exp_m1(self) -> Dual<T> {
        Dual(self.0.exp_m1(), self.0.exp() * self.1)
    }

    fn ln_1p(self) -> Dual<T> {
        Dual(self.0.ln_1p(), self.1 / (self.0 + T::one()))
    }

    fn sinh(self) -> Dual<T> {
        Dual(self.0.sinh(), self.0.cosh() * self.1)
    }

    fn cosh(self) -> Dual<T> {
        Dual(self.0.cosh(), self.0.sinh() * self.1)
    }

    fn tanh(self) -> Dual<T> {
        Dual(self.0.tanh(), self.1 / (self.0.cosh() * self.0.cosh()))
    }

    fn asinh(self) -> Dual<T> {
        Dual(self.0.asinh(), self.1 / (T::one() + self.0 * self.0).sqrt())
    }

    fn acosh(self) -> Dual<T> {
        Dual(self.0.acosh(), self.1 / (-T::one() + self.0 * self.0).sqrt())
    }

    fn atanh(self) -> Dual<T> {
        Dual(self.0.atanh(), self.1 / (T::one() - self.0 * self.0))
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.integer_decode()
    }
}

impl<F, T> Signed for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F>,
    Dual<T>: Float,
{
    fn abs(&self) -> Self {
        Float::abs(*self)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        Float::abs_sub(*self, *other)
    }

    fn signum(&self) -> Self {
        Float::abs(*self)
    }

    fn is_positive(&self) -> bool {
        *self > Self::zero()
    }

    fn is_negative(&self) -> bool {
        *self < Self::zero()
    }
}

impl<F, T> AbsDiffEq for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + AbsDiffEq<Epsilon = T> + Zero,
{
    type Epsilon = Dual<T>;

    fn default_epsilon() -> Dual<T> {
        Dual(T::default_epsilon(), T::zero())
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Dual<T>) -> bool {
        self.0.abs_diff_eq(&other.0, epsilon.0)
    }
}

impl<F, T> RelativeEq for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + RelativeEq<Epsilon = T> + Zero,
{
    fn default_max_relative() -> Dual<T> {
        Dual(T::default_max_relative(), T::zero())
    }

    fn relative_eq(&self, other: &Self, epsilon: Dual<T>, max_relative: Dual<T>) -> bool {
        self.0.relative_eq(&other.0, epsilon.0, max_relative.0)
    }
}

impl<F, T> UlpsEq for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + UlpsEq<Epsilon = T> + Zero,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Dual<T>, max_ulps: u32) -> bool {
        self.0.ulps_eq(&other.0, epsilon.0, max_ulps)
    }
}

impl<F, T> RealField for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + RealField,
    Dual<T>: Float,
{
    fn is_sign_positive(&self) -> bool {
        Float::is_sign_positive(*self)
    }

    fn is_sign_negative(&self) -> bool {
        Float::is_sign_negative(*self)
    }

    fn copysign(self, sign: Self) -> Self {
        if sign.is_sign_positive() {
            Float::abs(self)
        } else {
            -Float::abs(self)
        }
    }

    fn max(self, other: Self) -> Self {
        Float::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        Float::min(self, other)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    fn atan2(self, other: Self) -> Self {
        Float::atan2(self, other)
    }

    fn min_value() -> Option<Dual<T>> {
        T::min_value().map(|x| Dual(x, T::zero()))
    }

    fn max_value() -> Option<Dual<T>> {
        T::max_value().map(|x| Dual(x, T::zero()))
    }

    fn pi() -> Dual<T> {
        Dual(T::pi(), T::zero())
    }

    fn two_pi() -> Dual<T> {
        Dual(T::two_pi(), T::zero())
    }

    fn frac_pi_2() -> Dual<T> {
        Dual(T::frac_pi_2(), T::zero())
    }

    fn frac_pi_3() -> Dual<T> {
        Dual(T::frac_pi_3(), T::zero())
    }

    fn frac_pi_4() -> Dual<T> {
        Dual(T::frac_pi_4(), T::zero())
    }

    fn frac_pi_6() -> Dual<T> {
        Dual(T::frac_pi_6(), T::zero())
    }

    fn frac_pi_8() -> Dual<T> {
        Dual(T::frac_pi_8(), T::zero())
    }

    fn frac_1_pi() -> Dual<T> {
        Dual(T::frac_1_pi(), T::zero())
    }

    fn frac_2_pi() -> Dual<T> {
        Dual(T::frac_2_pi(), T::zero())
    }

    fn frac_2_sqrt_pi() -> Dual<T> {
        Dual(T::frac_2_sqrt_pi(), T::zero())
    }

    fn e() -> Dual<T> {
        Dual(T::e(), T::zero())
    }

    fn log2_e() -> Dual<T> {
        Dual(T::log2_e(), T::zero())
    }

    fn log10_e() -> Dual<T> {
        Dual(T::log10_e(), T::zero())
    }

    fn ln_2() -> Dual<T> {
        Dual(T::ln_2(), T::zero())
    }

    fn ln_10() -> Dual<T> {
        Dual(T::ln_10(), T::zero())
    }
}

impl<F, T> ComplexField for Dual<T>
where
    F: FloatScalar,
    T: Deriv<Base = F> + ComplexField + RealField,
    Dual<T>: Float,
{
    type RealField = Dual<T>;

    fn from_real(re: Self) -> Self {
        re
    }

    fn real(self) -> Self {
        self
    }

    fn imaginary(self) -> Self {
        Self::zero()
    }

    fn modulus(self) -> Self {
        Float::abs(self)
    }

    fn modulus_squared(self) -> Self {
        self * self
    }

    fn argument(self) -> Self {
        if self >= Self::zero() {
            Self::zero()
        } else {
            Self::pi()
        }
    }

    fn norm1(self) -> Self {
        Float::abs(self)
    }

    fn scale(self, factor: Self) -> Self {
        self * factor
    }

    fn unscale(self, factor: Self) -> Self {
        self / factor
    }

    fn floor(self) -> Self {
        Float::floor(self)
    }

    fn ceil(self) -> Self {
        Float::ceil(self)
    }

    fn round(self) -> Self {
        Float::round(self)
    }

    fn trunc(self) -> Self {
        Float::trunc(self)
    }

    fn fract(self) -> Self {
        Float::fract(self)
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Float::mul_add(self, a, b)
    }

    fn abs(self) -> Self {
        Float::abs(self)
    }

    fn hypot(self, other: Self) -> Self {
        Float::hypot(self, other)
    }

    fn recip(self) -> Self {
        Float::recip(self)
    }

    fn conjugate(self) -> Self {
        self
    }

    fn sin(self) -> Self {
        Float::sin(self)
    }

    fn cos(self) -> Self {
        Float::cos(self)
    }

    fn sin_cos(self) -> (Self, Self) {
        Float::sin_cos(self)
    }

    fn tan(self) -> Self {
        Float::tan(self)
    }

    fn asin(self) -> Self {
        Float::asin(self)
    }

    fn acos(self) -> Self {
        Float::acos(self)
    }

    fn atan(self) -> Self {
        Float::atan(self)
    }

    fn sinh(self) -> Self {
        Float::sinh(self)
    }

    fn cosh(self) -> Self {
        Float::cosh(self)
    }

    fn tanh(self) -> Self {
        Float::tanh(self)
    }

    fn asinh(self) -> Self {
        Float::asinh(self)
    }

    fn acosh(self) -> Self {
        Float::acosh(self)
    }

    fn atanh(self) -> Self {
        Float::atanh(self)
    }

    fn log(self, base: Self) -> Self {
        Float::log(self, base)
    }

    fn log2(self) -> Self {
        Float::log2(self)
    }

    fn log10(self) -> Self {
        Float::log10(self)
    }

    fn ln(self) -> Self {
        Float::ln(self)
    }

    fn ln_1p(self) -> Self {
        Float::ln_1p(self)
    }

    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }

    fn exp(self) -> Self {
        Float::exp(self)
    }

    fn exp2(self) -> Self {
        Float::exp2(self)
    }

    fn exp_m1(self) -> Self {
        Float::exp_m1(self)
    }

    fn powi(self, n: i32) -> Self {
        Float::powi(self, n)
    }

    fn powf(self, n: Self) -> Self {
        Float::powf(self, n)
    }

    fn powc(self, n: Self) -> Self {
        Float::powf(self, n)
    }

    fn cbrt(self) -> Self {
        Float::cbrt(self)
    }

    fn is_finite(&self) -> bool {
        Float::is_finite(*self)
    }

    fn try_sqrt(self) -> Option<Self> {
        if self >= Self::zero() {
            Some(Float::sqrt(self))
        } else {
            None
        }
    }
}

pub trait FloatDeriv: Deriv<Base = Self::Scalar> + Float + RealField + ComplexField<RealField = Self> {
    type Scalar: FloatScalar;
}

impl<F> FloatDeriv for F
where
    F: FloatScalar + RealField,
{
    type Scalar = F;
}

impl<F, T> FloatDeriv for Dual<T>
where
    F: FloatScalar,
    T: FloatDeriv<Scalar = F>,
{
    type Scalar = F;
}

pub trait Transpose {
    type Target;

    fn up_transpose(_: Self::Target) -> Self;

    fn down_transpose(self) -> Self::Target;
}

impl<T> Transpose for (Dual<T>,) {
    type Target = Dual<(T,)>;

    fn up_transpose(x: Self::Target) -> Self {
        (Dual(x.0 .0, x.1 .0),)
    }

    fn down_transpose(self) -> Self::Target {
        Dual((self.0 .0,), (self.0 .1,))
    }
}

impl<T1, T2> Transpose for (Dual<T1>, Dual<T2>) {
    type Target = Dual<(T1, T2)>;

    fn up_transpose(x: Self::Target) -> Self {
        (Dual(x.0 .0, x.1 .0), Dual(x.0 .1, x.1 .1))
    }

    fn down_transpose(self) -> Self::Target {
        Dual((self.0 .0, self.1 .0), (self.0 .1, self.1 .1))
    }
}

impl<T1, T2, T3> Transpose for (Dual<T1>, Dual<T2>, Dual<T3>) {
    type Target = Dual<(T1, T2, T3)>;

    fn up_transpose(x: Self::Target) -> Self {
        (Dual(x.0 .0, x.1 .0), Dual(x.0 .1, x.1 .1), Dual(x.0 .2, x.1 .2))
    }

    fn down_transpose(self) -> Self::Target {
        Dual((self.0 .0, self.1 .0, self.2 .0), (self.0 .1, self.1 .1, self.2 .1))
    }
}

impl<T1, T2, T3, T4> Transpose for (Dual<T1>, Dual<T2>, Dual<T3>, Dual<T4>) {
    type Target = Dual<(T1, T2, T3, T4)>;

    fn up_transpose(x: Self::Target) -> Self {
        (Dual(x.0 .0, x.1 .0), Dual(x.0 .1, x.1 .1), Dual(x.0 .2, x.1 .2), Dual(x.0 .3, x.1 .3))
    }

    fn down_transpose(self) -> Self::Target {
        Dual((self.0 .0, self.1 .0, self.2 .0, self.3 .0), (self.0 .1, self.1 .1, self.2 .1, self.3 .1))
    }
}

impl<T, const N: usize> Transpose for [Dual<T>; N]
where
    T: Clone,
{
    type Target = Dual<[T; N]>;

    fn up_transpose(x: Self::Target) -> Self {
        array::from_fn(|i| Dual(x.0[i].clone(), x.1[i].clone()))
    }

    fn down_transpose(self) -> Self::Target {
        Dual(array::from_fn(|i| self[i].0.clone()), array::from_fn(|i| self[i].1.clone()))
    }
}

impl<T> Transpose for Vec<Dual<T>>
where
    T: Clone,
{
    type Target = Dual<Vec<T>>;

    fn up_transpose(x: Self::Target) -> Self {
        assert_eq!(x.0.len(), x.1.len());
        x.0.iter().cloned().zip(x.1.iter().cloned()).map(|(a, b)| Dual(a, b)).collect()
    }

    fn down_transpose(self) -> Self::Target {
        Dual(self.iter().cloned().map(|x| x.0).collect(), self.iter().cloned().map(|x| x.1).collect())
    }
}

impl<T, R, C> Transpose for Matrix<Dual<T>, R, C, <DefaultAllocator as Allocator<Dual<T>, R, C>>::Buffer>
where
    T: Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<Dual<T>, R, C>,
    DefaultAllocator: Allocator<T, R, C>,
{
    type Target = Dual<Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>>;

    fn up_transpose(x: Self::Target) -> Self {
        assert_eq!(x.0.nrows(), x.1.nrows());
        assert_eq!(x.0.ncols(), x.1.ncols());
        Matrix::<Dual<T>, R, C, _>::from_fn(|i, j| Dual(x.0[(i, j)].clone(), x.1[(i, j)].clone()))
    }

    fn down_transpose(self) -> Self::Target {
        Dual(
            Matrix::<T, R, C, _>::from_fn(|i, j| self[(i, j)].0.clone()),
            Matrix::<T, R, C, _>::from_fn(|i, j| self[(i, j)].1.clone()),
        )
    }
}

pub trait FloatDerivApproxEq<F>
where
    F: FloatScalar,
{
    fn float_deriv_approx_eq(self, other: Self) -> bool;
}

impl<F> FloatDerivApproxEq<F> for F
where
    F: FloatScalar + RelativeEq,
{
    fn float_deriv_approx_eq(self, other: F) -> bool {
        relative_eq!(self, other)
    }
}

impl<F, T> FloatDerivApproxEq<F> for Dual<T>
where
    F: FloatScalar,
    T: FloatDerivApproxEq<F>,
{
    fn float_deriv_approx_eq(self, other: Dual<T>) -> bool {
        self.0.float_deriv_approx_eq(other.0) && self.1.float_deriv_approx_eq(other.1)
    }
}

impl<F, T1, T2> FloatDerivApproxEq<F> for (T1, T2)
where
    F: FloatScalar,
    T1: FloatDerivApproxEq<F>,
    T2: FloatDerivApproxEq<F>,
{
    fn float_deriv_approx_eq(self, other: Self) -> bool {
        self.0.float_deriv_approx_eq(other.0) && self.1.float_deriv_approx_eq(other.1)
    }
}

impl<F, T1, T2, T3> FloatDerivApproxEq<F> for (T1, T2, T3)
where
    F: FloatScalar,
    T1: FloatDerivApproxEq<F>,
    T2: FloatDerivApproxEq<F>,
    T3: FloatDerivApproxEq<F>,
{
    fn float_deriv_approx_eq(self, other: Self) -> bool {
        self.0.float_deriv_approx_eq(other.0) && self.1.float_deriv_approx_eq(other.1) && self.2.float_deriv_approx_eq(other.2)
    }
}

impl<F, T, R, C> FloatDerivApproxEq<F> for Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>
where
    F: FloatScalar,
    T: FloatDerivApproxEq<F> + Scalar,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    fn float_deriv_approx_eq(self, other: Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a.clone().float_deriv_approx_eq(b.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial() {
        fn f<D: FloatDeriv<Scalar = f64>>(x: D) -> D {
            D::r#const(3.0) * x + x * x
        }

        assert!(f(7.0).float_deriv_approx_eq(70.0));
        assert!(f(Dual(7.0, 1.0)).float_deriv_approx_eq(Dual(70.0, 17.0)));
        assert!(f(Dual(Dual(7.0, 1.0), Dual(1.0, 0.0))).float_deriv_approx_eq(Dual(Dual(70.0, 17.0), Dual(17.0, 2.0))));
    }

    #[test]
    fn multiple_variables() {
        fn f<D: FloatDeriv<Scalar = f64>>(x: D, y: D) -> (D, D, D) {
            (x + y, Float::exp(x), Float::sin(y))
        }

        let (exp, sin, cos) = (Float::exp, Float::sin, Float::cos);

        assert!(f(2.0, 3.0).float_deriv_approx_eq((5.0, exp(2.0), Float::sin(3.0))));
        assert!(f(Dual(2.0, 1.0), Dual(3.0, 0.0)).float_deriv_approx_eq((Dual(5.0, 1.0), Dual(exp(2.0), exp(2.0)), Dual(sin(3.0), 0.0))));
        assert!(f(Dual(2.0, 0.0), Dual(3.0, 1.0)).float_deriv_approx_eq((Dual(5.0, 1.0), Dual(exp(2.0), 0.0), Dual(sin(3.0), cos(3.0)))));
    }

    #[test]
    fn matrices() {
        use nalgebra::{Matrix2, Rotation2};

        fn f<D: FloatDeriv<Scalar = f64>>(angle: D) -> Matrix2<D> {
            *Rotation2::new(angle).matrix()
        }

        let (sin, cos) = (Float::sin, Float::cos);

        assert!(f(2.0).float_deriv_approx_eq(Matrix2::new(cos(2.0), -sin(2.0), sin(2.0), cos(2.0))));
        assert!(f(Dual(2.0, 1.0)).down_transpose().float_deriv_approx_eq(Dual(
            Matrix2::new(cos(2.0), -sin(2.0), sin(2.0), cos(2.0)),
            Matrix2::new(-sin(2.0), -cos(2.0), cos(2.0), -sin(2.0))
        )));

        let Dual(m, d_m) = f(Dual(2.0, 1.0)).down_transpose();
        assert!((d_m * m.transpose()).float_deriv_approx_eq(Matrix2::new(0.0, -1.0, 1.0, 0.0)));
    }
}
