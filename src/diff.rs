use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::{
    cmp::Ordering,
    num::FpCategory,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
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
pub struct Dual<T>(T, T);

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

pub trait FloatDeriv<F>: Deriv<Base = F> + Float
where
    F: FloatScalar,
{
}

impl<F> FloatDeriv<F> for F where F: FloatScalar {}

impl<F, T> FloatDeriv<F> for Dual<T>
where
    F: FloatScalar,
    T: FloatDeriv<F>,
{
}

pub trait FloatDerivApproxEq<F>
where
    F: FloatScalar,
{
    fn float_deriv_approx_eq(self, other: Self) -> bool;
}

impl<F> FloatDerivApproxEq<F> for F
where
    F: FloatScalar,
{
    fn float_deriv_approx_eq(self, other: F) -> bool {
        (self - other).abs() < F::from(0.0001).unwrap()
    }
}

impl<F, T> FloatDerivApproxEq<F> for Dual<T>
where
    F: FloatScalar,
    T: FloatDeriv<F> + FloatDerivApproxEq<F>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial() {
        fn f<T: FloatDeriv<f64>>(x: T) -> T {
            T::r#const(3.0) * x + x * x
        }
        assert!(f(7.0).float_deriv_approx_eq(70.0));
        assert!(f(Dual(7.0, 1.0)).float_deriv_approx_eq(Dual(70.0, 17.0)));
        assert!(f(Dual(Dual(7.0, 1.0), Dual(1.0, 0.0))).float_deriv_approx_eq(Dual(Dual(70.0, 17.0), Dual(17.0, 2.0))));
    }

    #[test]
    fn multiple_variables() {
        fn f<T: FloatDeriv<f64>>(x: T, y: T) -> (T, T, T) {
            (x + y, x.exp(), y.sin())
        }
        assert!(f(2.0, 3.0).float_deriv_approx_eq((5.0, 2.0.exp(), 3.0.sin())));
        assert!(f(Dual(2.0, 1.0), Dual(3.0, 0.0)).float_deriv_approx_eq((Dual(5.0, 1.0), Dual(2.0.exp(), 2.0.exp()), Dual(3.0.sin(), 0.0))));
        assert!(f(Dual(2.0, 0.0), Dual(3.0, 1.0)).float_deriv_approx_eq((Dual(5.0, 1.0), Dual(2.0.exp(), 0.0), Dual(3.0.sin(), 3.0.cos()))));
    }
}
