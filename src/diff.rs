use std::ops::{Add, Mul};

use num_traits::{float::Float, Zero};

trait VectorSpace: Sized + Add<Self, Output = Self> {
    fn zero() -> Self;
}

trait Space {
    type Base;
    type Tangent: VectorSpace;

    fn r#const(_: Self::Base) -> Self;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Dual<T: Space>(T, T::Tangent);

impl<F> VectorSpace for F
where
    F: Float,
{
    fn zero() -> F {
        <F as Zero>::zero()
    }
}

impl<F> Space for F
where
    F: Float,
{
    type Base = F;
    type Tangent = F;

    fn r#const(x: F) -> F {
        x
    }
}

impl<F, T> Add<Dual<T>> for Dual<T>
where
    F: Float,
    T: Space<Base = F> + Add<T, Output = T>,
{
    type Output = Dual<T>;

    fn add(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0 + other.0, self.1 + other.1)
    }
}

impl<F, T> VectorSpace for Dual<T>
where
    F: Float,
    T: Space<Base = F> + VectorSpace,
{
    fn zero() -> Dual<T> {
        Dual(T::zero(), T::Tangent::zero())
    }
}

impl<F, T> Space for Dual<T>
where
    F: Float,
    T: Space<Base = F>,
    Dual<T>: VectorSpace,
{
    type Base = F;
    type Tangent = Dual<T>;

    fn r#const(x: F) -> Self {
        Dual(T::r#const(x), T::Tangent::zero())
    }
}

impl<F, T> Mul<Dual<T>> for Dual<T>
where
    T: Space<Base = F> + Mul<T, Output = T> + Mul<T::Tangent, Output = T::Tangent> + Copy,
{
    type Output = Dual<T>;

    fn mul(self, other: Dual<T>) -> Dual<T> {
        Dual(self.0 * other.0, self.0 * other.1 + other.0 * self.1)
    }
}

trait FloatDeriv<F>: Space<Base = F> + Add<Self, Output = Self> + Mul<Self, Output = Self> + Copy
where
    F: Float,
{
}

impl<F> FloatDeriv<F> for F where F: Float {}

impl<F, T> FloatDeriv<F> for Dual<T>
where
    F: Float,
    T: FloatDeriv<F> + Space<Base = F> + Mul<T, Output = T> + Mul<T::Tangent, Output = T::Tangent> + Copy,
    Dual<T>: Space<Base = F>,
    T::Tangent: Copy,
{
}

#[test]
fn test_1() {
    fn f<T: FloatDeriv<f64>>(x: T) -> T {
        T::r#const(3.0) * x + x * x
    }
    assert_eq!(f(7.0), 70.0);
    assert_eq!(f(Dual(7.0, 1.0)), Dual(70.0, 17.0));
    assert_eq!(f(Dual(Dual(7.0, 1.0), Dual(1.0, 0.0))), Dual(Dual(70.0, 17.0), Dual(17.0, 2.0)));
}
