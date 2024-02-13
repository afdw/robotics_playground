use crate::diff::FloatDeriv;
use nalgebra::{Matrix3, Vector3};

pub trait Vector3Ext<D> {
    fn cross_mat(self) -> Matrix3<D>;
}

impl<D> Vector3Ext<D> for Vector3<D>
where
    D: FloatDeriv,
{
    fn cross_mat(self) -> Matrix3<D> {
        Matrix3::new(D::zero(), -self.z, self.y, self.z, D::zero(), -self.x, -self.y, self.x, D::zero())
    }
}

pub trait Matrix3Ext<D> {
    fn cross_vec(self) -> Vector3<D>;
}

impl<D> Matrix3Ext<D> for Matrix3<D>
where
    D: FloatDeriv,
{
    fn cross_vec(self) -> Vector3<D> {
        Vector3::new(
            (self.m32 - self.m23) / D::from(2).unwrap(),
            (self.m13 - self.m31) / D::from(2).unwrap(),
            (self.m21 - self.m12) / D::from(2).unwrap(),
        )
    }
}
