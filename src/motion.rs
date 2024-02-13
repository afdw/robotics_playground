#![cfg_attr(doc, katexit::katexit)]

use crate::{
    diff::FloatDeriv,
    linalg::{Matrix3Ext, Vector3Ext},
};
use nalgebra::{Matrix3, Vector3};
use num_traits::{Inv, Zero};
use std::ops::{Add, Mul, Neg};

#[cfg_attr(doc, katexit::katexit)]
/// Coordinate change from $B$ to $A$, called $\mathbf{T}$, or ${}^A\mathbf{T}_B$ in full.
///
/// For a vector $\mathbf{v}$:
/// $$
/// \begin{gathered}
///     \mathbf{T} \mathbf{v} = \mathbf{R} \mathbf{v} + \mathbf{t} \text{, and} \\\\
///     \begin{bmatrix}
///         \\\\
///         \mathbf{T} \mathbf{v} \\\\
///         \\\\
///         1
///     \end{bmatrix} = \begin{bmatrix}
///         1 & 0 & 0 & \\\\
///         0 & 1 & 0 & \mathbf{t} \\\\
///         0 & 0 & 1 & \\\\
///         0 & 0 & 0 & 1
///     \end{bmatrix} \begin{bmatrix}
///         & & & 0 \\\\
///         & \mathbf{R} & & 0 \\\\
///         & & & 0 \\\\
///         0 & 0 & 0 & 1
///     \end{bmatrix} \begin{bmatrix}
///         \\\\
///         \mathbf{v} \\\\
///         \\\\
///         1
///     \end{bmatrix} = \begin{bmatrix}
///         & & & \\\\
///         & \mathbf{R} & & \mathbf{t} \\\\
///         & & & \\\\
///         0 & 0 & 0 & 1
///     \end{bmatrix} \begin{bmatrix}
///         \\\\
///         \mathbf{v} \\\\
///         \\\\
///         1
///     \end{bmatrix} \text{.}
/// \end{gathered}
/// $$
///
/// This has two interpretations:
/// 1. Given the coordinates of a point in $B$, produce its coordinates in $A$.
/// 2. Given the coordinates of a point attached to $C$ in $A$, produce its coordinates still in $A$, but after $C$ moved from $A$ to $B$ (and so the point
///    moved with it).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Motion<D> {
    /// Rotation component of the change from $B$ to $A$, called $\mathbf{R}$.
    ///
    /// This is normally an orthogonal matrix.
    pub rotation: Matrix3<D>,
    /// Translation component of the change from $B$ to $A$, called $\mathbf{t}$.
    ///
    /// Expressed in $A$ coordinates.
    pub translation: Vector3<D>,
}

impl<D> Motion<D>
where
    D: FloatDeriv,
{
    pub fn identity() -> Motion<D> {
        Motion {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }
}

impl<D> Mul<Motion<D>> for Motion<D>
where
    D: FloatDeriv,
{
    type Output = Motion<D>;

    fn mul(self, other: Motion<D>) -> Motion<D> {
        Motion {
            rotation: self.rotation * other.rotation,
            translation: self.translation + self.rotation * other.translation,
        }
    }
}

impl<D> Inv for Motion<D>
where
    D: FloatDeriv,
{
    type Output = Motion<D>;

    fn inv(self) -> Motion<D> {
        Motion {
            rotation: self.rotation.transpose(),
            translation: -self.rotation.transpose() * self.translation,
        }
    }
}

impl<D> Mul<Vector3<D>> for Motion<D>
where
    D: FloatDeriv,
{
    type Output = Vector3<D>;

    fn mul(self, other: Vector3<D>) -> Vector3<D> {
        self.rotation * other + self.translation
    }
}

/// An element of the tangent space to [`Motion`] at [`Motion::identity`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Twist<D> {
    /// Derivative of [`Motion::rotation`], after applying [`cross_vec`](Matrix3Ext::cross_vec) to it.
    ///
    /// This is possible, as the derivative of the rotation part of the motion is always antisymmetric at identity.
    pub angular: Vector3<D>,
    /// Derivative of [`Motion::translation`].
    pub linear: Vector3<D>,
}

impl<D> Zero for Twist<D>
where
    D: FloatDeriv,
{
    fn zero() -> Self {
        Twist {
            angular: Vector3::zero(),
            linear: Vector3::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.angular.is_zero() && self.linear.is_zero()
    }
}

impl<D> Add<Twist<D>> for Twist<D>
where
    D: FloatDeriv,
{
    type Output = Twist<D>;

    fn add(self, other: Twist<D>) -> Twist<D> {
        Twist {
            angular: self.angular + other.angular,
            linear: self.linear + other.linear,
        }
    }
}

impl<D> Mul<D> for Twist<D>
where
    D: FloatDeriv,
{
    type Output = Twist<D>;

    fn mul(self, other: D) -> Twist<D> {
        Twist {
            angular: self.angular * other,
            linear: self.linear * other,
        }
    }
}

impl<D> Neg for Twist<D>
where
    D: FloatDeriv,
{
    type Output = Twist<D>;

    fn neg(self) -> Twist<D> {
        Twist {
            angular: -self.angular,
            linear: self.linear,
        }
    }
}

impl<D> Mul<Vector3<D>> for Twist<D>
where
    D: FloatDeriv,
{
    type Output = Vector3<D>;

    fn mul(self, other: Vector3<D>) -> Vector3<D> {
        self.angular.cross_mat() * other + self.linear
    }
}

impl<D> Mul<Twist<D>> for Motion<D>
where
    D: FloatDeriv,
{
    type Output = Twist<D>;

    fn mul(self, other: Twist<D>) -> Twist<D> {
        Twist {
            angular: self.rotation * other.angular,
            linear: self.rotation * (self.rotation.transpose() * self.translation).cross_mat() * other.angular + self.rotation * other.linear,
        }
    }
}

impl<D> Twist<D>
where
    D: FloatDeriv,
{
    pub fn cross(self, other: Twist<D>) -> Twist<D> {
        Twist { angular: self.angular.cross(&other.angular), linear: self.angular.cross(&other.linear) + self.linear.cross(&other.angular) }
    }
}
