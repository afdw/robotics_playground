use crate::motion::Motion;
use nalgebra::Vector3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShapeType {
    Box,
    Cylinder,
    Ellipsoid,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Shape<D> {
    pub r#type: ShapeType,
    pub dimensions: Vector3<D>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Body<D> {
    pub shape: Shape<D>,
    pub position: Motion<D>,
}
