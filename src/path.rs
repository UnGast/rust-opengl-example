use std::borrow::Borrow;

use ::nalgebra::base::{Vector3};

pub struct PathSegment {
   position: Vector3<f32>
}

impl PathSegment {
    pub fn new(position: Vector3<f32>) -> Self {
        PathSegment { position }
    }
}

pub struct Path {
    segments: Vec<PathSegment>
}

impl Path {
    pub fn new(segments: Vec<PathSegment>) -> Self {
        Path { segments }
    }

    pub fn makeLoopForSegment(&self, index: usize) -> Vec<Vertex> {
        let prev = &self.segments[index - 1];
        let next = &self.segments[index + 1];
        let alpha_bisec_vec: Vector3<f32> = prev.position.normalize() + next.position.normalize();
        println!("vec {}", alpha_bisec_vec);
        Vec::from([Vertex { position: alpha_bisec_vec }])
    }

    pub fn makeLoopMesh(&self) -> Mesh {
        let mut vertices = Vec::<Vertex>::new();
        for i in 1..self.segments.len() - 1 {
            vertices.append(&mut self.makeLoopForSegment(i));
        }
        Mesh { vertices }
    }
}

pub struct Vertex {
    pub position: Vector3<f32>
}

impl Vertex {
    pub fn new(position: Vector3<f32>) -> Self {
        Vertex { position }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>) -> Self {
        Mesh { vertices }
    }
}