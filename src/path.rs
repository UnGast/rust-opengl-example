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
        let prev_seg = &self.segments[index - 1];
        let curr_seg = &self.segments[index];
        let next_seg = &self.segments[index + 1];

        let edge1: Vector3<f32> = curr_seg.position - prev_seg.position;
        let edge2: Vector3<f32> = next_seg.position - curr_seg.position;

        let ang_bisec_vec: Vector3<f32> = find_angle_bisecting_vec(&edge1, &edge2);//.normalize() + edge2.normalize();
        let cross_vec: Vector3<f32> = edge1.cross(&ang_bisec_vec).normalize();
        println!("ALPHA BISC {}", ang_bisec_vec);
        println!("CROSS {}", cross_vec);

        let loop_vert_count = 10;
        let mut vertices = Vec::<Vertex>::new();

        for i in 0..loop_vert_count {
            let angle = std::f32::consts::PI * 2.0 * (i as f32 / loop_vert_count as f32);
            let new_pos: Vector3<f32> = curr_seg.position + angle.cos() * ang_bisec_vec + angle.sin() * cross_vec;
            println!("angle {} {} {}", angle, angle.cos(), angle.sin());
            let new_vert = Vertex { position: new_pos };
            println!("new_vert {}", new_pos);
            vertices.push(new_vert);
        }

        Vec::from(vertices)
    }

    pub fn makeLoopMesh(&self) -> Mesh {
        let mut vertices = Vec::<Vertex>::new();
        let mut triangles = Vec::<Triangle>::new();
        for i in 1..self.segments.len() - 1 {
            let mut new_verts = self.makeLoopForSegment(i);
            if i > 1 {
                for ver_i in 0..new_verts.len() {
                    triangles.append(&mut Vec::from([
                        Triangle::new(
                            vertices.len() + ver_i, 
                            vertices.len() + ver_i - new_verts.len(),
                            vertices.len() + ver_i - new_verts.len() + 1),
                        Triangle::new(
                            vertices.len() + ver_i,
                            vertices.len() + ver_i - new_verts.len() + 1,
                            vertices.len() + (ver_i + 1) % new_verts.len()
                        )
                    ]))
                }
            }
            vertices.append(&mut new_verts);
        }
        println!("MADE VERTS {}", vertices.len());
        Mesh { vertices, triangles }
    }
}

fn find_angle_bisecting_vec(vector1: &Vector3<f32>, vector2: &Vector3<f32>) -> Vector3<f32> {
    let mut bisec: Vector3<f32> = vector1.cross(vector2);
    println!("GOT BISEC {} {}", bisec, bisec.magnitude_squared());
    if bisec.magnitude_squared() == 0.0 {
        if vector1.z != 0.0 {
            let x = 1f32;
            let y = 1f32;
            let z = (-vector1.x - vector1.y) / vector1.z;
            bisec = Vector3::new(x, y, z);
        } else if vector1.y != 0.0 {
            let x = 1f32;
            let z = 1f32;
            let y = (-vector1.x - vector1.z) / vector1.y;
            bisec = Vector3::new(x, y, z);
        } else if vector1.x != 0.0 {
            let z = 1f32;
            let y = 1f32;
            let x = (-vector1.y - vector1.z) / vector1.x;
            bisec = Vector3::new(x, y, z);
        } else {
            bisec = Vector3::new(1.0, 0.0, 0.0);
        }
    }
    bisec
}

#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: Vector3<f32>
}

impl Vertex {
    pub fn new(position: Vector3<f32>) -> Self {
        Vertex { position }
    }
}

pub struct Triangle {
    pub v1: usize,
    pub v2: usize,
    pub v3: usize
}

impl Triangle {
    pub fn new(v1: usize, v2: usize, v3: usize) -> Self {
        Triangle { v1, v2, v3 }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triangle>
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, triangles: Vec<Triangle>) -> Self {
        Mesh { vertices, triangles }
    }

    pub fn unfold_vertices(&self) -> Vec<Vertex> {
        let mut unfolded = Vec::<Vertex>::new();

        for triangle in self.triangles.iter() {
            unfolded.append(&mut Vec::from([
                self.vertices[triangle.v1].clone(),
                self.vertices[triangle.v2].clone(),
                self.vertices[triangle.v3].clone(),
            ]));
        }

        unfolded
    }
}