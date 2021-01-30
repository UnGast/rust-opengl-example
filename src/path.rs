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

    pub fn connect_points_straight(points: Vec<Vector3<f32>>) -> Self {
        let segments = points.iter().map(|p| PathSegment::new(*p)).collect();
        Path { segments }
    }

    fn makeLoopForSegment(&self, index: usize, vertices_per_loop: usize) -> Vec<Vertex> {
        let curr_seg = &self.segments[index];

        let (right_vec1, right_vec2) = {
            if index > 0 && index < self.segments.len() - 1 {
                let prev_seg = &self.segments[index - 1];
                let next_seg = &self.segments[index + 1];

                let edge1: Vector3<f32> = curr_seg.position - prev_seg.position;
                let edge2: Vector3<f32> = next_seg.position - curr_seg.position;

                let ang_bisec_vec: Vector3<f32> = find_angle_bisecting_vec(&edge1, &edge2).normalize();//.normalize() + edge2.normalize();
                let cross_vec: Vector3<f32> = edge1.cross(&ang_bisec_vec).normalize();

                (ang_bisec_vec as Vector3<f32>, cross_vec as Vector3<f32>)
            } else if index == 0 {
                let next_seg = &self.segments[index + 1];
                let edge: Vector3<f32> = next_seg.position - curr_seg.position;
                let right_vec1 = get_any_vector_normal_to(&edge).normalize();
                let right_vec2 = edge.cross(&right_vec1).normalize();

                (right_vec1, right_vec2)
            } else if index == self.segments.len() - 1 {
                let prev_seg = &self.segments[index - 1];
                let edge: Vector3<f32> = curr_seg.position - prev_seg.position;
                let right_vec1 = get_any_vector_normal_to(&edge).normalize();
                let right_vec2 = edge.cross(&right_vec1).normalize();

                (right_vec1, right_vec2)
            } else {
                panic!("can't make a loop for index {}", index);
            }
        };
        //println!("ALPHA BISC {}", ang_bisec_vec);
        //println!("CROSS {}", cross_vec);

        let mut vertices = Vec::<Vertex>::new();

        for i in 0..vertices_per_loop {
            let angle = std::f32::consts::PI * 2.0 * (i as f32 / vertices_per_loop as f32);
            let new_pos: Vector3<f32> = curr_seg.position + angle.cos() * right_vec1 + angle.sin() * right_vec2;
            //println!("angle {} {} {}", angle, angle.cos(), angle.sin());
            let new_vert = Vertex { position: new_pos };
            println!("MADE VERT {}", new_vert.position);
            //println!("new_vert {}", new_pos);
            vertices.push(new_vert);
        }

        Vec::from(vertices)
    }

    pub fn makeLoopMesh(&self) -> Mesh {
        let vertices_per_loop = 10 as usize;
        let mut vertices = Vec::<Vertex>::new();
        let mut triangles = Vec::<Triangle>::new();

        if self.segments.len() > 1 {
            for i in 0..self.segments.len() {
                println!("index of segment {}", i);
                let mut new_verts = self.makeLoopForSegment(i, vertices_per_loop);
                new_verts.push(Vertex::new(self.segments[i].position));
                let center_i = vertices.len() + new_verts.len() - 1;

                /*for vert_i in 0..new_verts.len() - 1 {
                    let triangle = Triangle::new(
                        vertices.len() + vert_i,
                        vertices.len() + (vert_i + 1) % (new_verts.len() - 1),
                        center_i
                    );
                    triangles.push(triangle);
                    println!("made triangle {} {}, {}", triangle.v1, triangle.v2, triangle.v3);
                }*/

                if i > 0 {
                    let prev_loop_vertex_0_i = (i - 1) * (vertices_per_loop + 1);
                    let curr_loop_vertex_0_i = i * (vertices_per_loop + 1);
                    for new_vert_i in 0..new_verts.len() - 1 {
                        let prev_loop_curr_vert_i = prev_loop_vertex_0_i + new_vert_i;
                        let prev_loop_next_vert_i = prev_loop_vertex_0_i + (new_vert_i + 1) % (vertices_per_loop + 1);
                        let curr_loop_curr_vert_i = curr_loop_vertex_0_i + new_vert_i;
                        let curr_loop_next_vert_i = curr_loop_vertex_0_i + (new_vert_i + 1) % (vertices_per_loop + 1);
                        triangles.push(Triangle::new(prev_loop_curr_vert_i, curr_loop_curr_vert_i, curr_loop_next_vert_i));
                        triangles.push(Triangle::new(prev_loop_curr_vert_i, prev_loop_next_vert_i, curr_loop_next_vert_i));
                    }
                }
                /*
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
                }*/
                vertices.append(&mut new_verts);
            }
        }
        println!("MADE VERTS {}", vertices.len());
        println!("MADE TRIANGLES {}", triangles.len());
        Mesh { vertices, triangles }
    }
}

fn get_any_vector_normal_to(vector: &Vector3<f32>) -> Vector3<f32> {
    if vector.z != 0.0 {
        let x = 1f32;
        let y = 1f32;
        let z = (-vector.x - vector.y) / vector.z;
        Vector3::new(x, y, z)
    } else if vector.y != 0.0 {
        let x = 1f32;
        let z = 1f32;
        let y = (-vector.x - vector.z) / vector.y;
        Vector3::new(x, y, z)
    } else if vector.x != 0.0 {
        let z = 1f32;
        let y = 1f32;
        let x = (-vector.y - vector.z) / vector.x;
        Vector3::new(x, y, z)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    }
}

fn find_angle_bisecting_vec(vector1: &Vector3<f32>, vector2: &Vector3<f32>) -> Vector3<f32> {
    let mut bisec: Vector3<f32> = vector1.cross(vector2);
    println!("GOT BISEC {} {}", bisec, bisec.magnitude_squared());
    if bisec.magnitude_squared() == 0.0 {
        return get_any_vector_normal_to(vector1);
    }
    return bisec
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: Vector3<f32>
}

impl Vertex {
    pub fn new(position: Vector3<f32>) -> Self {
        Vertex { position }
    }
}

#[derive(Copy, Clone)]
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