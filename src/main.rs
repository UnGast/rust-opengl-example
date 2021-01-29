use ::gl::types::*;
use glfw::ffi::{glfwGetTime, glfwGetWindowAttrib, glfwGetWindowFrameSize, glfwGetWindowSize};
use ::glfw::{Context};
use nalgebra::{Point3, Vector3};
use std::{ffi::{CStr, CString, c_void}, os::raw::c_char, thread::current, time::SystemTime};
use std::ptr;
use ::glam::*;
mod path;
use path::{Mesh, Path, PathSegment, Triangle, Vertex};

const VERTEX_SHADER_SOURCE: &str = r#"#version 330 core
layout (location = 0) in vec3 inPos;

uniform mat4 modelTransform;
uniform mat4 viewProjection;

void main() {
    gl_Position = viewProjection * modelTransform * vec4(inPos, 1);
}
"#;

const FRAGMENT_SHADER_SOURCE: &str = r#"#version 330 core

out vec4 outFragmentColor;

void main() {
    outFragmentColor = vec4(0.5, 0.5, 1, 1);
}
"#;

trait OpenGLStringConversionExt {
    fn as_gl(&self) -> CString;
}

impl OpenGLStringConversionExt for String {
    fn as_gl(&self) -> CString {
        CString::new(self.as_bytes()).unwrap()
    }
}

impl OpenGLStringConversionExt for str {
    fn as_gl(&self) -> CString {
        CString::new(self.as_bytes()).unwrap()
    }
}

fn vec_i8_to_u8(v: Vec<i8>) -> Vec<u8> {
    let mut v = std::mem::ManuallyDrop::new(v);

    let p = v.as_mut_ptr();
    let len = v.len();
    let cap = v.capacity();

    unsafe { Vec::from_raw_parts(p as *mut u8, len, cap) }
}
enum ShaderType {
    Vertex, Fragment
}

impl ShaderType {
    fn gl_id(&self) -> GLuint {
        match self {
            ShaderType::Vertex => gl::VERTEX_SHADER,
            ShaderType::Fragment => gl::FRAGMENT_SHADER
        }
    }
}

struct Shader {
    id: GLuint,
    shader_type: ShaderType,
    source: String
}

impl Shader {
    fn new(shader_type: ShaderType, source: String) -> Self {
        let id;
        unsafe {
            id = match shader_type {
                ShaderType::Vertex => gl::CreateShader(gl::VERTEX_SHADER),
                ShaderType::Fragment => gl::CreateShader(gl::FRAGMENT_SHADER),
            };

            gl::ShaderSource(id, 1, &source.as_gl().as_ptr(), ptr::null());
            gl::CompileShader(id);

            let mut success = gl::FALSE as GLint;
            gl::GetShaderiv(id, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                let mut info_log = Vec::with_capacity(512);
                info_log.set_len(512 - 1);
                let mut len: i32 = 0;
                gl::GetShaderInfoLog(id, 512, &mut len, info_log.as_mut_ptr());
                info_log.set_len(len as usize);
                println!("shader compilation failed:\n{}", std::str::from_utf8(&vec_i8_to_u8(info_log)).unwrap());
            }
        }
        
        Self { id, shader_type, source }
    }
}

struct ShaderProgram {
    id: GLuint,
    vertex_shader: Shader,
    fragment_shader: Shader
}

impl ShaderProgram {
    fn new(vertex_shader_source_: String, fragment_shader_source_: String) -> Self {
        let vertex_shader = Shader::new(ShaderType::Vertex, vertex_shader_source_);
        let fragment_shader = Shader::new(ShaderType::Fragment, fragment_shader_source_);

        let id;
        unsafe {
            id = gl::CreateProgram();
            gl::AttachShader(id, vertex_shader.id);
            gl::AttachShader(id, fragment_shader.id);
            gl::LinkProgram(id);

            let mut success = gl::FALSE as GLint;
            gl::GetProgramiv(id, gl::LINK_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                let mut info_log = Vec::with_capacity(512);
                info_log.set_len(512 - 1);
                let mut len: i32 = 0;
                gl::GetProgramInfoLog(id, 512, &mut len, info_log.as_mut_ptr());
                info_log.set_len(len as usize);
                println!("linking program failed:\n{}", std::str::from_utf8(&vec_i8_to_u8(info_log)).unwrap());
            }
        }

        Self { id, vertex_shader, fragment_shader }
    }

    fn set_uniform_mat4(&self, name: &str, matrix: Mat4) {
        unsafe {
            gl::UniformMatrix4fv(gl::GetUniformLocation(self.id, name.as_gl().as_ptr()), 1, gl::FALSE, matrix.to_cols_array().as_ptr());
        }
    }
}

fn main() {
    println!("Hello, world!");

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

    let (mut window, events) = glfw.create_window(500, 500, "OpenGL", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window");

    window.make_current();

    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    let shaderProgram = ShaderProgram::new(VERTEX_SHADER_SOURCE.to_string(), FRAGMENT_SHADER_SOURCE.to_string());

    let mut test_path = Path::new(Vec::from([
        PathSegment::new(Vector3::new(-1.0, 0.0, 0.0)),
        PathSegment::new(Vector3::new(-0.5, 0.0, 0.0)),
        PathSegment::new(Vector3::new(0.0, 0.0, 0.0)),
        PathSegment::new(Vector3::new(0.5, 0.0, 0.0)),
        PathSegment::new(Vector3::new(1.0, 0.0, 00.0)),
    ]));

    let mut test_mesh = Mesh::new(Vec::from([
        // bottom
        Vertex::new(Vector3::new(0.0, 0.0, 0.0)),
        Vertex::new(Vector3::new(1.0, 0.0, 0.0)),
        Vertex::new(Vector3::new(1.0, 0.0, 1.0)),
        Vertex::new(Vector3::new(0.0, 0.0, 1.0)),

        // front
        Vertex::new(Vector3::new(0.0, 1.0, 0.0)),
        Vertex::new(Vector3::new(1.0, 1.0, 0.0)),

        // right
        //Vertex::new(Vector3::new())
    ]), Vec::from([
        // bottom
        Triangle::new(0, 1, 2),
        Triangle::new(0, 2, 3),
        // front
        Triangle::new(0, 4, 5),
        Triangle::new(0, 1, 5),
        // right
        //Triangle::new()
    ]));
    test_mesh = test_path.makeLoopMesh();

    let (mut vao, mut vbo) = (0, 0);
    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::BindVertexArray(vao);

        // opengl matrix math
        // https://solarianprogrammer.com/2013/05/22/opengl-101-matrices-projection-view-model/

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 3 * std::mem::size_of::<f32>() as i32, ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::BindVertexArray(0);
    }

    let mut last_frame_time = SystemTime::now();
    let model_rotation_speed = Vec3::new(0.5, 0.5, 0.0);
    let mut model_rotation = Vec3::zero();

    while !window.should_close() {
        let current_frame_time = SystemTime::now();
        let delta_time = current_frame_time.duration_since(last_frame_time).unwrap();
        last_frame_time = current_frame_time;

        unsafe {
            let (window_width, window_height) = window.get_size();
            
            gl::Viewport(0, 0, window_width, window_height);

            gl::ClearColor(0.0, 0.0, 0.4, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            
            gl::BindVertexArray(vao);
            gl::UseProgram(shaderProgram.id);

            let mut raw_vertices = test_mesh.unfold_vertices();
            let mut vertices = raw_vertices.iter().fold(
            Vec::<f32>::new(), 
            |res, vertex| {
                [res, Vec::<f32>::from([vertex.position.x, vertex.position.y, vertex.position.z])].concat()
            });
            /*vertices = vec![
                0.0, 0.0, 5.0,
                1.0, 1.0, 5.0,
                0.0, 1.0, 5.0,
                0.0, 0.0, 5.0,
                1.0, 0.0, 5.0,
                1.0, 1.0, 5.0
            ];*/

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(gl::ARRAY_BUFFER, vertices.len() as isize * std::mem::size_of::<f32>() as isize,
                &vertices[0] as *const f32 as *const c_void,
                gl::STATIC_DRAW);

            let delta_rotation = model_rotation_speed * (delta_time.as_millis() as f32 / 1000.0);
            model_rotation += delta_rotation;
            model_rotation.x = model_rotation.x % 1.0;
            model_rotation.y = model_rotation.y % 1.0;
            model_rotation.z = model_rotation.z % 1.0;
            let model_rotation_angle = model_rotation * std::f32::consts::PI * 2.0;

            let model_transform = Mat4::from_translation(Vec3::new(0.0, 0.0, 10.0))
                .mul_mat4(&Mat4::from_rotation_ypr(model_rotation_angle.x, model_rotation_angle.y, model_rotation_angle.z));

            let player_pos = Vec3::new(0.0, 10.0, 0.0);
            let player_look_direction = Vec3::new(0.0, -0.5, 1.0);
            let view = Mat4::look_at_rh(player_pos + player_look_direction, player_pos, Vec3::new(0.0, 1.0, 0.0));
            let projection = Mat4::perspective_lh(::std::f32::consts::PI / 2.0, window_width as f32 / window_height as f32, 0.0, 1000.0);
            let view_projection = projection.mul_mat4(&view);

            let test_vec = const_vec4!([1.0, 0.0, 10.0, 1.0]);
            let test_result = view.mul_vec4(test_vec);
            let test_result_finished = test_result;

            //println!("test_vec {}", test_result_finished);
            shaderProgram.set_uniform_mat4("viewProjection", view_projection);
            shaderProgram.set_uniform_mat4("modelTransform", model_transform);
            //gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
            gl::DrawArrays(gl::TRIANGLES, 0, raw_vertices.len() as i32);
        }
        
        window.swap_buffers();
        glfw.poll_events();
    }
}