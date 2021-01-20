use ::gl::types::*;
use ::glfw::{Context};
use std::ffi::{CString};
use std::ptr;

const VERTEX_SHADER_SOURCE: &str = r#"#version 330 core;
layout (location = 0) in vec3 inPos;

void main() {
    gl_Position = vec4(inPos, 1);
}
"#;

const FRAGMENT_SHADER_SOURCE: &str = r#"
    #version 330 core;

    out vec4 outFragmentColor;

    void main() {
        outFragmentColor = vec4(1, 0, 0, 1);
    }
"#;

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
        let id = shader_type.gl_id();
        let c_str_src = CString::new(source.clone()).unwrap().as_ptr();
        unsafe {
            gl::ShaderSource(id, 1, &c_str_src, ptr::null());
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

    let (shaderProgram, VAO) = unsafe {
        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        let c_str_vert = CString::new(VERTEX_SHADER_SOURCE).unwrap().as_ptr();
        gl::ShaderSource(vertex_shader, 1, &c_str_vert, ptr::null());
        gl::CompileShader(vertex_shader);

        let mut success = gl::FALSE as GLint;
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE as GLint {
            let mut infoLog = Vec::with_capacity(512);
            infoLog.set_len(512 - 1);
            let mut len: i32 = 0;
            gl::GetShaderInfoLog(vertex_shader, 512, &mut len, infoLog.as_mut_ptr());
            infoLog.set_len(len as usize);
            println!("vertex shader compilation failed:\n{}", std::str::from_utf8(&vec_i8_to_u8(infoLog)).unwrap());
        }

        let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        ((), ())
    };

    while !window.should_close() {
        unsafe {
            gl::ClearColor(0.2, 0.3, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
        
        window.swap_buffers();
        glfw.poll_events();
    }
}

/*fn process_events(window: &mut glfw::Window, events: &Receiver<(f64, glfw:WindowEvent)>) {
    for (_, event) in glfw::flush_messages(events) {
        match event {
            glfw::WindowEvent::Key()
        }
    }
}*/