#![feature(proc_macro_hygiene, custom_inner_attributes)]

pub mod body;
pub mod diff;
pub mod linalg;
pub mod motion;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");
    let event_loop = winit::event_loop::EventLoopBuilder::new().build()?;
    let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new()
        .set_window_builder(
            winit::window::WindowBuilder::new()
                .with_title("Robotic Playground")
                .with_inner_size(winit::dpi::PhysicalSize::new(800, 480)),
        )
        .build(&event_loop);
    event_loop.run(move |event, event_loop_window_target| match event {
        winit::event::Event::WindowEvent { event: window_event, .. } => match window_event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop_window_target.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                use glium::{implement_vertex, Surface};

                let mut frame = display.draw();
                frame.clear_color(0.02, 0.02, 0.02, 1.0);

                let vertex_shader_src = r#"
                    #version 330

                    in vec2 position;

                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                    }
                "#
                .trim();

                let fragment_shader_src = r#"
                    #version 330

                    out vec4 color;

                    void main() {
                        color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                "#
                .trim();

                let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

                #[derive(Copy, Clone)]
                struct Vertex {
                    position: [f32; 2],
                }
                implement_vertex!(Vertex, position);

                let vertex1 = Vertex { position: [-0.5, -0.5] };
                let vertex2 = Vertex { position: [0.0, 0.5] };
                let vertex3 = Vertex { position: [0.5, -0.25] };
                let shape = vec![vertex1, vertex2, vertex3];
                let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
                let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
                frame
                    .draw(&vertex_buffer, indices, &program, &glium::uniforms::EmptyUniforms, &Default::default())
                    .unwrap();

                frame.finish().unwrap();
            }
            _ => (),
        },
        winit::event::Event::AboutToWait => {
            window.request_redraw();
        }
        _ => (),
    })?;
    Ok(())
}
