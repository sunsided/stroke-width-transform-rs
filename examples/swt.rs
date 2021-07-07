use clap::{AppSettings, Clap};
use image::{ImageBuffer, Luma};
use show_image::{
    create_window,
    event::{VirtualKeyCode, WindowEvent},
};
use stroke_width_transform_rs::StrokeWidthTransform;

/// Testing Stroke Width Transforms.
#[derive(Clap)]
#[clap(version = "0.1", author = "Markus Mayer <widemeadows@gmail.com>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
    /// Enables bright on dark stroke detection.
    #[clap(short, long)]
    bright_on_dark: bool,

    /// Displays the image.
    #[clap(short, long)]
    show: bool,

    /// The image file to process.
    #[clap(validator(file_exists))]
    input: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts: Opts = Opts::parse();

    let img = image::open(opts.input)?;
    let src = img.clone();

    let swt = if opts.bright_on_dark {
        StrokeWidthTransform::default_bright_on_dark()
    } else {
        StrokeWidthTransform::default()
    };
    let result = swt.apply(&img.into_rgb8());

    if opts.show {
        show_image::run_context(move || display_image(result));
    }

    Ok(())
}

fn display_image(img: ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<(), Box<dyn std::error::Error>> {
    let window = create_window("image", Default::default())?;
    window.set_image("Source", img)?;

    for event in window.event_channel()? {
        if let WindowEvent::KeyboardInput(event) = event {
            println!("{:#?}", event);
            if event.input.key_code == Some(VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }

    Ok(())
}

fn file_exists(val: &str) -> Result<(), String> {
    let meta = match std::fs::metadata(val) {
        Ok(meta) => meta,
        Err(_) => return Err(format!("The specified file does not exist: {}", val)),
    };

    if !meta.is_file() {
        return Err(format!("Not a file: {}", val));
    }

    Ok(())
}
