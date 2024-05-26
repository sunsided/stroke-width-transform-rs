use clap::Parser;
use imageproc::window::display_image;
use std::time::Instant;
use stroke_width_transform::StrokeWidthTransform;

/// Testing Stroke Width Transforms.
#[derive(Parser)]
#[clap(version = "0.1", author = "Markus Mayer <widemeadows@gmail.com>")]
struct Opts {
    /// Enables bright on dark stroke detection.
    #[arg(short, long)]
    bright_on_dark: bool,

    /// Displays the image.
    #[arg(short, long)]
    show: bool,

    /// The image file to process.
    #[arg(value_parser = file_exists)]
    input: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts: Opts = Opts::parse();

    let img = image::open(opts.input)?;

    let swt = if opts.bright_on_dark {
        StrokeWidthTransform::default_bright_on_dark()
    } else {
        StrokeWidthTransform::default()
    };
    let start = Instant::now();
    let result = swt.apply(&img.into_rgb8());
    let duration = start.elapsed();

    println!("Processed image in {duration:?}");

    if opts.show {
        display_image("", &result, 500, 500);
    }

    Ok(())
}

fn file_exists(val: &str) -> Result<String, String> {
    let meta = match std::fs::metadata(val) {
        Ok(meta) => meta,
        Err(_) => return Err(format!("The specified file does not exist: {}", val)),
    };

    if !meta.is_file() {
        return Err(format!("Not a file: {}", val));
    }

    Ok(String::from(val))
}
