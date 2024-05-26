use stroke_width_transform::StrokeWidthTransform;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img_1 = image::open("images/ocr.png")?;
    let img_2 = image::open("images/text.jpg")?;
    let img_3 = image::open("images/train-station.jpg")?;

    let swt = StrokeWidthTransform::default();
    let _ = swt.apply(&img_1.into_rgb8());
    let _ = swt.apply(&img_2.into_rgb8());
    let _ = swt.apply(&img_3.into_rgb8());
    Ok(())
}
