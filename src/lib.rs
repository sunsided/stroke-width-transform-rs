use image::{
    imageops::{resize, FilterType},
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Primitive,
    Rgb, RgbImage,
};
use imageproc::definitions::Image;
use imageproc::{edges, gradients};
use num_traits::real::Real;
use num_traits::{Bounded, Num, NumCast, Pow, ToPrimitive};

#[derive(Debug)]
struct Directions {
    x: Image<Luma<f32>>,
    y: Image<Luma<f32>>,
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct Position {
    x: u32,
    y: u32,
}

type Ray = Vec<Position>;

pub struct StrokeWidthTransform {
    one_over_gamma: f32,
    dark_on_bright: bool,
    canny_low: f32,
    canny_high: f32,
}

impl Default for StrokeWidthTransform {
    fn default() -> Self {
        let gamma = 2.2;
        Self {
            one_over_gamma: 1.0 / gamma,
            dark_on_bright: true,
            canny_low: 20.,
            canny_high: 75.,
        }
    }
}

impl StrokeWidthTransform {
    pub fn default_bright_on_dark() -> Self {
        Self {
            dark_on_bright: false,
            ..Self::default()
        }
    }

    /// Applies the Stroke Width Transformation to the image.
    pub fn apply(&self, img: &RgbImage) -> GrayImage {
        let gray = self.gleam(img);

        // Temporarily increase the image size for edge detection to work (better).
        let gray = Self::double_the_size(gray);
        let edges = self.get_edges(&gray);
        let directions = self.get_gradient_directions(&gray);

        // The grayscale image is not required anymore; we can free some memory.
        drop(gray);

        let swt = self.transform(edges, directions);

        swt
    }

    fn transform(&self, edges: GrayImage, directions: Directions) -> GrayImage {
        let mut rays: Vec<Ray> = Vec::new();

        let (width, height) = edges.dimensions();
        let mut swt: Image<Luma<u32>> = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let edge = unsafe { edges.unsafe_get_pixel(x, y) };
                // TODO: Verify edge value range, should be either 0 or 255.
                if edge[0] < 128 {
                    continue;
                }

                if let Some(ray) =
                    self.process_pixel(Position { x, y }, &edges, &directions, &mut swt)
                {
                    rays.push(ray);
                }
            }
        }

        let swt = convert_u32_to_u8_img(swt);
        // Next-generation println! debugging:
        // swt.save("swt-out.jpg");

        swt
    }

    /// Obtains the stroke width starting from the specified position.
    fn process_pixel(
        &self,
        pos: Position,
        edges: &GrayImage,
        directions: &Directions,
        swt: &mut Image<Luma<u32>>,
    ) -> Option<Ray> {
        // Keep track of the image dimensions for boundary tests.
        let (width, height) = edges.dimensions();

        // The direction in which we travel the gradient depends on the type of text
        // we want to find. For dark text on light background, follow the opposite
        // direction (into the dark are); for light text on dark background, follow
        // the gradient as is.
        let gradient_direction: f32 = if self.dark_on_bright { -1. } else { 1. };

        // Starting from the current pixel we will shoot a ray into the direction
        // of the pixel's gradient and keep track of all pixels in that direction
        // that still lie on an edge.
        let mut ray = Vec::new();
        ray.push(pos);

        // Obtain the direction to step into.
        // TODO: Obtain arctan of directions initially, then obtain dir_x and dir_y using cos and sin here.
        //       See below for another use of the directions.
        let dir_x = unsafe { directions.x.unsafe_get_pixel(pos.x, pos.y) }[0];
        let dir_y = unsafe { directions.y.unsafe_get_pixel(pos.x, pos.y) }[0];

        // Since some pixels have no gradient, normalization of the gradient
        // is a division by zero for them, resulting in NaN. These values
        // should not bother us since we explicitly tested for an edge before.
        debug_assert!(!dir_x.is_nan());
        debug_assert!(!dir_y.is_nan());

        // Traverse the pixels along the direction.
        let mut prev_pos = Position { x: 0, y: 0 };
        let mut steps_taken: usize = 0;
        loop {
            // Advance to the next pixel on the line.
            steps_taken += 1;
            let cur_x =
                (pos.x as f32 + gradient_direction * dir_x * steps_taken as f32).floor() as i64;
            let cur_y =
                (pos.y as f32 + gradient_direction * dir_y * steps_taken as f32).floor() as i64;

            // If we reach the edge of the image without crossing a stroke edge,
            // we discard the result.
            if (cur_x < 0 || cur_x >= width as _) || (cur_y < 0 || cur_y >= height as _) {
                return None;
            }

            // The cast is safe because we know the position lies within the image range.
            let cur_x = cur_x as u32;
            let cur_y = cur_y as u32;

            // If the step width was too small, continue;
            let cur_pos = Position { x: cur_x, y: cur_y };
            if cur_pos == prev_pos {
                continue;
            }
            prev_pos = cur_pos;

            // The point is either on the line or the end of it, so we register it.
            ray.push(cur_pos);

            // If that pixel is not an edge, we are still on the line and
            // need to continue scanning.
            let edge = unsafe { edges.unsafe_get_pixel(cur_x, cur_y) }[0];
            // TODO: Verify edge value range, should be either 0 or 255.
            if edge < 128 {
                continue;
            }

            // If this edge is pointed in a direction approximately opposite of the
            // one we started in, it is approximately parallel. This means we
            // just found the other side of the stroke.
            // The original paper suggests the gradients need to be opposite +/- PI/6.
            // Since the dot product is the cosine of the enclosed angle and
            // cos(pi/6) = 0.8660254037844387, we can discard all values that exceed
            // this threshold.
            // TODO: arctan + cos and sin.
            let cur_dir_x = unsafe { directions.x.unsafe_get_pixel(cur_x, cur_y) }[0];
            let cur_dir_y = unsafe { directions.y.unsafe_get_pixel(cur_x, cur_y) }[0];
            let dot_product = dir_x * cur_dir_x + dir_y * cur_dir_y;
            if dot_product >= -0.866 {
                return None;
            }

            // Paint each of the pixels on the ray with their determined stroke width.
            let delta_x = (cur_pos.x as i64 - pos.x as i64);
            let delta_y = (cur_pos.y as i64 - pos.y as i64);
            let stroke_width = ((delta_x * delta_x + delta_y * delta_y) as f32)
                .sqrt()
                .floor() as u32;

            for p in ray.iter() {
                unsafe {
                    swt.unsafe_put_pixel(p.x, p.y, [stroke_width].into());
                }
            }

            return Some(ray);
        }
    }

    /// Doubles the size of the image.
    /// This is a workaround for the fact that we don't have control over the Gaussian filter
    /// kernel size in `edges::canny`. Because we do know that blurring is applied, we
    /// apply simple filtering only when up-sampling.
    fn double_the_size(img: GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        resize(&img, width * 2, height * 2, FilterType::Triangle)
    }

    /// Opposite of `double_the_size`
    #[allow(unused)]
    fn halve_the_size<I>(img: Image<I>) -> Image<I>
    where
        I: Pixel + 'static,
    {
        let (width, height) = img.dimensions();
        resize(&img, width / 2, height / 2, FilterType::Gaussian)
    }

    /// Detects edges.
    fn get_edges(&self, img: &GrayImage) -> GrayImage {
        edges::canny(&img, self.canny_low, self.canny_high)
    }

    /// Detects image gradients.
    fn get_gradient_directions(&self, img: &GrayImage) -> Directions {
        let grad_x = gradients::horizontal_scharr(&img);
        let grad_y = gradients::vertical_scharr(&img);

        let (width, height) = img.dimensions();
        debug_assert_eq!(width, grad_x.dimensions().0);
        debug_assert_eq!(height, grad_x.dimensions().1);

        let mut out_x: Image<Luma<f32>> = ImageBuffer::new(width, height);
        let mut out_y: Image<Luma<f32>> = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let gx = unsafe { grad_x.unsafe_get_pixel(x, y) };
                let gy = unsafe { grad_y.unsafe_get_pixel(x, y) };

                let gx = gx[0].to_f32().unwrap();
                let gy = gy[0].to_f32().unwrap();

                let inv_norm = 1. / (gx * gx + gy * gy).sqrt();
                let gx = gx * inv_norm;
                let gy = gy * inv_norm;

                unsafe {
                    out_x.unsafe_put_pixel(x, y, [gx].into());
                    out_y.unsafe_put_pixel(x, y, [gy].into());
                }
            }
        }

        Directions { x: out_x, y: out_y }
    }

    /// Implements Gleam grayscale conversion from
    /// Kanan & Cottrell 2012: "Color-to-Grayscale: Does the Method Matter in Image Recognition?"
    /// http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740
    fn gleam(&self, image: &RgbImage) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut out: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let rgb = unsafe { image.unsafe_get_pixel(x, y) };

                let r = self.gamma(u8_to_f32(rgb[0]));
                let g = self.gamma(u8_to_f32(rgb[1]));
                let b = self.gamma(u8_to_f32(rgb[2]));
                let l = mean(r, g, b);
                let p = f32_to_u8(l);

                unsafe { out.unsafe_put_pixel(x, y, [p].into()) }
            }
        }

        out
    }

    /// Applies a gamma transformation to the input.
    #[inline]
    fn gamma(&self, x: f32) -> f32 {
        x.pow(self.one_over_gamma)
    }
}

#[inline]
fn u8_to_f32(x: u8) -> f32 {
    const SCALE_U8_TO_F32: f32 = 1.0 / 255.0;
    x.to_f32().unwrap() * SCALE_U8_TO_F32
}

#[inline]
fn f32_to_u8(x: f32) -> u8 {
    const SCALE_F32_TO_U8: f32 = 255.0;
    NumCast::from((x * SCALE_F32_TO_U8).clamp(0.0, 255.0)).unwrap()
}

#[inline]
fn mean(r: f32, g: f32, b: f32) -> f32 {
    const ONE_THIRD: f32 = 1.0 / 3.0;
    (r + g + b) * ONE_THIRD
}

/// Helper function to map u32 value range to u8 value range.
fn convert_u32_to_u8_img(image: Image<Luma<u32>>) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut out: GrayImage = ImageBuffer::new(width, height);

    let max_value = image.pixels().fold(0u32, |max, px| max.max(px[0]));
    let scaler = if max_value > 0 {
        255. / (max_value as f32)
    } else {
        0.
    };

    for y in 0..height {
        for x in 0..width {
            let pixel = unsafe { image.unsafe_get_pixel(x, y) };
            let v = pixel[0].to_f32().unwrap();
            let scaled = (v * scaler).to_u8().unwrap();
            unsafe { out.unsafe_put_pixel(x, y, [scaled].into()) }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_to_f32_works() {
        assert_eq!(u8_to_f32(0), 0.);
        assert_eq!(u8_to_f32(255), 1.);
    }

    #[test]
    fn f32_to_u8_works() {
        assert_eq!(f32_to_u8(1.0), 255);
        assert_eq!(f32_to_u8(2.0), 255);
        assert_eq!(f32_to_u8(-1.0), 0);
    }

    #[test]
    fn gamma_works() {
        let mut swt = StrokeWidthTransform {
            one_over_gamma: 2.,
            ..StrokeWidthTransform::default()
        };
        assert_eq!(swt.gamma(1.0), 1.0);
        assert_eq!(swt.gamma(2.0), 4.0);
    }

    #[test]
    fn mean_works() {
        assert_eq!(mean(-1., 0., 1.), 0.);
        assert_eq!(mean(1., 2., 3.), 2.);
        assert_eq!(mean(0., 0., 1.), 1. / 3.);
    }
}
