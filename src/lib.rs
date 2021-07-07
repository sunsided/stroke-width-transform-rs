use image::{
    imageops::{resize, FilterType},
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Primitive,
    Rgb, RgbImage,
};
use imageproc::definitions::Image;
use imageproc::{edges, gradients};
use num_traits::{Bounded, Num, NumCast, Pow, ToPrimitive};

#[derive(Debug)]
struct Directions {
    x: Image<Luma<f32>>,
    y: Image<Luma<f32>>,
}

#[derive(Debug, Copy, Clone, Default)]
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
    /// Applies the Stroke Width Transformation to the image.
    pub fn apply(&self, img: &RgbImage) -> GrayImage {
        let gray = self.gleam(img);

        // Temporarily increase the image size for edge detection to work (better).
        let gray = Self::double_the_size(gray);
        let edges = self.get_edges(&gray);
        let gradients = self.get_gradient_directions(&gray);

        // The grayscale image is not required anymore; we can free some memory.
        drop(gray);

        let swt = self.transform(edges, gradients);

        swt
    }

    fn transform(&self, edges: GrayImage, directions: Directions) -> GrayImage {
        let mut rays: Vec<Ray> = Vec::new();

        let (width, height) = edges.dimensions();
        let mut swt: GrayImage = ImageBuffer::new(width, height);

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

        edges
    }

    /// Obtains the stroke width starting from the specified position.
    fn process_pixel(
        &self,
        pos: Position,
        edges: &GrayImage,
        directions: &Directions,
        swt: &mut GrayImage,
    ) -> Option<Ray> {
        todo!()
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
