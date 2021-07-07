use image::{
    imageops::{resize, FilterType},
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Primitive,
    Rgb, RgbImage,
};
use imageproc::edges;
use num_traits::{Bounded, Num, NumCast, Pow, ToPrimitive};

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
        let (width, height) = img.dimensions();
        let gray = self.gleam(img);
        let gray = Self::double_the_size(gray);
        let edges = edges::canny(&gray, self.canny_low, self.canny_high);
        edges
    }

    /// Doubles the size of the image.
    /// This is a workaround for the fact that we don't have control over the Gaussian filter
    /// kernel size in `edges::canny`. Because we do know that blurring is applied, we
    /// apply simple filtering only when up-sampling.
    fn double_the_size(img: GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();
        resize(&img, width * 2, height * 2, FilterType::Triangle)
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
