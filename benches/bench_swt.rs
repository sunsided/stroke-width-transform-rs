use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stroke_width_transform::StrokeWidthTransform;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("SWT (ocr.png)", |b| {
        let img = image::open("images/ocr.png")
            .expect("test image is missing")
            .into_rgb8();
        let swt = StrokeWidthTransform::default();
        b.iter(|| swt.apply(black_box(&img)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
