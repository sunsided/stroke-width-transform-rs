# Stroke Width Transform

This is a Rust re-implementation of the [sunsided/stroke-width-transform] repo.
It provides a test implementation of the Stroke Width Transform algorithm described
in the paper [Detecting Text in Natural Scenes with Stroke Width Transform] (PDF [here]):

> We present a novel image operator that seeks to find the value of stroke width for each image pixel, and demonstrate its use on the task of text detection in natural images. The suggested operator is local and data dependent, which makes it fast and robust enough to eliminate the need for multi-scale computation or scanning windows. Extensive testing shows that the suggested scheme outperforms the latest published algorithms. Its simplicity allows the algorithm to detect texts in many fonts and languages.

## Example

To apply SWT to an example image and show the result, run:

```console
$ cargo run --release --example swt -- --show images/train-station.jpg
```

The immediate SWT output is a map in which each pixel value corresponds to the
length of the stroke the pixel is on.

| Example Input       | Example Output (SWT)     |
|---------------------|--------------------------|
| ![](images/ocr.png) | ![](.readme/ocr-swt.jpg) |

## Original publication

```bibtex
@InProceedings{epshtein2010detecting,
    author = {Epshtein, Boris and Ofek, Eyal and Wexler, Yonatan},
    title = {Detecting Text in Natural Scenes with Stroke Width Transform},
    year = {2010},
    month = {June},
    abstract = {We present a novel image operator that seeks to find the value of stroke width for each image pixel, and demonstrate its use on the task of text detection in natural images. The suggested operator is local and data dependent, which makes it fast and robust enough to eliminate the need for multi-scale computation or scanning windows. Extensive testing shows that the suggested scheme outperforms the latest published algorithms. Its simplicity allows the algorithm to detect texts in many fonts and languages.},
    publisher = {IEEE - Institute of Electrical and Electronics Engineers},
    url = {https://www.microsoft.com/en-us/research/publication/detecting-text-in-natural-scenes-with-stroke-width-transform/},
}
```

## License

The code in this repository is available under the MIT license (see [LICENSE.md]).

[sunsided/stroke-width-transform]: https://github.com/sunsided/stroke-width-transform
[Detecting Text in Natural Scenes with Stroke Width Transform]: https://www.microsoft.com/en-us/research/publication/detecting-text-in-natural-scenes-with-stroke-width-transform/
[here]: https://github.com/sunsided/stroke-width-transform/blob/master/paper/201020CVPR20TextDetection.pdf
[LICENSE.md]: LICENSE.md
