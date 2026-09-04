# Third-party components

## PaddleOCR (upstream repo used for training + inference)

- Repo: https://github.com/PaddlePaddle/PaddleOCR
- Pinned commit: **`211989f046`**
- Required patch: **`patches/paddleocr-rec_lcnetv3-outofplace-add.patch`**

The patch changes one line in `LearnableRepLayer` (`ppocr/modeling/backbones/rec_lcnetv3.py`):
an in-place `out += conv(x)` becomes an out-of-place `out = out + conv(x)`. The in-place form
breaks autograd when the accumulator tensor is shared upstream — which happens under the
PP-OCRv6 recognizer's CTC/GTC dual-path — silently producing wrong / NaN gradients. **Any
reproduction of the paper's training must apply this patch.** (It does not affect inference,
so evaluating the released models does not require it.)

### Set up the fork

    git clone https://github.com/PaddlePaddle/PaddleOCR.git code/PaddleOCR
    cd code/PaddleOCR
    git checkout 211989f046
    git apply ../../patches/paddleocr-rec_lcnetv3-outofplace-add.patch

If `git apply` reports an offset/whitespace mismatch on a future PaddleOCR revision, the change
is a single line — edit `ppocr/modeling/backbones/rec_lcnetv3.py` in `LearnableRepLayer.forward`
and replace `out += conv(x)` with `out = out + conv(x)` in the `for conv in self.conv_kxk:` loop.

## Base model weights

`PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec` pretrained weights are fetched by the training
sbatch files from Baidu's official pretrained-model bucket (URLs are in the sbatch files).

## Datasets

See `DATA.md`. Each dataset retains its own license; we do not redistribute them.
