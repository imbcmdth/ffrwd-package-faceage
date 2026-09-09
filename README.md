# ffrwd/faceage

Ages on faces, for blurring the children in a video and nobody else.
`ages` takes the face boxes `ffrwd/rfdetr`'s `detect_faces` puts
beside the picture and returns the same boxes with an age on each;
the query keeps the rows under an age; `faces_mask` turns those into a
matte; and `ffrwd/mask_tools` blurs or mosaics where the matte says.
Everything downstream of the model is native ffmpeg.

```
ffrwd run ffrwd/faceage:blur-children -v source=class.mp4 -v max_age=16 -v dest=blurred.mp4
```

## Model export

- `ages(v, boxes, hold DEFAULT 5, every DEFAULT 1)` returns
  `STRUCT(v video_stream, faces STRUCT(class text, conf number, x number, y number, w number, h number, age number)[])` -
  the picture untouched, and every face row it was handed with `age`
  added, in years to one decimal. `boxes` is the record `detect_faces`
  returns, so the call is written `ages(ffrwd.rfdetr.detect_faces(v))`.

A face is not scored afresh every frame. The module follows boxes from
one frame to the next by overlap, and the age a row carries is the
mean of that face's last thirty scores, so the number settles instead
of jittering between frames. `every` is how many frames apart one face
is scored - the work is one inference per face per `every` frames -
and `hold` keeps a face the detector lost for that many frames,
re-emitted at its last box, so a blur does not flicker off for the one
frame a head turns. Both are for video: on a slideshow of unrelated
stills set `hold` to 0, or a face in the same place on the next still
is taken for the same person.

The graph reads a crop of each box padded by ten percent on every
side, resized to 224x224 the way the model was trained, off the frame
ffmpeg hands the module already converted to RGB with the stream's own
range and matrix. It also carries a gender head, which this module
does not read.

## Choosing `max_age`

The recipes keep a face when its estimated age is under `max_age`.
That predicate is evaluated per row while the modules run and holds a
literal alone, so `max_age` has no default and has to be given.

Pick it with a margin. The model's error is smallest where it
matters here: on its own benchmark it is 1.5 years off on ages 0 to 12
and 2.9 on 13 to 17, and on a check of sixty labelled faces through
this package it was 1.3 years off under 16, with no child estimated
over 25 and no adult under 16. For a room of young children and adult
teachers, 16 leaves every child well inside the line and every adult
well outside it. Where teenagers and young adults share the frame the
line is closer and the error matters more; the `ages` recipe prints
every face with its estimate, which is how to see where a threshold
lands before committing to it.

Missing a face is the error that costs, so keep `conf` low and lean
on the margin rather than the threshold. Faces the detector never
finds are never blurred: see the `ffrwd/rfdetr` README for how face
size in the frame decides that.

## Utility

- `faces_mask(v, faces, grow DEFAULT 0, feather DEFAULT 0)` - the rows
  rasterized into a matte; `grow` pads each box in pixels, `feather`
  softens the edge. It exists because `boxes_mask` in `ffrwd/rfdetr`
  declares the six-field row and the compiler wants an exact match;
  this one takes the seven-field row `ages` returns.

## Recipes

- `blur-children` - every face aged, the ones under `max_age` blurred.
  `max_age` is required; `conf`, `grow`, `feather`, `sigma`, `hold`,
  `every` and `track` are optional.
- `mosaic-children` - the same with a mosaic, `size` in place of
  `sigma`.
- `ages` - every face with its age as NDJSON, one line per face per
  frame, no video written.

Run `ffrwd list ffrwd/faceage` for each one's variables, or read the
header of the recipe file. To blur every face regardless of age, use
`ffrwd/rfdetr`'s `blur-faces`; under HIPAA a full-face image is an
identifier at any age, and the age filter here is for a policy that
exempts consenting adults.

## Running it

The age model is a 300M-parameter transformer run once per face per
`every` frames. On an RTX 4090 through CUDA that is about five
milliseconds a face, so ten faces at 30 frames a second cost about a
second and a half of GPU time per second of video at `every` 1, and a
third of a second at `every` 5, where the estimate is the same to a
tenth of a year.

DirectML loads this graph and then fails inside the backbone on its
first inference, so the model's pin says `not_on: directml` and the
sidecar's GPU walk skips it: on Windows that lands on CUDA, and on a
machine without it the CPU runs the model at about seven times the
wall clock.

The module is declared impure on purpose: a tracker carries state
from one frame to the next, and sharing the stream among parallel
instances would hand each of them every other frame, where two people
in one place are one face by overlap.

## Building

The modules build against the wit from the installed `ffrwd/wasm`
package and crop, resize and normalize with the
[ffrwd-frame](https://github.com/imbcmdth/ffrwd-frame) crate. The
package depends on `ffrwd/rfdetr` 0.2.0 for `detect_faces`:

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
ffrwd install
```

## License

This package is **Apache-2.0**, and so are the weights. The age model
is [FaceAge ClientScan](https://huggingface.co/TrungTran/faceage_ClientScan),
TrungTran's DINOv3 ViT-L/16 with a CORAL ordinal-regression head,
released under Apache-2.0. Built with DINOv3: the backbone is Meta's,
under the [DINOv3 License](https://huggingface.co/imbcmdth/faceage-onnx/blob/main/DINOv3_LICENSE.md),
which permits this use and asks that the license travel with the
model and that the attribution be shown.

The weights are not in the archive: the manifest pins them - repo,
revision, file and sha256 - and `ffrwd install` fetches and verifies
them. The pinned file is the source's fp32 ONNX halved to fp16 with
its input and outputs left fp32, 612 MB, and lives in
[imbcmdth/faceage-onnx](https://huggingface.co/imbcmdth/faceage-onnx)
with the conversion scripts, the validation against the source (a
mean age difference of 0.012 years) and the license beside it, run
through `wasi:nn` on the machine's own ONNX Runtime.
