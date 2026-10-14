# ffrwd/faceage

Ages on faces, for blurring the children in a video and nobody else.
`ages` reads the picture and the face boxes `ffrwd/rfdetr`'s
`detect_faces` writes about it, and returns the same boxes with an age
on each; the query keeps the rows under an age; `ffrwd/rfdetr`'s
`boxes_mask` turns those into a matte; and `ffrwd/mask_tools` blurs or
mosaics where the matte says. Everything downstream of the model is
native ffmpeg.

Requires ffrwd 0.30, whose `ffrwd/wasm` is 0.19.1.

```
ffrwd run ffrwd/faceage:blur-children -v source=class.mp4 -v max_age=16 -v dest=blurred.mp4
```

## Model export

- `ages(v, boxes, hold DEFAULT 5, every DEFAULT 1)` returns
  `STRUCT(class text, conf number, x number, y number, w number, h number, age number, start_t number, id number)[]`:
  one row per face it is following on each frame, the class,
  confidence and box it was handed with `age` added, in years to one
  decimal, and the face's name: `start_t`, the time it was first seen,
  and `id`, how many faces were first seen before it. `boxes` is what
  `detect_faces` returns, and both name the same picture, so the call is
  written `ages(v, ffrwd.rfdetr.detect_faces(v))`. The picture is
  fetched only on a frame where a face is due a score.

```sql
SELECT ffrwd.mask_tools.blur_where(
         v, ffrwd.rfdetr.boxes_mask(
              v,
              ARRAY(SELECT r FROM unnest(ffrwd.faceage.ages(v, ffrwd.rfdetr.detect_faces(v))) r
                    WHERE r.age < 16),
              grow => 8, feather => 4))
```

The call is written once and runs once; the gather narrows its rows on
their way to the matte.

A face is not scored afresh every frame. The module follows boxes from
one frame to the next by overlap. A new face is scored on each of its
first three frames and then once every `every` frames, so the work is
about one inference per face per `every` frames. The age a row carries
is the mean of the scores that face took over its last thirty frames,
and never of fewer than three, so the number settles instead of
jittering between frames. A far-off first score is outvoted on the
next frame, as at `every` 1. Because the window is counted in frames
rather than scores, an age follows a change as fast at `every` 5 as at
1: when the tracker hands an adult's track to a child beside them, the
child's own scores outvote the adult's in about a dozen frames. `hold`
keeps a face the detector lost for that many frames, re-emitted at its
last box, so a blur does not flicker off for the one frame a head
turns. Both are for video: on a slideshow of unrelated stills set
`hold` to 0, or a face in the same place on the next still is taken
for the same person.

The graph reads a crop of each box padded by ten percent on every
side, resized to 224x224 the way the model was trained, off the frame
ffmpeg hands the module already converted to RGB with the stream's own
range and matrix. It also carries a gender head, which this module
does not read.

## On the GPU

`ages_gpu(v, boxes, hold, every, memory DEFAULT 'gpu')` is the GPU build
of `ages`, with the same rows. The host makes the frame's due crops into
one batched tensor (`ffrwd:gpu/nn.crops-tensor`), cropped the same way
and resized with Pillow's bicubic kernel, and the model runs once a
frame for every face due a score. With `memory => 'gpu'` the frame stays
on the sidecar's device and is never read back; with `'host'` the
tensor is made on the CPU. On the classroom check clip the tensor is
within a mean of 0.001 of the host build's (0 to 1 units; the host build
rounds to 8 bits between its two resize passes), each face's age is
within 0.4 years of the host build's for every child and 1 year for the
adult, and the same faces are under 16 on every frame. `ages_gpu`
needs ffrwd 0.30.

`blur-children-gpu` and `mosaic-children-gpu` are the recipes with
every step's GPU build: `ffrwd/rfdetr`'s `detect_faces_gpu` and
`boxes_mask_gpu`, and `ffrwd/mask_tools`'s `blur_where_gpu` or
`mosaic_where_gpu`.

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

## Mattes

`ffrwd/rfdetr`'s `boxes_mask` reads `x`, `y`, `w` and `h` and lets
every other field pass, so it rasterizes `ages`' rows as they are.
0.1's `faces_mask`, a copy of it that took the seven-field row, is gone.

## Recipes

- `blur-children`: every face aged, the ones under `max_age` blurred.
  `max_age` is required; `conf`, `grow`, `feather`, `sigma`, `hold`,
  `every` and `track` are optional.
- `mosaic-children`: the same with a mosaic, `size` in place of
  `sigma`.
- `ages`: every face with its age as NDJSON, one line per face per
  frame, no video written.
- `blur-children-gpu`, `mosaic-children-gpu`: the first two with every
  step on the GPU.

Run `ffrwd list ffrwd/faceage` for each one's variables, or read the
header of the recipe file. To blur every face regardless of age, use
`ffrwd/rfdetr`'s `blur-faces`; under HIPAA a full-face image is an
identifier at any age, and the age filter here is for a policy that
exempts consenting adults.

## Running it

The age model is a 300M-parameter transformer. A frame's due faces run
through it as one batch: on an RTX 4090 through CUDA, one face takes
3.4 ms, five take 6.8 ms and ten take 12.1 ms.

A face is scored on each of its first three frames and then once every
`every` frames. The four blur and mosaic recipes use `every` 5 unless
it is given; `ages` and `ages_gpu` called directly, and the `ages`
recipe, keep 1. On the classroom check clip, five faces a frame,
`every` 5 blurs the same faces on every frame as `every` 1, each
child's age stays within 0.8 years of `every` 1's and the adult's
within 2.3, and the chain runs about 50 percent faster:

| `every` | 720p | 4K |
|---|---|---|
| 1 | 67 | 62 |
| 2 | 83 | 74 |
| 3 | 88 | 85 |
| 5 | 103 | 92 |
| 10 | 116 | 101 |

These are sustained frames a second through `blur-children-gpu` with
`h264_d3d12va` encoding on the 4090, start-up excluded. The whole
343-frame clip, start-up included, runs at 52 frames a second at 720p
and 47 at 4K at `every` 5, against 40 and 37 at `every` 1. `every` 10
also changed no blur decision on the clip and is about a tenth faster
at 4K, but its age is the mean of three scores rather than six, so one
far-off score moves it further. Give `every` 1 to score every face on
every frame.

DirectML loads this graph and then fails inside the backbone on its
first inference, so the model's pin says `not_on: directml`. The
sidecar's GPU walk then takes a provider no model of the run excludes:
on Windows that is CUDA where CUDA is installed, and on a machine
without it the CPU runs the model at about seven times the wall clock.

The module is declared impure on purpose: a tracker carries state
from one frame to the next, and sharing the stream among parallel
instances would hand each of them every other frame, where two people
in one place are one face by overlap.

## From 0.1

0.1's `ages` handed the picture back with its rows beside it, and the
recipes wrote `ages(...)` twice, once for the picture and once for the
rows the gather narrowed. 0.2 writes the rows alone and the recipes
write the call once: `ages(v, detect_faces(v))`, its rows into
`boxes_mask(v, ...)`. A row gains `start_t` and `id`, which name the
face it belongs to.

On the check clip of sixty faces the rows are 0.1's row for row, the
same boxes, confidences and ages, and `blur-children` and
`mosaic-children` are the same frame for frame. Each row carries the
`pts` and `time` it did, and a face's `start_t` and `id` beside them.

## Building

```
cargo build --target wasm32-wasip2 --release
cargo test --release
ffrwd install
```

`ages` is a node written with
[`ffrwd-node`](https://github.com/imbcmdth/ffrwd-node), which carries
the world it is built against, so there is no `build.rs` and nothing to
install before building; `common` holds the graph run through
`wasi:nn`. It crops, resizes and normalizes with the
[ffrwd-frame](https://github.com/imbcmdth/ffrwd-frame) crate.
`ages_gpu.wasm` is the same source with the `gpu` feature on, built by
the `ages-gpu` crate, with the `ffrwd:gpu` bindings over the copy of its
wit in `ages/wit-gpu/`. The
package depends on `ffrwd/rfdetr` 0.5.0 for `detect_faces` and
`boxes_mask`, which `ffrwd install` fetches with the model.

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
them. The pinned file, `faceage-dino-vitl16-fused-fp16.onnx`, is the
source network rebuilt with fused ONNX Runtime operators: its MatMuls
and attention are fp16, and its residual stream, LayerNorms, input and
outputs fp32. It is 611 MB and needs ONNX Runtime 1.22 or later, the
version `ffrwd setup nn` installs. It lives in
[imbcmdth/faceage-onnx](https://huggingface.co/imbcmdth/faceage-onnx)
with the script that builds it, the validation against the source (a
mean age difference of 0.0036 years on the CPU and 0.0086 through
CUDA) and the license beside it, and runs through `wasi:nn` on the
machine's own ONNX Runtime.
