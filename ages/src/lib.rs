//! Ages on the faces a detector already found: the boxes it wrote for each
//! frame come back with an age on each. Every box is cropped with the ten
//! percent padding the model was trained on, resized into its 224 square with
//! Pillow's bicubic, and run through the graph; the CORAL head's hundred
//! thresholds sum to an age in years.
//!
//! A face is not scored afresh every frame. Boxes are followed from frame to
//! frame by overlap, and a face carries the running mean of its own last
//! thirty scores, so the number a row reports settles rather than jitters.
//! `every` is how many frames apart a face is scored, so the work is one
//! inference per face per `every` frames, and `hold` keeps a face the
//! detector lost, re-emitted where it was last seen, so what reads these rows
//! does not flicker for the frame a head turns. Each row names its face by
//! the time it was first seen, `start_t`, and by how many faces were first
//! seen before it, `id`.
//!
//! The graph is FaceAge ClientScan, a DINOv3 backbone with a CORAL age head,
//! run through `wasi:nn` and bound to the name `ages`. The gender head the
//! graph also carries is not read. The picture is fetched only on a frame
//! where a face is due a score.
//!
//! The frame arrives as `rgba`, so the colour was settled upstream by ffmpeg,
//! which reads the stream's own range and matrix rather than guessing at them;
//! `ffrwd-frame` takes it from there. That the range is right is not
//! fussiness: a limited-range frame taken for a full-range one is a face
//! flattened by a seventh, and a flatter face reads as a younger one. On the
//! check clip that alone was worth two years of mean error and put a face of
//! 66 at 39.
//!
//! One source, two builds. `ages.wasm` is the module above, cropping on the
//! CPU. `ages_gpu.wasm` (the `gpu` feature, built by the `ages-gpu` crate) is
//! a node beside `ffrwd:gpu`: the host makes each frame's due crops into one
//! tensor, the same crops made the same way, and on `memory=gpu` the frame
//! stays on the device. See `gpu.rs`.

#[cfg(feature = "gpu")]
mod gpu;

use std::collections::VecDeque;

use faceage_common::nn::{Model, Returned};
use faceage_common::{coral_age, iou, PAD};
use ffrwd_frame::Rect;
#[cfg(not(feature = "gpu"))]
use ffrwd_frame::{Filter, Rgba, IMAGENET};
use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

/// What the export calls its input tensor.
const INPUT_NAME: &str = "face";

/// The CORAL head's width: a hundred ordinal thresholds, which is what tells
/// `age_logits` from the two-wide gender head beside it.
const THRESHOLDS: usize = 100;

/// The class every row carries. The boxes read are faces and the rows written
/// are the same faces.
const FACE: &str = "face";

/// How many of a face's own scores its age is the mean of.
const MEMORY: usize = 30;

/// How much two boxes have to overlap to be the same face a frame later.
const MATCH_IOU: f64 = 0.3;

#[cfg(not(feature = "gpu"))]
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"hold":{"type":"integer","minimum":0,"maximum":600,"default":5},"every":{"type":"integer","minimum":1,"maximum":600,"default":1}},"additionalProperties":false}"#;

/// The GPU build's: the same two, and where `v` is held.
#[cfg(feature = "gpu")]
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"hold":{"type":"integer","minimum":0,"maximum":600,"default":5},"every":{"type":"integer","minimum":1,"maximum":600,"default":1},"memory":{"type":"string","enum":["host","gpu"],"default":"host"}},"additionalProperties":false}"#;

/// The rows this module reads: a detector's face boxes. Other fields pass.
const BOXES_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number"},"x":{"type":"number"},"y":{"type":"number"},"w":{"type":"number"},"h":{"type":"number"}},"required":["class","conf","x","y","w","h"]}"#;

const FACES_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number"},"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"},"age":{"type":"number"},"start_t":{"type":"number"},"id":{"type":"integer"}},"required":["class","conf","x","y","w","h","age","start_t","id"],"additionalProperties":false}"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub(crate) struct Params {
    hold: u32,
    every: u32,
}

impl Default for Params {
    fn default() -> Self {
        Params { hold: 5, every: 1 }
    }
}

/// One row this module writes: the box it was handed, with the age of the
/// face in it and the face's name.
#[derive(Serialize)]
struct Row {
    class: String,
    conf: f64,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
    age: f64,
    start_t: f64,
    id: u64,
}

/// One row this module reads.
#[derive(Deserialize)]
struct InRow {
    class: String,
    conf: f64,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// One box on the frame: x, y, width and height in pixels.
type FaceBox = (f64, f64, f64, f64);

/// The crop a box names: the box widened by `PAD` of its own width and height
/// on every side and clamped to the frame. None when nothing of it lands
/// there, which is a box there is no face to score.
fn crop(b: FaceBox, width: usize, height: usize) -> Option<Rect> {
    Rect::padded(b.0, b.1, b.2, b.3, PAD, width, height)
}

/// One face the detector reported on this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Det {
    conf: f64,
    b: FaceBox,
}

/// One face followed across frames.
#[derive(Debug)]
struct Track {
    /// When it was first seen, and how many faces were first seen before it.
    start_t: f64,
    id: u64,
    /// Where it was last seen, and how sure the detector was of it.
    b: FaceBox,
    conf: f64,
    /// Its last `MEMORY` scores, oldest first.
    scores: VecDeque<f64>,
    /// Frames since it was last scored.
    since: u32,
    /// Consecutive frames the detector has not found it.
    missed: u32,
}

impl Track {
    fn new(det: Det, start_t: f64, id: u64) -> Track {
        Track {
            start_t,
            id,
            b: det.b,
            conf: det.conf,
            scores: VecDeque::with_capacity(MEMORY),
            since: 0,
            missed: 0,
        }
    }

    /// One more score, the oldest dropped once there are `MEMORY` of them.
    fn scored(&mut self, age: f64) {
        if self.scores.len() == MEMORY {
            self.scores.pop_front();
        }
        self.scores.push_back(age);
        self.since = 0;
    }

    /// The age this face reports: the mean of what it has scored, to one
    /// decimal, since neither the model nor the mean of a handful of its
    /// answers says anything finer.
    fn age(&self) -> f64 {
        let sum: f64 = self.scores.iter().sum();
        let mean = sum / self.scores.len().max(1) as f64;
        (mean * 10.0).round() / 10.0
    }

    /// Whether this face is due a score: always the first time, and then
    /// whenever `every` frames have gone by since the last one.
    fn due(&self, every: u32) -> bool {
        self.scores.is_empty() || self.since >= every
    }

    fn row(&self) -> Row {
        Row {
            class: FACE.to_string(),
            // To four places: the detector's own precision is nowhere near the
            // sixteen digits an f32 widened to an f64 prints.
            conf: (self.conf * 10_000.0).round() / 10_000.0,
            x: self.b.0.round() as i64,
            y: self.b.1.round() as i64,
            w: self.b.2.round() as i64,
            h: self.b.3.round() as i64,
            age: self.age(),
            start_t: self.start_t,
            id: self.id,
        }
    }
}

/// Which returned tensor is the age head, by shape: the CORAL thresholds are a
/// hundred wide and the gender head beside them is two, so the name is never
/// read and either order resolves.
fn age_output(shapes: &[Vec<u32>]) -> Result<usize, String> {
    shapes
        .iter()
        .position(|dimensions| matches!(dimensions.as_slice(), [_, width] if *width as usize == THRESHOLDS))
        .ok_or_else(|| {
            format!(
                "ages: the graph returned {shapes:?}, and this module wants \
                 the CORAL head's [batch, {THRESHOLDS}] age logits"
            )
        })
}

/// A frame's rows as the boxes to score: what is not a face is not this
/// module's, and a box with no area is not a face.
fn faces_of(rows: Vec<InRow>) -> Vec<Det> {
    rows.into_iter()
        .filter(|row| row.class == FACE && row.w > 0.0 && row.h > 0.0)
        .map(|row| Det {
            conf: row.conf,
            b: (row.x, row.y, row.w, row.h),
        })
        .collect()
}

/// This frame's boxes against the faces already being followed: the surest
/// overlap first, one box to one face, and nothing paired under `MATCH_IOU`.
/// The answer is one slot per box, naming the face it is, or none for a box
/// that is a face not seen before.
fn match_boxes(tracks: &[Track], dets: &[Det]) -> Vec<Option<usize>> {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (d, det) in dets.iter().enumerate() {
        for (t, track) in tracks.iter().enumerate() {
            let overlap = iou(det.b, track.b);
            if overlap >= MATCH_IOU {
                pairs.push((overlap, d, t));
            }
        }
    }
    // Greatest overlap first; ties settled by the order the boxes and the
    // tracks are already in, so a frame's answer never depends on the sort.
    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let mut of_det = vec![None; dets.len()];
    let mut taken = vec![false; tracks.len()];
    for (_, d, t) in pairs {
        if of_det[d].is_none() && !taken[t] {
            of_det[d] = Some(t);
            taken[t] = true;
        }
    }
    of_det
}

/// One frame: the boxes matched to the faces being followed, the ones due a
/// score scored, and a row for every face still on the picture.
///
/// `score` is handed the boxes to run, in the order they are to come back, and
/// is the only part of this that needs the graph - which is what lets the
/// following, the cadence and the carry-forward be tested without one. It is
/// not called at all when nothing is due.
fn advance(
    faces: &mut Faces,
    dets: &[Det],
    params: Params,
    score: impl FnOnce(&[FaceBox]) -> Result<Vec<f64>, String>,
) -> Result<Vec<Row>, String> {
    let Faces { tracks, begun, now } = faces;
    let of_det = match_boxes(tracks, dets);

    // The boxes seen this frame, onto the faces they belong to. A box that
    // matched nothing is a new face, appended in the order the rows arrived.
    let mut matched = vec![false; tracks.len()];
    let mut seen: Vec<(usize, Det)> = Vec::with_capacity(dets.len());
    for (d, det) in dets.iter().enumerate() {
        let at = match of_det[d] {
            Some(t) => {
                matched[t] = true;
                t
            }
            None => {
                tracks.push(Track::new(*det, *now, *begun));
                *begun += 1;
                matched.push(true);
                tracks.len() - 1
            }
        };
        seen.push((at, *det));
    }

    // What is due a score, at the box this frame put it at.
    let mut due: Vec<usize> = Vec::new();
    let mut boxes: Vec<FaceBox> = Vec::new();
    for (at, det) in &seen {
        if tracks[*at].due(params.every) {
            due.push(*at);
            boxes.push(det.b);
        }
    }
    if !due.is_empty() {
        let ages = score(&boxes)?;
        if ages.len() != due.len() {
            return Err(format!(
                "ages: {} crops went to the graph and {} ages came back",
                due.len(),
                ages.len()
            ));
        }
        for (at, age) in due.iter().zip(ages) {
            tracks[*at].scored(age);
        }
    }

    // Every face still on the picture: the ones seen this frame at the box
    // this frame put them at, the ones the detector lost where they were.
    for (at, det) in &seen {
        let track = &mut tracks[*at];
        track.b = det.b;
        track.conf = det.conf;
        track.missed = 0;
    }
    for (t, track) in tracks.iter_mut().enumerate() {
        if !matched[t] {
            track.missed += 1;
        }
        track.since = track.since.saturating_add(1);
    }
    // Past `hold` frames unmatched a face is gone. With `hold` 0 a face the
    // detector did not find this frame is gone at once.
    tracks.retain(|track| track.missed <= params.hold);
    Ok(tracks.iter().map(Track::row).collect())
}

/// The ages out of what one call returned: the CORAL head picked out by shape,
/// and each row of it summed.
fn decode(returned: &[Returned], count: usize) -> Result<Vec<f64>, String> {
    let shapes: Vec<Vec<u32>> = returned.iter().map(|r| r.dimensions.clone()).collect();
    let logits = &returned[age_output(&shapes)?].values;
    if logits.len() != count * THRESHOLDS {
        return Err(format!(
            "ages: {count} crops went to the graph and {} age logits came back",
            logits.len()
        ));
    }
    Ok(logits.chunks(THRESHOLDS).map(coral_age).collect())
}

/// The faces being followed, and what naming a new one needs.
#[derive(Default)]
struct Faces {
    tracks: Vec<Track>,
    /// How many faces have been first seen so far.
    begun: u64,
    /// This frame's time, in seconds.
    now: f64,
}

struct Ages {
    v: u32,
    boxes: u32,
    width: usize,
    height: usize,
    params: Params,
    /// Where `v` is held.
    #[cfg(feature = "gpu")]
    memory: gpu::Memory,
    faces: Faces,
    /// Whether a frame's crops still go through as one batched tensor. The
    /// model's batch dimension is dynamic; a host that will not carry one
    /// turns this off for the rest of the run.
    batched: bool,
    model: Model,
}

#[cfg(not(feature = "gpu"))]
impl Ages {
    /// This frame's crops scored. The model's batch dimension is dynamic, so
    /// a frame's faces go through as one tensor; a host that refuses that is
    /// asked one crop at a time from there on.
    fn score(&mut self, frame: &[u8], boxes: &[FaceBox]) -> Result<Vec<f64>, String> {
        use faceage_common::SIDE;

        let (width, height) = (self.width, self.height);
        let rects: Vec<Rect> = boxes
            .iter()
            .filter_map(|b| crop(*b, width, height))
            .collect();
        let batch = ffrwd_frame::tensors(
            &Rgba::new(frame, width, height)?,
            &rects,
            SIDE,
            SIDE,
            Filter::Bicubic,
            IMAGENET,
        );
        let side = SIDE as u32;

        if rects.len() > 1 && self.batched {
            let tried = self
                .model
                .run(&[rects.len() as u32, 3, side, side], &batch)
                .and_then(|returned| decode(&returned, rects.len()));
            match tried {
                Ok(ages) => return Ok(ages),
                Err(why) => {
                    self.batched = false;
                    eprintln!(
                        "ages: a batch of {} crops was refused ({why}); \
                         one crop per call for the rest of this run",
                        rects.len()
                    );
                }
            }
        }

        // One crop's own bytes out of the batch: fp32, three planes of the square.
        let mut ages = Vec::with_capacity(rects.len());
        for one in batch.chunks(3 * SIDE * SIDE * 4) {
            let returned = self.model.run(&[1, 3, side, side], one)?;
            ages.push(decode(&returned, 1)?[0]);
        }
        Ok(ages)
    }
}

#[cfg(feature = "gpu")]
impl Ages {
    /// This frame's crops scored, the host making them into the tensor
    /// (`ffrwd:gpu/nn.crops-tensor`) from frame `index` of `v` wherever the
    /// frame is held. As on the host build, a frame's faces go through as
    /// one tensor, and a host that refuses that is asked one crop at a time
    /// from there on.
    fn score(&mut self, index: u32, boxes: &[FaceBox]) -> Result<Vec<f64>, String> {
        let (width, height) = (self.width, self.height);
        let rects: Vec<gpu::Crop> = boxes
            .iter()
            .filter_map(|b| crop(*b, width, height))
            .map(gpu::crop_rect)
            .collect();
        let options = gpu::options(Self::NAME);
        let v = self.v;

        if rects.len() > 1 && self.batched {
            let tried = self
                .model
                .run_tensor(|| gpu::crops(v, index, &rects, &options))
                .and_then(|returned| decode(&returned, rects.len()));
            match tried {
                Ok(ages) => return Ok(ages),
                Err(why) => {
                    self.batched = false;
                    eprintln!(
                        "ages: a batch of {} crops was refused ({why}); \
                         one crop per call for the rest of this run",
                        rects.len()
                    );
                }
            }
        }

        let mut ages = Vec::with_capacity(rects.len());
        for one in &rects {
            let returned = self
                .model
                .run_tensor(|| gpu::crops(v, index, std::slice::from_ref(one), &options))?;
            ages.push(decode(&returned, 1)?[0]);
        }
        Ok(ages)
    }
}

/// The tracker's params out of the node's.
#[cfg(not(feature = "gpu"))]
fn tracking(params: &Params) -> Params {
    *params
}

#[cfg(feature = "gpu")]
fn tracking(settings: &gpu::Settings) -> Params {
    settings.track
}

/// The formats `v` takes: `rgba`, cropped here.
#[cfg(not(feature = "gpu"))]
fn pixel_formats(_: &Params) -> &'static [&'static str] {
    &["rgba"]
}

#[cfg(feature = "gpu")]
use gpu::pixel_formats;

impl Node for Ages {
    const NAME: &'static str = "ages";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    #[cfg(not(feature = "gpu"))]
    type Params = Params;
    #[cfg(feature = "gpu")]
    type Params = gpu::Settings;

    fn shape(params: &Self::Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(
                Input::video("v")
                    .clock()
                    .pixel_formats(pixel_formats(params)),
            )
            .input(Input::rows("boxes").schema_json(BOXES_SCHEMA))
            .output(Output::rows("faces").schema_json(FACES_SCHEMA)))
    }

    fn init(params: Self::Params, init: &Init) -> Result<Ages> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("ages reads frames, and `v` is not video")?;
        Ok(Ages {
            v: v.id,
            boxes: init.stream("boxes")?.id,
            width: video.width as usize,
            height: video.height as usize,
            params: tracking(&params),
            #[cfg(feature = "gpu")]
            memory: params.memory,
            faces: Faces::default(),
            batched: true,
            model: Model::load(Self::NAME, Self::NAME, INPUT_NAME)?,
        })
    }

    fn set_params(&mut self, params: Self::Params) -> Result<()> {
        // Where `v` is held is the shape's, settled at init.
        #[cfg(feature = "gpu")]
        if params.memory != self.memory {
            return Err("ages cannot move `v` between host and gpu memory while it runs".into());
        }
        self.params = tracking(&params);
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        // A box with nothing of it on the picture cannot be cropped, so it
        // never becomes a face to follow.
        let (width, height) = (self.width, self.height);
        let dets: Vec<Det> = faces_of(tick.rows(self.boxes)?)
            .into_iter()
            .filter(|det| crop(det.b, width, height).is_some())
            .collect();
        self.faces.now = tick.time_base().seconds(frame.pts);
        let mut faces = std::mem::take(&mut self.faces);
        #[cfg(not(feature = "gpu"))]
        let rows = advance(&mut faces, &dets, self.params, |boxes| {
            self.score(&tick.fetch(self.v, frame.index), boxes)
        });
        // The host makes the crops from the frame wherever it is held, so the
        // frame is never fetched here.
        #[cfg(feature = "gpu")]
        let rows = advance(&mut faces, &dets, self.params, |boxes| {
            self.score(frame.index, boxes)
        });
        self.faces = faces;
        Ok(out.rows("faces", frame.pts, &rows?)?)
    }
}

ffrwd_node::export!(Ages);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::{read_params, Kind, Runner};

    fn det(x: f64, y: f64, w: f64, h: f64) -> Det {
        Det {
            conf: 0.9,
            b: (x, y, w, h),
        }
    }

    /// A frame's rows, read the way the node reads them.
    fn rows(items: &[&str]) -> Vec<InRow> {
        items
            .iter()
            .map(|row| ffrwd_node::parse(row).unwrap())
            .collect()
    }

    /// A frame's worth of scoring that answers the ages it was built with, and
    /// records how many crops it was asked for.
    fn answers(list: &[f64]) -> impl Fn(&[FaceBox]) -> Result<Vec<f64>, String> + '_ {
        move |boxes| {
            assert!(
                boxes.len() <= list.len(),
                "the frame asked for {} crops and the fixture has {}",
                boxes.len(),
                list.len()
            );
            Ok(list[..boxes.len()].to_vec())
        }
    }

    /// What one row says, as the value of one of its keys.
    fn field(row: &Row, key: &str) -> f64 {
        let parsed = serde_json::to_value(row).expect("a row");
        parsed[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{key} in {parsed}"))
    }

    #[test]
    fn a_frames_rows_become_the_boxes_to_score() {
        let dets = faces_of(rows(&[
            r#"{"class":"face","conf":0.93,"x":222,"y":62,"w":417,"h":500}"#,
            r#"{"class":"face","conf":0.4,"x":10,"y":10,"w":20,"h":30,"extra":"ignored"}"#,
        ]));
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].b, (222.0, 62.0, 417.0, 500.0));
        assert_eq!(dets[0].conf, 0.93);
        assert_eq!(dets[1].b, (10.0, 10.0, 20.0, 30.0));
    }

    #[test]
    fn rows_that_are_not_faces_are_skipped_rather_than_refused() {
        let dets = faces_of(rows(&[
            r#"{"class":"person","conf":0.9,"x":10,"y":10,"w":20,"h":20}"#,
            r#"{"class":"face","conf":0.9,"x":1,"y":2,"w":3,"h":4}"#,
            r#"{"class":"face","conf":0.9,"x":10,"y":10,"w":0,"h":20}"#,
        ]));
        assert_eq!(dets.len(), 1, "the one face row, and nothing else");
        assert_eq!(dets[0].b, (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn a_box_matches_the_face_it_overlaps_most_one_to_one() {
        let mut faces = Faces::default();
        let first = [det(0.0, 0.0, 100.0, 100.0), det(200.0, 0.0, 100.0, 100.0)];
        advance(
            &mut faces,
            &first,
            Params::default(),
            answers(&[10.0, 40.0]),
        )
        .unwrap();
        assert_eq!(faces.tracks.len(), 2);

        // Both faces moved a little, and the rows arrive in the other order.
        let moved = [det(205.0, 4.0, 100.0, 100.0), det(6.0, 3.0, 100.0, 100.0)];
        let of_det = match_boxes(&faces.tracks, &moved);
        assert_eq!(of_det, vec![Some(1), Some(0)]);
    }

    #[test]
    fn a_box_overlapping_nothing_enough_starts_a_new_face() {
        let mut faces = Faces::default();
        advance(
            &mut faces,
            &[det(0.0, 0.0, 100.0, 100.0)],
            Params::default(),
            answers(&[10.0]),
        )
        .unwrap();
        // A quarter overlap is 1/7 of the union, under the 0.3 it takes.
        let of_det = match_boxes(&faces.tracks, &[det(50.0, 50.0, 100.0, 100.0)]);
        assert_eq!(of_det, vec![None]);
        // Half of each is a third of the union, over it.
        let of_det = match_boxes(&faces.tracks, &[det(50.0, 0.0, 100.0, 100.0)]);
        assert_eq!(of_det, vec![Some(0)]);
    }

    #[test]
    fn two_boxes_over_one_face_leave_the_second_a_new_face() {
        let mut faces = Faces::default();
        advance(
            &mut faces,
            &[det(0.0, 0.0, 100.0, 100.0)],
            Params::default(),
            answers(&[10.0]),
        )
        .unwrap();
        let of_det = match_boxes(
            &faces.tracks,
            &[det(4.0, 0.0, 100.0, 100.0), det(8.0, 0.0, 100.0, 100.0)],
        );
        assert_eq!(of_det, vec![Some(0), None], "the closer box takes the face");
    }

    #[test]
    fn a_faces_age_is_the_running_mean_of_its_scores() {
        let mut faces = Faces::default();
        let one = [det(0.0, 0.0, 100.0, 100.0)];
        for age in [10.0, 20.0, 30.0] {
            let out = advance(&mut faces, &one, Params::default(), answers(&[age])).unwrap();
            assert_eq!(out.len(), 1);
        }
        // 10, 20 and 30 scored, so twenty.
        assert!((faces.tracks[0].age() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn the_running_mean_forgets_past_thirty_scores() {
        let mut track = Track::new(det(0.0, 0.0, 10.0, 10.0), 0.0, 0);
        for _ in 0..MEMORY {
            track.scored(10.0);
        }
        assert_eq!(track.scores.len(), MEMORY);
        assert!((track.age() - 10.0).abs() < 1e-9);
        for _ in 0..MEMORY {
            track.scored(40.0);
        }
        assert_eq!(track.scores.len(), MEMORY, "thirty and no more");
        assert!(
            (track.age() - 40.0).abs() < 1e-9,
            "the first thirty are gone"
        );
    }

    #[test]
    fn every_is_how_many_frames_apart_a_face_is_scored() {
        for every in [1u32, 2, 5] {
            let mut faces = Faces::default();
            let one = [det(0.0, 0.0, 100.0, 100.0)];
            let mut scored = 0;
            for _ in 0..20 {
                advance(&mut faces, &one, Params { hold: 5, every }, |boxes| {
                    scored += boxes.len();
                    Ok(vec![25.0; boxes.len()])
                })
                .unwrap();
            }
            // Twenty frames, the first one always scored and then one every
            // `every` after it.
            let want = 1 + (20 - 1) / every as usize;
            assert_eq!(scored, want, "every {every}");
        }
    }

    #[test]
    fn a_face_between_scores_still_reports_its_age_at_this_frames_box() {
        let mut faces = Faces::default();
        let params = Params { hold: 5, every: 5 };
        advance(
            &mut faces,
            &[det(0.0, 0.0, 100.0, 100.0)],
            params,
            answers(&[12.0]),
        )
        .unwrap();
        let out = advance(&mut faces, &[det(20.0, 0.0, 100.0, 100.0)], params, |_| {
            panic!("nothing is due on the second frame")
        })
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(field(&out[0], "x"), 20.0, "the box is this frame's");
        assert_eq!(field(&out[0], "age"), 12.0, "the age is the one it has");
    }

    #[test]
    fn a_face_the_detector_lost_is_held_and_then_dropped() {
        let mut faces = Faces::default();
        let params = Params { hold: 2, every: 1 };
        let one = [det(10.0, 20.0, 100.0, 100.0)];
        advance(&mut faces, &one, params, answers(&[30.0])).unwrap();

        for frame in 1..=2 {
            let out = advance(&mut faces, &[], params, answers(&[])).unwrap();
            assert_eq!(out.len(), 1, "frame {frame} is still holding it");
            assert_eq!(field(&out[0], "x"), 10.0, "where it was last seen");
            assert_eq!(field(&out[0], "age"), 30.0);
            assert_eq!(field(&out[0], "conf"), 0.9);
        }
        let out = advance(&mut faces, &[], params, answers(&[])).unwrap();
        assert!(out.is_empty(), "past hold it is gone");
        assert!(faces.tracks.is_empty());
    }

    #[test]
    fn hold_zero_carries_nothing() {
        let mut faces = Faces::default();
        let params = Params { hold: 0, every: 1 };
        advance(
            &mut faces,
            &[det(10.0, 20.0, 100.0, 100.0)],
            params,
            answers(&[30.0]),
        )
        .unwrap();
        let out = advance(&mut faces, &[], params, answers(&[])).unwrap();
        assert!(out.is_empty(), "the frame it is not found it is gone");
        assert!(faces.tracks.is_empty());
    }

    #[test]
    fn a_face_found_again_inside_hold_keeps_the_age_it_had() {
        let mut faces = Faces::default();
        let params = Params { hold: 3, every: 1 };
        let one = [det(10.0, 20.0, 100.0, 100.0)];
        advance(&mut faces, &one, params, answers(&[30.0])).unwrap();
        advance(&mut faces, &[], params, answers(&[])).unwrap();
        let out = advance(&mut faces, &one, params, answers(&[40.0])).unwrap();
        assert_eq!(faces.tracks.len(), 1, "the same face, not a new one");
        // 30 and 40 scored, so 35.
        assert_eq!(field(&out[0], "age"), 35.0);
    }

    #[test]
    fn a_row_carries_the_class_the_box_and_the_age_to_one_decimal() {
        let mut track = Track::new(
            Det {
                conf: 0.87941,
                b: (222.0, 62.0, 417.0, 500.0),
            },
            1.5,
            3,
        );
        track.scored(11.111);
        track.scored(12.222);
        assert_eq!(
            serde_json::to_string(&track.row()).unwrap(),
            r#"{"class":"face","conf":0.8794,"x":222,"y":62,"w":417,"h":500,"age":11.7,"start_t":1.5,"id":3}"#
        );
    }

    #[test]
    fn a_frame_with_no_faces_at_all_writes_no_rows() {
        let mut faces = Faces::default();
        let out = advance(&mut faces, &[], Params::default(), answers(&[])).unwrap();
        assert!(out.is_empty());
    }

    fn parse_params(params: &str) -> Result<Params, String> {
        read_params::<Params>(PARAMS_SCHEMA, params).map(|(params, _)| params)
    }

    #[test]
    fn params_default_to_what_the_schema_publishes() {
        assert_eq!(
            parse_params("").expect("empty is the defaults"),
            Params::default()
        );
        assert_eq!(
            parse_params("{}").expect("and so is an empty object"),
            Params::default()
        );
        assert_eq!(
            parse_params(r#"{"hold":0,"every":10}"#).expect("both given"),
            Params { hold: 0, every: 10 }
        );
        assert_eq!(
            parse_params(r#"{"hold":12.0}"#).expect("a whole number spelled as one"),
            Params { hold: 12, every: 1 }
        );
    }

    #[test]
    fn params_outside_the_schema_are_refused_by_name() {
        for bad in [
            r#"{"hold":-1}"#,
            r#"{"hold":601}"#,
            r#"{"every":0}"#,
            r#"{"every":601}"#,
            r#"{"every":2.5}"#,
        ] {
            let error = parse_params(bad).expect_err(bad);
            assert!(
                error.contains(if bad.contains("hold") {
                    "hold"
                } else {
                    "every"
                }),
                "{error}"
            );
        }
        assert!(
            parse_params(r#"{"conf":0.3}"#).is_err(),
            "and so is a param this module has none of"
        );
    }

    #[test]
    fn a_face_is_named_by_when_it_was_first_seen_and_how_many_came_before() {
        let mut faces = Faces::default();
        let params = Params { hold: 5, every: 1 };
        faces.now = 0.5;
        advance(
            &mut faces,
            &[det(0.0, 0.0, 100.0, 100.0)],
            params,
            answers(&[10.0]),
        )
        .unwrap();
        faces.now = 1.0;
        let both = [det(4.0, 0.0, 100.0, 100.0), det(300.0, 0.0, 100.0, 100.0)];
        let out = advance(&mut faces, &both, params, answers(&[20.0, 30.0])).unwrap();
        let named: Vec<(f64, f64)> = out
            .iter()
            .map(|row| (field(row, "start_t"), field(row, "id")))
            .collect();
        assert_eq!(named, [(0.5, 0.0), (1.0, 1.0)]);
        faces.now = 1.5;
        let out = advance(&mut faces, &[], params, answers(&[])).unwrap();
        assert_eq!(
            (field(&out[1], "start_t"), field(&out[1], "id")),
            (1.0, 1.0),
            "a held face keeps its name"
        );
    }

    #[test]
    fn the_faces_leave_as_rows_alone_and_the_tracker_is_one_instance() {
        let bound = ["v".to_owned(), "boxes".to_owned()];
        let shape = Runner::<Ages>::shape("", &bound).expect("a shape");
        assert_eq!(shape.clock_input(), Some("v"));
        assert_eq!(shape.inputs[0].accepts.pixel_formats, ["rgba"]);
        let boxes = shape.find_input("boxes").expect("a rows input");
        assert_eq!(boxes.schema.as_deref(), Some(BOXES_SCHEMA));
        assert_eq!(shape.outputs.len(), 1, "no picture leaves");
        assert_eq!(shape.outputs[0].name, "faces");
        assert_eq!(shape.outputs[0].kind, Kind::Data);
        assert!(
            !shape.pure,
            "a face is followed from the frame before, so the stream is never shared out"
        );
    }

    #[test]
    fn the_age_head_is_told_from_the_gender_head_by_shape_in_either_order() {
        assert_eq!(age_output(&[vec![1, 100], vec![1, 2]]).expect("found"), 0);
        assert_eq!(age_output(&[vec![4, 2], vec![4, 100]]).expect("found"), 1);
        let error = age_output(&[vec![1, 2]]).expect_err("no age head");
        assert!(error.starts_with("ages: "), "{error}");
    }

    #[test]
    fn the_coral_thresholds_of_a_batch_decode_one_age_per_crop() {
        // Two crops: one past twenty thresholds, one past sixty.
        let mut logits = vec![-20.0f32; 2 * THRESHOLDS];
        logits[..20].fill(20.0);
        logits[THRESHOLDS..THRESHOLDS + 60].fill(20.0);
        let ages: Vec<f64> = logits.chunks(THRESHOLDS).map(coral_age).collect();
        assert!((ages[0] - 20.0).abs() < 1e-4);
        assert!((ages[1] - 60.0).abs() < 1e-4);
    }
}
