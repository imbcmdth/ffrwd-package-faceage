//! Ages on the faces a detector already found: the frame passes through
//! untouched, and the boxes riding it come back with an age on each. Every box
//! is cropped with the ten percent padding the model was trained on, resized
//! into its 224 square with Pillow's bicubic, and run through the graph; the
//! CORAL head's hundred thresholds sum to an age in years.
//!
//! A face is not scored afresh every frame. Boxes are followed from frame to
//! frame by overlap, and a face carries the running mean of its own last
//! thirty scores, so the number a row reports settles rather than jitters.
//! `every` is how many frames apart a face is scored - the work is one
//! inference per face per `every` frames - and `hold` keeps a face the
//! detector lost, re-emitted where it was last seen, so what reads these rows
//! does not flicker for the frame a head turns.
//!
//! The graph is FaceAge ClientScan, a DINOv3 backbone with a CORAL age head,
//! run through `wasi:nn`. The module never opens a file - the host binds the
//! graph to a name with `-nn ages=<path>` and this module asks for that name
//! and nothing else. The gender head the graph also carries is not read.
//!
//! The frame arrives as `rgba`, so the colour was settled upstream by ffmpeg,
//! which reads the stream's own range and matrix rather than guessing at them;
//! `ffrwd-frame` takes it from there. That the range is right is not
//! fussiness: a limited-range frame taken for a full-range one is a face
//! flattened by a seventh, and a flatter face reads as a younger one - on the
//! check clip that alone was worth two years of mean error and put a face of
//! 66 at 39.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:faceage-ages/ages",
    generate_all,
});

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use faceage_common::{coral_age, iou, le_f32s, PAD, SIDE};
use ffrwd_frame::{Filter, Rect, Rgba, IMAGENET};
use serde::{Deserialize, Serialize};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn ages=<path>`.
const MODEL: &str = "ages";

/// What the export calls its input tensor.
const INPUT_NAME: &str = "face";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

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

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"hold":{"type":"integer","minimum":0,"maximum":600,"default":5},"every":{"type":"integer","minimum":1,"maximum":600,"default":1}},"additionalProperties":false}"#;

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number"},"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"},"age":{"type":"number"}},"required":["class","conf","x","y","w","h","age"],"additionalProperties":false}"#;

fn default_hold() -> f64 {
    5.0
}

fn default_every() -> f64 {
    1.0
}

/// The params as they arrive: whole numbers, but read as numbers, since the
/// declaration spells them `number` and a host may hand `5` over as `5.0`.
#[derive(Clone, Copy, Debug, Deserialize)]
// The schema says these two and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Given {
    #[serde(default = "default_hold")]
    hold: f64,
    #[serde(default = "default_every")]
    every: f64,
}

/// The params settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Params {
    hold: u32,
    every: u32,
}

impl Default for Params {
    fn default() -> Self {
        Params { hold: 5, every: 1 }
    }
}

/// One row this module writes: the box it was handed, with the age of the face
/// in it.
#[derive(Serialize)]
struct Row {
    class: String,
    conf: f64,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
    age: f64,
}

/// One row this module reads. Extra keys are ignored; a row without the four
/// coordinates is not a box and never parses.
#[derive(Deserialize)]
struct InRow {
    class: Option<String>,
    #[serde(default)]
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
    fn new(det: Det) -> Track {
        Track {
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

    fn row(&self) -> String {
        serde_json::to_string(&Row {
            class: FACE.to_string(),
            // To four places: the detector's own precision is nowhere near the
            // sixteen digits an f32 widened to an f64 prints.
            conf: (self.conf * 10_000.0).round() / 10_000.0,
            x: self.b.0.round() as i64,
            y: self.b.1.round() as i64,
            w: self.b.2.round() as i64,
            h: self.b.3.round() as i64,
            age: self.age(),
        })
        .expect("row serializes")
    }
}

/// What `init` settled, plus the graph it loaded and what the run has learned
/// about it.
struct Opened {
    width: usize,
    height: usize,
    params: Params,
    /// What the graph calls its input, settled by the first call that works.
    input_name: Cell<&'static str>,
    /// Whether a frame's crops still go through as one batched tensor. The
    /// model's batch dimension is dynamic; a host that will not carry one
    /// turns this off for the rest of the run.
    batched: Cell<bool>,
    /// The faces being followed.
    tracks: RefCell<Vec<Track>>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let given: Given = if trimmed.is_empty() {
        Given {
            hold: default_hold(),
            every: default_every(),
        }
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("ages cannot read its params: {e}"))?
    };
    let whole = |name: &str, value: f64, low: f64, high: f64| -> Result<u32, String> {
        if !value.is_finite() || !(low..=high).contains(&value) || value.fract() != 0.0 {
            return Err(format!(
                "ages needs {name} a whole number between {low} and {high}, got {value}"
            ));
        }
        Ok(value as u32)
    };
    Ok(Params {
        hold: whole("hold", given.hold, 0.0, 600.0)?,
        every: whole("every", given.every, 1.0, 600.0)?,
    })
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("ages: {what}: {code} ({})", error.data())
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

/// The rows of a frame as the boxes to score: what is not an object with four
/// coordinates is not a box, and what is not a face is not this module's.
fn read_rows(rows: &[String]) -> Vec<Det> {
    let mut dets = Vec::new();
    for row in rows {
        let Ok(parsed) = serde_json::from_str::<InRow>(row) else {
            continue;
        };
        if parsed.class.as_deref() != Some(FACE) {
            continue;
        }
        if parsed.w <= 0.0 || parsed.h <= 0.0 {
            continue;
        }
        dets.push(Det {
            conf: parsed.conf,
            b: (parsed.x, parsed.y, parsed.w, parsed.h),
        });
    }
    dets
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
    tracks: &mut Vec<Track>,
    dets: &[Det],
    params: Params,
    score: impl FnOnce(&[FaceBox]) -> Result<Vec<f64>, String>,
) -> Result<Vec<String>, String> {
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
                tracks.push(Track::new(*det));
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

/// One tensor of `count` crops through the graph, however the graph names its
/// input.
fn compute(opened: &Opened, input: &[u8], count: usize) -> Result<Vec<(String, Tensor)>, String> {
    let dimensions = [count as u32, 3, SIDE as u32, SIDE as u32];
    let name = opened.input_name.get();
    let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
    match opened.context.compute(vec![(name.to_string(), tensor)]) {
        Ok(returned) => Ok(returned),
        // An export whose input is not called what this one calls it. The
        // host takes a position where it takes a name, so the retry names none,
        // and the name that worked is kept for every call after this one.
        Err(_) if name == INPUT_NAME => {
            opened.input_name.set(INPUT_INDEX);
            let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
            opened
                .context
                .compute(vec![(INPUT_INDEX.to_string(), tensor)])
                .map_err(|e| failed("compute", &e))
        }
        Err(e) => Err(failed("compute", &e)),
    }
}

/// The ages out of what one call returned: the CORAL head picked out by shape,
/// and each row of it summed.
fn decode(returned: Vec<(String, Tensor)>, count: usize) -> Result<Vec<f64>, String> {
    let tensors: Vec<Tensor> = returned.into_iter().map(|(_, tensor)| tensor).collect();
    let shapes: Vec<Vec<u32>> = tensors.iter().map(Tensor::dimensions).collect();
    let logits = le_f32s(&tensors[age_output(&shapes)?].data());
    if logits.len() != count * THRESHOLDS {
        return Err(format!(
            "ages: {count} crops went to the graph and {} age logits came back",
            logits.len()
        ));
    }
    Ok(logits.chunks(THRESHOLDS).map(coral_age).collect())
}

/// This frame's crops scored. The model's batch dimension is dynamic, so a
/// frame's faces go through as one tensor; a host that refuses that is asked
/// one crop at a time from there on.
fn score(opened: &Opened, frame: &[u8], boxes: &[FaceBox]) -> Result<Vec<f64>, String> {
    let (width, height) = (opened.width, opened.height);
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

    if rects.len() > 1 && opened.batched.get() {
        match compute(opened, &batch, rects.len()).and_then(|r| decode(r, rects.len())) {
            Ok(ages) => return Ok(ages),
            Err(why) => {
                opened.batched.set(false);
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
        let age = decode(compute(opened, one, 1)?, 1)?;
        ages.push(age[0]);
    }
    Ok(ages)
}

/// One frame's rows in, its rows out. The picture is only fetched when there
/// is something on it to score.
fn run(opened: &Opened, window: &InWindow, i: u32) -> Result<Vec<String>, String> {
    // A box with nothing of it on the picture cannot be cropped, so it never
    // becomes a face to follow.
    let dets: Vec<Det> = read_rows(&window.rows(i))
        .into_iter()
        .filter(|det| crop(det.b, opened.width, opened.height).is_some())
        .collect();
    let mut tracks = opened.tracks.borrow_mut();
    advance(&mut tracks, &dets, opened.params, |boxes| {
        score(opened, &window.fetch(i), boxes)
    })
}

struct Ages;

impl Guest for Ages {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "ages".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // A call does NOT depend only on what it was handed: a face is
            // followed from the frame before, and its age is the mean of what
            // it has scored since. Declaring this pure lets the host share the
            // stream out among parallel instances, and then each instance sees
            // every second or fourth frame - on which two people standing in
            // the same place ARE one face by overlap, and their ages are
            // averaged together. Measured on the 60-face check clip: 17 of 60
            // rows came back carrying two to four faces' scores, and the run
            // was not repeatable. False is what makes the host host this
            // serially, which is the only way a tracker is right.
            pure: false,
            one_to_one: true,
            // The boxes to score arrive as the rows riding each frame.
            reads_rows: true,
            // And what leaves is this module's own: the same faces, aged.
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("ages reads frames, and this stream is audio".to_string());
        };
        if video.pix_fmt != "rgba" {
            return Err(format!(
                "ages does not accept pixel format {}",
                video.pix_fmt
            ));
        }
        let parsed = parse_params(&params)?;

        // The graph is loaded once per instance, and the session built once:
        // the first crop is what a provider picks its kernels on, and every
        // one after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                params: parsed,
                input_name: Cell::new(INPUT_NAME),
                batched: Cell::new(true),
                tracks: RefCell::new(Vec::new()),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.params = parsed;
            }
        });
        Ok(())
    }

    fn process(window: &InWindow, _trailing: Vec<String>, _last: bool) -> Processed {
        // The final call carries nothing: window and stride are 1, so no frame
        // is ever left over.
        let mut out = Vec::with_capacity(window.len() as usize);
        OPENED.with(|opened| {
            let borrowed = opened.borrow();
            let opened = borrowed
                .as_ref()
                .expect("init loads the graph before any frame arrives");
            for i in 0..window.len() {
                // `process` has no way to say no, so a graph that failed
                // mid-stream stops the run rather than dropping faces on the
                // floor.
                let rows = run(opened, window, i).unwrap_or_else(|m| panic!("{m}"));
                out.push(OutFrame {
                    pts: window.pts(i),
                    // The picture leaves untouched; the rows are the work.
                    frame: FramePayload::Same,
                    rows,
                });
            }
        });
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(Ages);

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f64, y: f64, w: f64, h: f64) -> Det {
        Det {
            conf: 0.9,
            b: (x, y, w, h),
        }
    }

    fn rows(items: &[&str]) -> Vec<String> {
        items.iter().map(|r| r.to_string()).collect()
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
    fn field(row: &str, key: &str) -> f64 {
        let parsed: serde_json::Value = serde_json::from_str(row).expect("a row");
        parsed[key].as_f64().unwrap_or_else(|| panic!("{key} in {row}"))
    }

    #[test]
    fn a_frames_rows_become_the_boxes_to_score() {
        let dets = read_rows(&rows(&[
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
        let dets = read_rows(&rows(&[
            r#"{"shot":4}"#,
            "not json at all",
            r#"{"class":"person","conf":0.9,"x":10,"y":10,"w":20,"h":20}"#,
            r#"{"conf":0.9,"x":10,"y":10,"w":20,"h":20}"#,
            r#"{"class":"face","conf":0.9,"x":1,"y":2,"w":3,"h":4}"#,
            r#"{"class":"face","conf":0.9,"x":10,"y":10,"w":0,"h":20}"#,
        ]));
        assert_eq!(dets.len(), 1, "the one face row, and nothing else");
        assert_eq!(dets[0].b, (1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn a_box_matches_the_face_it_overlaps_most_one_to_one() {
        let mut tracks = vec![];
        let first = [det(0.0, 0.0, 100.0, 100.0), det(200.0, 0.0, 100.0, 100.0)];
        advance(&mut tracks, &first, Params::default(), answers(&[10.0, 40.0])).unwrap();
        assert_eq!(tracks.len(), 2);

        // Both faces moved a little, and the rows arrive in the other order.
        let moved = [det(205.0, 4.0, 100.0, 100.0), det(6.0, 3.0, 100.0, 100.0)];
        let of_det = match_boxes(&tracks, &moved);
        assert_eq!(of_det, vec![Some(1), Some(0)]);
    }

    #[test]
    fn a_box_overlapping_nothing_enough_starts_a_new_face() {
        let mut tracks = vec![];
        advance(
            &mut tracks,
            &[det(0.0, 0.0, 100.0, 100.0)],
            Params::default(),
            answers(&[10.0]),
        )
        .unwrap();
        // A quarter overlap is 1/7 of the union, under the 0.3 it takes.
        let of_det = match_boxes(&tracks, &[det(50.0, 50.0, 100.0, 100.0)]);
        assert_eq!(of_det, vec![None]);
        // Half of each is a third of the union, over it.
        let of_det = match_boxes(&tracks, &[det(50.0, 0.0, 100.0, 100.0)]);
        assert_eq!(of_det, vec![Some(0)]);
    }

    #[test]
    fn two_boxes_over_one_face_leave_the_second_a_new_face() {
        let mut tracks = vec![];
        advance(
            &mut tracks,
            &[det(0.0, 0.0, 100.0, 100.0)],
            Params::default(),
            answers(&[10.0]),
        )
        .unwrap();
        let of_det = match_boxes(&tracks, &[det(4.0, 0.0, 100.0, 100.0), det(8.0, 0.0, 100.0, 100.0)]);
        assert_eq!(of_det, vec![Some(0), None], "the closer box takes the face");
    }

    #[test]
    fn a_faces_age_is_the_running_mean_of_its_scores() {
        let mut tracks = vec![];
        let one = [det(0.0, 0.0, 100.0, 100.0)];
        for age in [10.0, 20.0, 30.0] {
            let out = advance(&mut tracks, &one, Params::default(), answers(&[age])).unwrap();
            assert_eq!(out.len(), 1);
        }
        // 10, 20 and 30 scored, so twenty.
        assert!((tracks[0].age() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn the_running_mean_forgets_past_thirty_scores() {
        let mut track = Track::new(det(0.0, 0.0, 10.0, 10.0));
        for _ in 0..MEMORY {
            track.scored(10.0);
        }
        assert_eq!(track.scores.len(), MEMORY);
        assert!((track.age() - 10.0).abs() < 1e-9);
        for _ in 0..MEMORY {
            track.scored(40.0);
        }
        assert_eq!(track.scores.len(), MEMORY, "thirty and no more");
        assert!((track.age() - 40.0).abs() < 1e-9, "the first thirty are gone");
    }

    #[test]
    fn every_is_how_many_frames_apart_a_face_is_scored() {
        for every in [1u32, 2, 5] {
            let mut tracks = vec![];
            let one = [det(0.0, 0.0, 100.0, 100.0)];
            let mut scored = 0;
            for _ in 0..20 {
                advance(
                    &mut tracks,
                    &one,
                    Params { hold: 5, every },
                    |boxes| {
                        scored += boxes.len();
                        Ok(vec![25.0; boxes.len()])
                    },
                )
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
        let mut tracks = vec![];
        let params = Params { hold: 5, every: 5 };
        advance(
            &mut tracks,
            &[det(0.0, 0.0, 100.0, 100.0)],
            params,
            answers(&[12.0]),
        )
        .unwrap();
        let out = advance(
            &mut tracks,
            &[det(20.0, 0.0, 100.0, 100.0)],
            params,
            |_| panic!("nothing is due on the second frame"),
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(field(&out[0], "x"), 20.0, "the box is this frame's");
        assert_eq!(field(&out[0], "age"), 12.0, "the age is the one it has");
    }

    #[test]
    fn a_face_the_detector_lost_is_held_and_then_dropped() {
        let mut tracks = vec![];
        let params = Params { hold: 2, every: 1 };
        let one = [det(10.0, 20.0, 100.0, 100.0)];
        advance(&mut tracks, &one, params, answers(&[30.0])).unwrap();

        for frame in 1..=2 {
            let out = advance(&mut tracks, &[], params, answers(&[])).unwrap();
            assert_eq!(out.len(), 1, "frame {frame} is still holding it");
            assert_eq!(field(&out[0], "x"), 10.0, "where it was last seen");
            assert_eq!(field(&out[0], "age"), 30.0);
            assert_eq!(field(&out[0], "conf"), 0.9);
        }
        let out = advance(&mut tracks, &[], params, answers(&[])).unwrap();
        assert!(out.is_empty(), "past hold it is gone");
        assert!(tracks.is_empty());
    }

    #[test]
    fn hold_zero_carries_nothing() {
        let mut tracks = vec![];
        let params = Params { hold: 0, every: 1 };
        advance(
            &mut tracks,
            &[det(10.0, 20.0, 100.0, 100.0)],
            params,
            answers(&[30.0]),
        )
        .unwrap();
        let out = advance(&mut tracks, &[], params, answers(&[])).unwrap();
        assert!(out.is_empty(), "the frame it is not found it is gone");
        assert!(tracks.is_empty());
    }

    #[test]
    fn a_face_found_again_inside_hold_keeps_the_age_it_had() {
        let mut tracks = vec![];
        let params = Params { hold: 3, every: 1 };
        let one = [det(10.0, 20.0, 100.0, 100.0)];
        advance(&mut tracks, &one, params, answers(&[30.0])).unwrap();
        advance(&mut tracks, &[], params, answers(&[])).unwrap();
        let out = advance(&mut tracks, &one, params, answers(&[40.0])).unwrap();
        assert_eq!(tracks.len(), 1, "the same face, not a new one");
        // 30 and 40 scored, so 35.
        assert_eq!(field(&out[0], "age"), 35.0);
    }

    #[test]
    fn a_row_carries_the_class_the_box_and_the_age_to_one_decimal() {
        let mut track = Track::new(Det {
            conf: 0.87941,
            b: (222.0, 62.0, 417.0, 500.0),
        });
        track.scored(11.111);
        track.scored(12.222);
        assert_eq!(
            track.row(),
            r#"{"class":"face","conf":0.8794,"x":222,"y":62,"w":417,"h":500,"age":11.7}"#
        );
    }

    #[test]
    fn a_frame_with_no_faces_at_all_writes_no_rows() {
        let mut tracks = vec![];
        let out = advance(&mut tracks, &[], Params::default(), answers(&[])).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn params_default_to_what_the_schema_publishes() {
        assert_eq!(parse_params("").expect("empty is the defaults"), Params::default());
        assert_eq!(parse_params("{}").expect("and so is an empty object"), Params::default());
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
            assert!(error.starts_with("ages "), "{error}");
        }
        assert!(
            parse_params(r#"{"conf":0.3}"#).is_err(),
            "and so is a param this module has none of"
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
