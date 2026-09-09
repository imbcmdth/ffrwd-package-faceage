//! What the faceage modules share: the box math the tracker follows faces
//! with, and the CORAL head the age comes out of. Nothing here touches
//! `wasi:nn` or the wit bindings, so a module compiles it in as plain Rust and
//! its tests run on the host.
//!
//! Turning a frame into a tensor is not here: `ages` asks the host for `rgba`
//! and `ffrwd-frame` crops, resizes and normalizes it.

/// The square the age model takes its crops at.
pub const SIDE: usize = 224;

/// How much of a box's own width and height is added on each side before the
/// crop is taken. The model was trained on crops padded this far, and the
/// source card reports its error growing from 3.56 to 3.76 years without it.
pub const PAD: f64 = 0.10;

/// How much two boxes - each `x, y, w, h` - overlap, over how much they cover
/// together. 0 when either is degenerate or they do not touch.
pub fn iou(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> f64 {
    let (ax0, ay0, aw, ah) = a;
    let (bx0, by0, bw, bh) = b;
    if aw <= 0.0 || ah <= 0.0 || bw <= 0.0 || bh <= 0.0 {
        return 0.0;
    }
    let overlap = |a0: f64, a1: f64, b0: f64, b1: f64| (a1.min(b1) - a0.max(b0)).max(0.0);
    let iw = overlap(ax0, ax0 + aw, bx0, bx0 + bw);
    let ih = overlap(ay0, ay0 + ah, by0, by0 + bh);
    let inter = iw * ih;
    let union = aw * ah + bw * bh - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// The logistic function. The age head is CORAL - a hundred ordinal
/// thresholds - so the age is the sum of these.
pub fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

/// The age in years the CORAL head's logits decode to: how many of its
/// hundred ordinal thresholds the face is past, counted softly.
pub fn coral_age(logits: &[f32]) -> f64 {
    logits.iter().map(|logit| f64::from(sigmoid(*logit))).sum()
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
pub fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_is_the_intersection_over_the_union() {
        let a = (0.0, 0.0, 10.0, 10.0);
        assert!((iou(a, a) - 1.0).abs() < 1e-12, "a box against itself");
        // Half of each box overlaps: 50 over 150.
        let half = iou(a, (5.0, 0.0, 10.0, 10.0));
        assert!((half - 1.0 / 3.0).abs() < 1e-12, "{half}");
        assert_eq!(iou(a, (20.0, 20.0, 10.0, 10.0)), 0.0, "no overlap at all");
        assert_eq!(iou(a, (0.0, 0.0, 0.0, 10.0)), 0.0, "a degenerate box");
    }

    #[test]
    fn the_coral_head_counts_the_thresholds_the_face_is_past() {
        // Twenty thresholds well past, eighty well short: twenty years.
        let mut logits = vec![-20.0f32; 100];
        logits[..20].fill(20.0);
        assert!((coral_age(&logits) - 20.0).abs() < 1e-4);
        // A threshold sitting on the fence is half a year.
        logits[20] = 0.0;
        assert!((coral_age(&logits) - 20.5).abs() < 1e-4);
        assert!(coral_age(&[-30.0; 100]) < 0.001, "a newborn");
        assert!((coral_age(&[30.0; 100]) - 100.0).abs() < 0.001, "and the top");
    }

    #[test]
    fn a_tensors_floats_come_back_out_of_its_bytes() {
        let values = [1.5f32, -2.25, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(le_f32s(&bytes), values);
        // A trailing part-word is not a float and is not read.
        assert_eq!(le_f32s(&bytes[..bytes.len() - 1]), values[..2]);
    }
}
