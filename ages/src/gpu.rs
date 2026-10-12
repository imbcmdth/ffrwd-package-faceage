//! The GPU build: a node beside `ffrwd:gpu`. The host makes a frame's due
//! crops into one tensor (`ffrwd:gpu/nn.crops-tensor`): the crops the host
//! build cuts, resized with the same Pillow bicubic and normalized the same
//! way, so the frame is never fetched.
//!
//! `memory` says where `v` is held. On `host`, the default, the frame arrives
//! as `rgba` and the host makes the tensor on the CPU. On `gpu`, the frame is
//! held on the device in any format the sidecar holds there, the tensor is
//! made on the device, and where the model's session reads the device's
//! memory it is bound to the session with no copy. Either way the age logits
//! come back to host memory, where the tracker reads them.

// `generate_all`: the world's interfaces come from other packages - ffrwd:gpu,
// wasi:webgpu and its deps - and without it bindgen expects them to have been
// generated somewhere else. wasi:nn is the bindings `faceage_common::nn`
// already generates, so the tensor `crops-tensor` answers is the one
// `Model` passes to `compute`. The path is the `ages` crate's, so the
// `ages-gpu` crate reaching this file finds it too.
mod bindings {
    wit_bindgen::generate!({
        path: "../ages/wit-gpu",
        world: "ffrwd:gpu/gpu-nn-module@0.1.0",
        with: {
            "wasi:nn/tensor@0.2.0-rc-2024-10-28": faceage_common::nn::wasi::nn::tensor,
            "wasi:nn/inference@0.2.0-rc-2024-10-28": faceage_common::nn::wasi::nn::inference,
            "wasi:nn/errors@0.2.0-rc-2024-10-28": faceage_common::nn::wasi::nn::errors,
        },
        generate_all,
    });
}

use bindings::exports::ffrwd::gpu::ports::Guest;
pub use bindings::ffrwd::gpu::nn::Rect as Crop;
use bindings::ffrwd::gpu::nn::{
    crops_tensor, ChannelOrder, Fit, FrameTensorOptions, ResizeFilter, TensorLayout,
};
use bindings::ffrwd::gpu::types::{FrameMemory, PortMemory};
use faceage_common::nn::wasi::nn::tensor::Tensor;
use faceage_common::SIDE;
use ffrwd_frame::{Rect, IMAGENET};
use serde::Deserialize;

use crate::{Params, PARAMS_SCHEMA};

/// Where `v`'s frames are held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Memory {
    #[default]
    Host,
    Gpu,
}

/// The GPU build's params: the tracker's, and where `v` is held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct Settings {
    #[serde(flatten)]
    pub track: Params,
    pub memory: Memory,
}

/// The formats `v` takes: any the sidecar holds on the device, or `rgba` in
/// host memory, which the host makes the tensor from on the CPU.
pub fn pixel_formats(settings: &Settings) -> &'static [&'static str] {
    match settings.memory {
        Memory::Gpu => &["rgba", "yuv420p", "yuv422p", "yuv444p"],
        Memory::Host => &["rgba"],
    }
}

/// How the host makes the crops into the graph's input: each stretched into
/// the 224 square with Pillow's bicubic, widened where it shrinks, planar RGB
/// in 0..1 normalized by ImageNet's mean and deviation - what the host build
/// makes with `ffrwd_frame::tensors(.., Filter::Bicubic, IMAGENET)`.
pub fn options(model: &str) -> FrameTensorOptions {
    let side = SIDE as u32;
    let [mr, mg, mb] = IMAGENET.mean;
    let [sr, sg, sb] = IMAGENET.std;
    FrameTensorOptions {
        model: model.to_string(),
        width: side,
        height: side,
        layout: TensorLayout::Nchw,
        order: ChannelOrder::Rgb,
        fit: Fit::Stretch,
        pad: 0.0,
        scale: 1.0,
        mean: (mr, mg, mb),
        std: (sr, sg, sb),
        filter: ResizeFilter::Bicubic,
        antialias: true,
    }
}

/// A crop as the host takes it: its left and top edges and its size.
pub fn crop_rect(rect: Rect) -> Crop {
    Crop {
        x: rect.x0 as u32,
        y: rect.y0 as u32,
        width: (rect.x1 - rect.x0) as u32,
        height: (rect.y1 - rect.y0) as u32,
    }
}

/// `crops` of frame `index` of stream `v` this tick, as one
/// `[N, 3, 224, 224]` tensor in the order given.
pub fn crops(
    v: u32,
    index: u32,
    crops: &[Crop],
    options: &FrameTensorOptions,
) -> Result<Tensor, String> {
    crops_tensor(v, index, crops, options).map_err(|e| format!("ages: {e}"))
}

struct Ports;

impl Guest for Ports {
    /// `v`, when `memory` is `gpu`. The faces leave as rows, which are not
    /// frames and are held nowhere.
    fn memory(params: String) -> Vec<PortMemory> {
        match ffrwd_node::read_params::<Settings>(PARAMS_SCHEMA, &params) {
            Ok((settings, _)) if settings.memory == Memory::Gpu => vec![PortMemory {
                port: "v".to_string(),
                memory: FrameMemory::Gpu,
            }],
            _ => Vec::new(),
        }
    }
}

bindings::export!(Ports with_types_in bindings);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::{read_params, Runner};

    fn settings(params: &str) -> Result<Settings, String> {
        read_params::<Settings>(PARAMS_SCHEMA, params).map(|(settings, _)| settings)
    }

    #[test]
    fn memory_defaults_to_host_beside_the_trackers_params() {
        assert_eq!(settings("").expect("the defaults"), Settings::default());
        assert_eq!(
            settings(r#"{"hold":2,"every":3,"memory":"gpu"}"#).expect("all three"),
            Settings {
                track: Params { hold: 2, every: 3 },
                memory: Memory::Gpu,
            }
        );
        let error = settings(r#"{"memory":"cpu"}"#).expect_err("not a memory");
        assert!(error.contains("memory"), "{error}");
    }

    #[test]
    fn v_is_held_on_the_gpu_only_when_memory_says_so() {
        let held = <Ports as Guest>::memory(r#"{"memory":"gpu"}"#.to_string());
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].port, "v");
        assert!(matches!(held[0].memory, FrameMemory::Gpu));
        for host in ["", "{}", r#"{"memory":"host"}"#, r#"{"memory":"cpu"}"#] {
            assert!(
                <Ports as Guest>::memory(host.to_string()).is_empty(),
                "{host}"
            );
        }
    }

    #[test]
    fn v_takes_what_the_device_holds_on_gpu_and_rgba_on_host() {
        let bound = ["v".to_owned(), "boxes".to_owned()];
        let on_gpu = Runner::<crate::Ages>::shape(r#"{"memory":"gpu"}"#, &bound).expect("a shape");
        assert_eq!(
            on_gpu.inputs[0].accepts.pixel_formats,
            ["rgba", "yuv420p", "yuv422p", "yuv444p"]
        );
        let on_host =
            Runner::<crate::Ages>::shape(r#"{"memory":"host"}"#, &bound).expect("a shape");
        assert_eq!(on_host.inputs[0].accepts.pixel_formats, ["rgba"]);
        for shape in [on_gpu, on_host] {
            assert_eq!(shape.outputs.len(), 1, "the rows and nothing else");
            assert_eq!(shape.outputs[0].name, "faces");
        }
    }

    #[test]
    fn a_crop_is_its_padded_rect_as_an_edge_and_a_size() {
        let rect = Rect::padded(100.0, 50.0, 40.0, 60.0, faceage_common::PAD, 1280, 720)
            .expect("on the frame");
        let crop = crop_rect(rect);
        assert_eq!((crop.x, crop.y, crop.width, crop.height), (96, 44, 48, 72));
        // Cut to the frame where the padding reaches past it.
        let rect = Rect::padded(0.0, 0.0, 40.0, 60.0, faceage_common::PAD, 1280, 720)
            .expect("on the frame");
        let crop = crop_rect(rect);
        assert_eq!((crop.x, crop.y, crop.width, crop.height), (0, 0, 44, 66));
    }

    #[test]
    fn the_tensor_is_the_host_builds_crop_resize_and_normalization() {
        let options = options("ages");
        assert_eq!(options.model, "ages");
        assert_eq!((options.width, options.height), (224, 224));
        assert!(matches!(options.layout, TensorLayout::Nchw));
        assert!(matches!(options.order, ChannelOrder::Rgb));
        assert!(matches!(options.fit, Fit::Stretch));
        assert!(matches!(options.filter, ResizeFilter::Bicubic));
        assert!(
            options.antialias,
            "Pillow widens the kernel where it shrinks"
        );
        assert_eq!(options.scale, 1.0);
        assert_eq!(options.mean, (0.485, 0.456, 0.406));
        assert_eq!(options.std, (0.229, 0.224, 0.225));
    }
}
