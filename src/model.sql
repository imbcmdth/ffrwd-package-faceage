-- The one model export, hosted as a wasm module the package ships. The
-- weights are pinned in the manifest and land beside the module at install.
--
-- `ages` reads the face boxes a detector put beside the picture and answers
-- the same faces with an age on each: it crops every box, runs the crop
-- through the age model, and returns the frame untouched with one row per
-- face - the class, the confidence and the box it was given, plus the age in
-- years. Boxes are followed from frame to frame, so a face keeps one
-- estimate rather than a fresh one per frame: `every` is how many frames
-- apart a face is scored, and the age a row carries is the running mean of
-- what that face scored. `hold` keeps a face the detector lost for that many
-- frames, re-emitted where it was last seen, so a blur does not flicker off
-- for the frame a head turns.
CREATE FUNCTION ages(v video_stream,
                     boxes STRUCT(class text, conf number,
                                  x number, y number, w number, h number)[],
                     hold number DEFAULT 5, every number DEFAULT 1)
RETURNS STRUCT(v video_stream, faces STRUCT(class text, conf number,
                                            x number, y number, w number, h number,
                                            age number)[])
  AS 'target/wasm32-wasip2/release/ages.wasm', 'ages' LANGUAGE wasm;
