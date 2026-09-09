-- The row consumer: a pure Rust module, no model. It takes exactly the
-- record `ages` returns - the six box fields with an age beside them - which
-- is why it is here rather than reused from the detector's own `boxes_mask`:
-- that one declares the six-field row and the compiler wants an exact match.
--
-- `faces_mask` rasterizes the faces into a grayscale matte - `grow` pads each
-- box outward in pixels, `feather` softens the edge over that many. Which
-- faces reach it is the query's business: filter the rows on `age` first and
-- the matte covers those faces alone.
CREATE FUNCTION faces_mask(v video_stream,
                           faces STRUCT(class text, conf number,
                                        x number, y number, w number, h number,
                                        age number)[],
                           grow number DEFAULT 0, feather number DEFAULT 0)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/faces_mask.wasm', 'faces_mask' LANGUAGE wasm;
