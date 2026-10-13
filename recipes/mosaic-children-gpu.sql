-- `mosaic-children` with every step's GPU build: the picture is decoded onto
-- the GPU and stays there through the detector, the ages, the matte and the
-- mosaic. The variables and the rows are `mosaic-children`'s.
-- The picture leaves the GPU as rgba, so `pix_fmt` asks libx264 for 4:2:0,
-- the format the host recipe writes.
-- variables: source (input media path), max_age (faces estimated younger than this are mosaicked; required, 16 is what this recipe was built for), conf (confidence threshold, defaults to 0.25), grow (pixels added around each box, defaults to 8), feather (how far the edge softens in pixels, defaults to 4), size (mosaic block size, defaults to 16), hold (frames a face the detector lost is carried for, defaults to 5), every (frames between one face's scores, defaults to 5), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/faceage/recipes/mosaic-children-gpu.sql -v source=class.mp4 -v max_age=16 -v dest=mosaic.mp4
COPY (
  SELECT ffrwd.mask_tools.mosaic_where_gpu(
           v,
           ffrwd.rfdetr.boxes_mask_gpu(
             v,
             ARRAY(SELECT r FROM unnest(ffrwd.faceage.ages_gpu(v, ffrwd.rfdetr.detect_faces_gpu(v, :conf), :hold, COALESCE(:every, 5))) r
                   WHERE r.age < :max_age),
             grow => COALESCE(:grow, 8), feather => COALESCE(:feather, 4)),
           COALESCE(:size, 16)),
         f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20, pix_fmt 'yuv420p')
