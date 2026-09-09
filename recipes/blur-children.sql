-- Blur the children and leave everyone else alone: every face the detector
-- finds is aged, and the ones estimated under `max_age` are blurred. `max_age`
-- has to be given - a runtime row predicate carries no default - and 16 is
-- what this recipe was built for. `grow` pads each box so hair and the jaw
-- line go with it.
-- variables: source (input media path), max_age (faces estimated younger than this are blurred; required, 16 is what this recipe was built for), conf (confidence threshold, defaults to 0.25), grow (pixels added around each box, defaults to 8), feather (how far the edge softens in pixels, defaults to 4), sigma (blur strength, defaults to 12), hold (frames a face the detector lost is carried for, defaults to 5), every (frames between one face's scores, defaults to 1), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/faceage/recipes/blur-children.sql -v source=class.mp4 -v max_age=16 -v dest=blurred.mp4
COPY (
  SELECT ffrwd.mask_tools.blur_where(
           v,
           ffrwd.faceage.faces_mask(
             ffrwd.faceage.ages(ffrwd.rfdetr.detect_faces(v, :conf), :hold, :every).v,
             ARRAY(SELECT r FROM unnest(ffrwd.faceage.ages(ffrwd.rfdetr.detect_faces(v, :conf), :hold, :every).faces) r
                   WHERE r.age < :max_age),
             COALESCE(:grow, 8), COALESCE(:feather, 4)),
           :sigma),
         f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
