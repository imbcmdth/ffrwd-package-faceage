-- Every face with its age as NDJSON, one line per face per frame - the class
-- `face`, the detector's confidence, the box in pixels, and the estimated age
-- in years. No video is written.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), hold (frames a face the detector lost is carried for, defaults to 5), every (frames between one face's scores, defaults to 1), track (video track index, defaults to the first), dest (output path, a .ndjson file)
-- example: ffrwd compile -f packages/ffrwd/faceage/recipes/ages.sql -v source=class.mp4 -v dest=ages.ndjson
COPY (
  SELECT ffrwd.faceage.ages(ffrwd.rfdetr.detect_faces(v, :conf), :hold, :every).faces
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest'
