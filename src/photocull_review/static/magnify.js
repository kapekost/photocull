/* Magnification: compare two takes closely, and stop where the raster does.
 *
 * **The cap is one source pixel per CSS pixel.** `devicePixelRatio` deliberately does
 * not enter it, which was forced by measurement rather than a design preference. Even
 * fitted, with no magnification applied anywhere, an ordinary display raster already
 * renders at more than one device pixel per source pixel on most real screens once
 * `devicePixelRatio` is factored in -- so a cap that counted device pixels would refuse
 * plenty of perfectly normal panes outright. On a real library the effect is smaller
 * but still real: a meaningful share of sampled panes already run over such a cap when
 * merely fitted, and across a sample it would allow noticeably less real magnification
 * than the source-pixel cap actually delivers. None of that is visible to the suite,
 * because Playwright's default device pixel ratio is 1 — which is why one test runs at 2.
 *
 * The quantity that actually bounds honesty is the raster. Magnifying a 1024px copy from
 * 523 CSS px to 1024 CSS px on a dpr-2 screen invents no detail and loses none — the same
 * pixels, larger, which is what "compare closely" asked for. Past one source pixel per
 * CSS pixel you are looking at interpolation in the coordinate system the layout is
 * written in, and that is what every image viewer calls "actual size".
 *
 * Everything here is a pure function of numbers. Nothing reads the DOM, nothing writes
 * it, and nothing knows what a pane is — so the cap can be stated at a boundary a
 * viewport size cannot conjure.
 */

/** Magnifying by less than this does not visibly change the photograph, and offering it
 *  would be the "silently capped control" this task exists to avoid. Below the floor the
 *  answer is "there is nothing more to see", said in words. */
export const MIN_USEFUL_SCALE = 1.05;

/** Two takes whose rasters differ by more than this are not comparable on sharpness.
 *  The same 1.35 `session.MIXED_RESOLUTION_RATIO` annotates clusters with — pinned
 *  across the two languages by a test, because a drift would warn about a different
 *  population than the annotation on the very same screen. */
export const MIXED_RESOLUTION_RATIO = 1.35;

/** How `object-fit: contain` scales a raster into a box: the limiting ratio. */
function containScale({ naturalWidth, naturalHeight, boxWidth, boxHeight }) {
  if (!naturalWidth || !naturalHeight || !boxWidth || !boxHeight) return null;
  return Math.min(boxWidth / naturalWidth, boxHeight / naturalHeight);
}

/** How far the fitted drawing may be magnified before one source pixel is one CSS pixel.
 *
 *  Equivalently, how many times the frame fits into the raster. Returns `null` — never a
 *  number — when a size is unknown, because an image that has not decoded yet reports
 *  `naturalWidth === 0`, and answering `Infinity` there would magnify by whatever the
 *  first render happened to compute.
 *
 *  A result **below 1** is an ordinary answer, not a failure: `contain` upscales as
 *  happily as it downscales, so a 480px copy in a 819x526 frame is already drawn larger
 *  than it is and there is nothing to magnify into. That is the state of a real, sizeable
 *  share of real clusters with no take above 640px. */
export function maxScale(view) {
  const fit = containScale(view);
  return fit === null ? null : 1 / fit;
}

/** The one scale both panes share: the lower of their caps.
 *
 *  The higher-resolution take gives up detail it could have shown, because equal size is
 *  the load-bearing rule of side-by-side comparison — a large hero beside a small
 *  alternate biases the eye regardless of which photograph is better. Views whose
 *  size is unknown (an evicted derivative, an image still decoding) are skipped rather
 *  than counted as zero: one take being unshowable makes the *comparison* impossible,
 *  not the looking. */
export function clusterMaxScale(views) {
  const caps = views.map(maxScale).filter((cap) => cap !== null);
  return caps.length ? Math.min(...caps) : null;
}

/** True when magnifying this cluster would show the owner something. */
export function canMagnify(views) {
  const cap = clusterMaxScale(views);
  return cap !== null && cap >= MIN_USEFUL_SCALE;
}

/** True when these takes are being magnified to a scale their rasters do not share, so
 *  one may look softer than the other because of its copy rather than its focus. */
export function isMixedResolution(views) {
  const sides = views.map((view) => view.naturalWidth).filter(Boolean);
  if (sides.length < 2) return false;
  return Math.max(...sides) > Math.min(...sides) * MIXED_RESOLUTION_RATIO;
}

/** Where to draw a raster in its frame at `scale`, centred on a normalised point of the
 *  photograph.
 *
 *  `centerX`/`centerY` are in the *photograph's* coordinates rather than the frame's, so
 *  "the same crop" means the same corner of each take even when the two rasters are
 *  different shapes — which real takes within a cluster often are.
 *
 *  The clamped centre comes back out so the caller can store it. Clamping the stored
 *  value rather than only the drawn position is what keeps a drag that runs past the edge
 *  from banking distance the owner then has to drag back. */
export function placement({ scale, centerX, centerY, ...view }) {
  const fit = containScale(view);
  if (fit === null) return null;
  const width = view.naturalWidth * fit * scale;
  const height = view.naturalHeight * fit * scale;

  const place = (box, drawn, center) => {
    // Smaller than the frame in this axis: centred, and there is nothing to pan.
    if (drawn <= box) return [(box - drawn) / 2, 0.5];
    const left = Math.min(0, Math.max(box - drawn, box / 2 - center * drawn));
    return [left, (box / 2 - left) / drawn];
  };

  const [left, clampedX] = place(view.boxWidth, width, centerX);
  const [top, clampedY] = place(view.boxHeight, height, centerY);
  return { width, height, left, top, centerX: clampedX, centerY: clampedY };
}
