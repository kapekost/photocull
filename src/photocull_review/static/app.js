/* The review UI: one cluster at a time, the proposed keeper beside one challenger.
 *
 * Three shapes here follow decisions rather than convenience.
 *
 * **The left pane is pinned to the scorer's proposed keeper for the whole cluster** and
 * the right pane steps through the other takes. Pinning matters: if the left pane
 * tracked "whichever photo is currently marked keep", flipping a mark would reshuffle
 * the layout under the owner's eyes mid-comparison, and the one job of this screen is
 * holding two photographs still beside each other.
 *
 * **Every cluster opens with a suggestion, ambiguous ones included.** The close call is
 * surfaced as prose attached to the proposal it qualifies, not as a warning in the list
 * above — an ambiguous cluster is byte-identical in shape to a confident one, so the
 * note has to sit where the recommendation is or a coin flip reads as a considered pick.
 *
 * **A favourite is intent, never a mutation** (per `docs/SPEC.md`). `F` records that
 * this photo should be favourited when write-back runs; this process never touches
 * Photos, and cannot — `photoscript` is not imported anywhere it could reach.
 *
 * The `window.__photocull` handle at the bottom is a deliberate test seam. Playwright
 * needs to assert on state a screenshot cannot show — which URLs were preloaded, what
 * the keymap contains — and the alternative is scattering `data-` attributes over the
 * DOM purely for tests. Nothing in the app reads it.
 */

import { KEYMAP, actionFor } from "/static/keymap.js";
import {
  canMagnify,
  clusterMaxScale,
  isMixedResolution,
  placement,
} from "/static/magnify.js";
import { createOverview } from "/static/overview.js";

/** How many clusters ahead to fetch images for. Two is enough to cover the pause
 *  between deciding one cluster and looking at the next, and small enough that even a
 *  session with thousands of clusters never has more than a handful of images in
 *  flight. */
const PRELOAD_AHEAD = 2;

const KEEP = "keep";
const CULL = "cull";
const UNSET = "unset";

const state = {
  doc: null,
  clusters: [],
  index: 0,
  challenger: 0,
  focus: "challenger",
  marks: new Map(),
  favorites: new Map(),
  //: What `C` replaced, per cluster, so pressing it again restores the marks and
  //: favourites that were actually on screen rather than the scorer's proposal.
  beforeCullAll: new Map(),
  decided: new Map(),
  submitted: [],
  helpOpen: false,
  //: Magnification is a way of looking, not a mode: it survives stepping to the next
  //: take, and every decision key still works while it is on. The crop is normalised to
  //: the *photograph* rather than the frame, which is what lets it survive — holding one
  //: corner still while the takes change underneath it is the whole point of the feature.
  magnified: false,
  crop: { x: 0.5, y: 0.5 },
};

/** Kept alive so the browser does not cancel an in-flight preload, and so a test can
 *  see what was asked for rather than inferring it from network timing. */
const preloaded = [];

const el = {
  loading: document.getElementById("loading"),
  failure: document.getElementById("failure"),
  app: document.getElementById("app"),
  position: document.getElementById("position"),
  progress: document.getElementById("progress"),
  notes: document.getElementById("notes"),
  message: document.getElementById("message"),
  hint: document.getElementById("hint"),
  help: document.getElementById("help"),
  helpKeys: document.getElementById("help-keys"),
  helpButton: document.getElementById("help-button"),
  basis: document.getElementById("basis"),
  panes: {
    keeper: document.getElementById("pane-keeper"),
    challenger: document.getElementById("pane-challenger"),
  },
  evidence: {
    keeper: document.getElementById("evidence-keeper"),
    challenger: document.getElementById("evidence-challenger"),
  },
  dashboard: document.getElementById("dashboard"),
  dashboardBody: document.getElementById("dashboard-body"),
  sheet: document.getElementById("final-check"),
  sheetHead: document.getElementById("sheet-head"),
  sheetLock: document.getElementById("sheet-lock"),
  sheetConfirm: document.getElementById("sheet-confirm"),
  sheetWriteback: document.getElementById("sheet-writeback"),
  sheetClose: document.getElementById("sheet-close"),
  sheetScroll: document.getElementById("sheet-scroll"),
  sheetNote: document.getElementById("sheet-note"),
  sheetRows: document.getElementById("sheet-rows"),
  sheetMore: document.getElementById("sheet-more"),
};

// --- talking to the server -------------------------------------------------------

async function getJson(path) {
  const response = await fetch(path, { credentials: "same-origin" });
  if (!response.ok) {
    throw new Error(`${path} answered ${response.status}`);
  }
  return response.json();
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }
  return { ok: response.ok, status: response.status, data };
}

const imageUrl = (uuid) => `/api/images/${encodeURIComponent(uuid)}`;

// --- reading the current position -------------------------------------------------

const current = () => state.clusters[state.index];

function keeperUuid(entry) {
  return entry.winner_uuid ?? entry.members[0].uuid;
}

/** The takes the challenger pane steps through: everything but the pinned keeper. */
function challengers(entry) {
  const pinned = keeperUuid(entry);
  return entry.members.filter((member) => member.uuid !== pinned);
}

function memberFor(entry, side) {
  if (side === "keeper") {
    return entry.members.find((member) => member.uuid === keeperUuid(entry));
  }
  const others = challengers(entry);
  return others.length ? others[state.challenger % others.length] : null;
}

function marksFor(entry) {
  const key = entry.cluster_key;
  if (!state.marks.has(key)) {
    const decision = state.decided.get(key);
    state.marks.set(key, { ...(decision ? decision.marks : entry.proposed) });
  }
  return state.marks.get(key);
}

function favoritesFor(entry) {
  const key = entry.cluster_key;
  if (!state.favorites.has(key)) {
    // Seeded from the recorded decision rather than from an empty set: a review survives
    // a closed browser by design, so after a reload the favourites of an already-decided
    // cluster live only in the session document. Starting empty would mean re-recording
    // that cluster silently un-favourites everything in it.
    const decision = state.decided.get(key);
    state.favorites.set(key, new Set(decision ? decision.favorites ?? [] : []));
  }
  return state.favorites.get(key);
}

/** Take a recorded decision into the page's own state, from wherever it was made.
 *
 *  Shared by `Enter` and by the contact sheet's click-to-keep, because both record the
 *  same thing and the sheet is the surface where getting it wrong is invisible: a photo
 *  drawn as kept while the log still reads `cull` is precisely the disagreement the final
 *  check exists to rule out. The favourites come back from the server rather than being
 *  assumed unchanged — the sheet drops one when it re-stages a photo. */
function applyDecision(key, data) {
  state.decided.set(key, {
    marks: data.marks,
    favorites: data.favorites ?? [],
    batch_id: data.batch_id,
  });
  state.marks.set(key, { ...data.marks });
  state.favorites.set(key, new Set(data.favorites ?? []));
}

function firstUndecided() {
  const index = state.clusters.findIndex(
    (entry) => !state.decided.has(entry.cluster_key)
  );
  return index === -1 ? 0 : index;
}

// --- rendering ---------------------------------------------------------------------

function formatScore(value) {
  return value === null || value === undefined ? null : value.toFixed(3);
}

/** When this cluster was taken, if any of its takes carries a date.

 *  Worth the line: a review is thousands of clusters long, and "these are from the
 *  Naxos trip" is often what settles whether a near-duplicate is worth keeping at all. */
function clusterDate(entry) {
  const stamp = entry.members.map((member) => member.date).find(Boolean);
  if (!stamp) return "";
  const when = new Date(stamp);
  if (Number.isNaN(when.getTime())) return "";
  return ` · ${when.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  })}`;
}

/** Every criterion any take in this cluster was scored on, so both panes list the same
 *  rows in the same order. Two things depend on it: the eye compares `sharpness` to
 *  `sharpness` on one line, and — because the panes then have identical heights — the
 *  frames above them come out exactly equal without measuring anything.
 *
 *  The union is deliberate redundancy and is **not** covered by a test, which is worth
 *  saying rather than leaving to be discovered: `scoring.score_photo` builds
 *  `sub_scores` as a literal with all eight keys, so no photo can be missing one today
 *  and a mutation reading only the first member's keys would survive the suite. It is
 *  kept anyway because it costs three lines and makes this layout rule self-contained
 *  instead of contingent on a dict literal two packages away. */
function criteriaOf(entry) {
  const names = new Set();
  for (const member of entry.members) {
    for (const name of Object.keys(member.sub_scores)) names.add(name);
  }
  return [...names].sort();
}

function renderScores(node, entry, member) {
  node.replaceChildren();
  if (!member) return;

  const rows = [["total", member.total]];
  for (const name of criteriaOf(entry)) {
    rows.push([name, member.sub_scores[name] ?? null]);
  }
  // Always rendered, in both panes. The keeper has no distance to itself, and printing
  // the row anyway is what keeps the two tables the same height.
  rows.push(["distance to keeper", member.distance_to_winner ?? null]);

  for (const [name, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = name;
    const detail = document.createElement("dd");
    const shown = formatScore(value);
    // An absent measurement is said in words. The decision log keeps `None` out for
    // the same reason it stays out of the UI: a real share of sub-scores are absent
    // on any real library, and printing 0.000 would report a measurement nobody took.
    // The keeper's distance to itself is a different kind of nothing — not
    // unmeasured, just not a question.
    const absent =
      name === "distance to keeper" && member.is_winner ? "—" : "not measured";
    detail.textContent = shown === null ? absent : shown;
    if (shown === null) detail.classList.add("absent");
    if (name === "total") {
      term.classList.add("total");
      detail.classList.add("total");
    }
    node.append(term, detail);
  }
}

function renderPane(side, entry) {
  const pane = el.panes[side];
  const member = memberFor(entry, side);
  const role = pane.querySelector("[data-role]");
  const badge = pane.querySelector("[data-badge]");
  const image = pane.querySelector("[data-photo]");
  const gone = pane.querySelector("[data-gone]");
  const closeCall = pane.querySelector("[data-close-call]");

  pane.classList.toggle("focused", state.focus === side);

  if (!member) {
    role.textContent = "No other take";
    badge.textContent = "";
    badge.removeAttribute("data-mark");
    image.hidden = true;
    gone.hidden = false;
    gone.textContent = "This cluster holds a single take.";
    renderScores(pane.querySelector("[data-scores]"), entry, null);
    closeCall.hidden = true;
    return;
  }

  const others = challengers(entry);
  // Most of the demo's clusters are pairs, and so is most of a real library — so
  // "Take 1 of 1" is the single most-shown label in the app, and it reads as a counter
  // that failed rather than as a statement about the cluster.
  role.textContent =
    side === "keeper"
      ? "Proposed keeper"
      : others.length === 1
        ? "The other take"
        : `Take ${(state.challenger % others.length) + 1} of ${others.length}`;

  const mark = marksFor(entry)[member.uuid] ?? UNSET;
  const favorited = favoritesFor(entry).has(member.uuid);
  badge.setAttribute("data-mark", mark);
  badge.replaceChildren(document.createTextNode(mark.toUpperCase()));
  if (favorited) {
    const star = document.createElement("span");
    star.className = "fav";
    star.textContent = " ★ favourite";
    badge.append(star);
  }

  // Two paths lead to "there is nothing to show", and both are ordinary. The session
  // document already knows which derivatives were missing when the session opened, so
  // that case never issues a request; the endpoint re-checks the file per request and
  // 410s if Photos reclaimed it since, which the error handler catches. Deciding from
  // the document first matters on re-render: an image whose `src` is unchanged fires no
  // second error, so a handler alone would silently un-hide it.
  const missing = member.display && member.display.available === false;
  const evicted = "No local copy of this take — Photos has reclaimed its derivative.";
  image.hidden = missing;
  gone.hidden = !missing;
  gone.textContent = missing ? evicted : "";
  if (!missing) {
    image.alt = `Take ${member.uuid}`;
    const wanted = imageUrl(member.uuid);
    if (image.getAttribute("src") !== wanted) {
      image.setAttribute("src", wanted);
    }
    image.onerror = () => {
      image.hidden = true;
      gone.hidden = false;
      gone.textContent = evicted;
    };
    // A take that has just been stepped to still reports the *previous* raster's
    // `naturalWidth` until the new one decodes, and magnification is arithmetic on that
    // number. Redrawing on load is what keeps a 480px take from being placed as though
    // it were the 1024px one that was on screen a moment ago.
    image.onload = () => applyMagnification();
  }

  // The close call belongs to the proposal, so only the keeper's copy is legible — but
  // the challenger renders the same text invisibly, because a note on one side only
  // would make that pane taller and its frame correspondingly shorter. Reserving the
  // space is what lets the equal-size rule hold without measuring either pane.
  const closeNote = (entry.annotations ?? []).find((note) => note.code === "ambiguous");
  closeCall.hidden = !closeNote;
  closeCall.textContent = closeNote ? closeNote.text : "";
  const reserved = Boolean(closeNote) && side !== "keeper";
  closeCall.classList.toggle("reserved", reserved);
  if (reserved) {
    closeCall.setAttribute("aria-hidden", "true");
  } else {
    closeCall.removeAttribute("aria-hidden");
  }

  renderScores(pane.querySelector("[data-scores]"), entry, member);
}

// --- magnification -------------------------------------------------------------------

/** Each pane's raster and frame, as plain numbers for `magnify.js`.
 *
 *  `null` for a pane with nothing decoded — an evicted derivative, or an image whose
 *  `src` changed a moment ago and has not loaded yet. Both are ordinary, and both must
 *  be skipped rather than counted as a zero-sized raster. */
function paneViews() {
  const views = {};
  for (const side of ["keeper", "challenger"]) {
    const pane = el.panes[side];
    const image = pane.querySelector("[data-photo]");
    const box = pane.querySelector("[data-frame]").getBoundingClientRect();
    views[side] =
      image.hidden || !image.naturalWidth
        ? null
        : {
            naturalWidth: image.naturalWidth,
            naturalHeight: image.naturalHeight,
            boxWidth: box.width,
            boxHeight: box.height,
          };
  }
  return views;
}

const liveViews = (views) => Object.values(views).filter(Boolean);

/** Draw the current magnification state, and return the scale actually used.
 *
 *  The single place a scale is computed, deliberately: it reads the frames as they are
 *  *now*, so a window resize, an annotation appearing, or the message strip opening
 *  underneath cannot leave the photographs drawn past their cap. Everything it writes is
 *  an inline style it also knows how to clear, so the fitted view is restored by removing
 *  what magnification added rather than by re-stating it. */
function applyMagnification() {
  const views = paneViews();
  const scale = state.magnified ? clusterMaxScale(liveViews(views)) : null;

  for (const side of ["keeper", "challenger"]) {
    const pane = el.panes[side];
    const frame = pane.querySelector("[data-frame]");
    const image = pane.querySelector("[data-photo]");
    const label = pane.querySelector("[data-zoom-label]");
    const spot = scale === null ? null : placement({ ...views[side], scale, centerX: state.crop.x, centerY: state.crop.y });

    if (!spot) {
      frame.classList.remove("magnified");
      for (const property of ["width", "height", "left", "top"]) {
        image.style[property] = "";
      }
      label.hidden = true;
      label.textContent = "";
      continue;
    }

    frame.classList.add("magnified");
    image.style.width = `${spot.width}px`;
    image.style.height = `${spot.height}px`;
    image.style.left = `${spot.left}px`;
    image.style.top = `${spot.top}px`;
    label.hidden = false;
    // The size of the copy on disk, not the size on screen. At this magnification that
    // is the whole question: how much is really there.
    //
    // The two are the same number **today**, and a mutation swapping them survives the
    // suite — `Z` goes straight to the cap, and at the cap the drawing is exactly the
    // raster's own size. Recorded rather than "fixed", because the equivalence is a
    // property of there being one zoom step: the day this grows an intermediate scale,
    // reading `spot` would quietly turn a statement about the photograph into a
    // statement about the window.
    label.textContent =
      `${views[side].naturalWidth} × ${views[side].naturalHeight} px` +
      ` · ${scale.toFixed(2)}×`;
  }
  return scale;
}

function exitMagnification() {
  state.magnified = false;
  state.crop = { x: 0.5, y: 0.5 };
  say(null);
  applyMagnification();
}

/** Move the crop by a drag on one pane, in that pane's CSS pixels.
 *
 *  Both panes move, because they share one crop — that is the feature. The stored centre
 *  comes back clamped so a drag that runs past the edge banks no distance the owner then
 *  has to drag back before anything moves. */
function panBy(side, dx, dy) {
  const views = paneViews();
  const view = views[side];
  const scale = clusterMaxScale(liveViews(views));
  if (!view || scale === null) return;
  const here = placement({ ...view, scale, centerX: state.crop.x, centerY: state.crop.y });
  const moved = placement({
    ...view,
    scale,
    centerX: state.crop.x - dx / here.width,
    centerY: state.crop.y - dy / here.height,
  });
  state.crop = { x: moved.centerX, y: moved.centerY };
  applyMagnification();
}

/** The sentences that say *why*, under the photograph they are about.
 *
 *  Written by the server (`explain.py`) and never derived here: the rules are about what
 *  the scorer did — which criteria it ranked on, which weights are zero — and those live
 *  in `ClusterConfig`, which this document deliberately does not ship in a form the
 *  browser could reconstruct them from. */
function renderEvidence(side, entry) {
  const node = el.evidence[side];
  const member = memberFor(entry, side);
  node.replaceChildren();
  const sentences = member?.evidence ?? [];
  if (!sentences.length) return;
  const list = document.createElement("ul");
  for (const sentence of sentences) {
    const item = document.createElement("li");
    item.textContent = sentence;
    list.append(item);
  }
  node.append(list);
}

/** What the cluster could be ranked on at all — one line for both takes, because it is
 *  a fact about the cluster. The large majority of real clusters drop a criterion, so
 *  rendering it per pane would say the same thing twice on nearly every screen. */
function renderBasis(entry) {
  const basis = entry.ranking_basis ?? null;
  el.basis.hidden = !basis;
  el.basis.textContent = basis ?? "";
}

function renderNotes(entry) {
  el.notes.replaceChildren();
  // The close call is deliberately absent from this list: it belongs beside the
  // proposal it qualifies, not in a pile of warnings about the cluster.
  for (const note of entry.annotations ?? []) {
    if (note.code === "ambiguous") continue;
    const line = document.createElement("p");
    line.className = "note";
    line.dataset.code = note.code;
    line.textContent = note.text;
    el.notes.append(line);
  }
}

function renderHint() {
  el.hint.replaceChildren();
  // Rendered from KEYMAP like the overlay is, but from the short `hint` phrasing: this
  // strip is on screen for the whole review, so it has to be scanned rather than read.
  // A null hint means the key is not in the footer — see the note in keymap.js.
  let first = true;
  for (const row of KEYMAP) {
    if (row.hint === null) continue;
    if (!first) el.hint.append(document.createTextNode("   ·   "));
    first = false;
    const key = document.createElement("kbd");
    key.textContent = row.footerLabel ?? row.label;
    el.hint.append(key, document.createTextNode(` ${row.hint}`));
  }
}

function renderHelp() {
  el.helpKeys.replaceChildren();
  for (const row of KEYMAP) {
    const term = document.createElement("dt");
    term.textContent = row.label;
    const detail = document.createElement("dd");
    detail.textContent = row.description;
    el.helpKeys.append(term, detail);
  }
  el.help.hidden = !state.helpOpen;
}

function say(text, kind = "error") {
  if (!text) {
    el.message.hidden = true;
    el.message.textContent = "";
    return;
  }
  el.message.hidden = false;
  el.message.textContent = text;
  el.message.dataset.kind = kind;
}

function render() {
  const entry = current();
  if (!entry) return;

  // The tally is what makes a large cluster workable. Stepping through many challengers
  // one at a time otherwise gives no sense of standing — you can see this take's badge
  // and nothing about the ones already behind you, and the dashboard is what answers
  // that properly.
  const marks = marksFor(entry);
  const kept = Object.values(marks).filter((mark) => mark === KEEP).length;
  const staged = Object.values(marks).filter((mark) => mark === CULL).length;
  el.position.textContent =
    `Cluster ${state.index + 1} of ${state.clusters.length}` +
    ` · ${entry.members.length} takes` +
    ` · keeping ${kept}, staging ${staged}` +
    clusterDate(entry);
  el.progress.textContent = `${state.decided.size} decided · ${
    state.clusters.length - state.decided.size
  } to go`;

  renderNotes(entry);
  renderPane("keeper", entry);
  renderPane("challenger", entry);
  renderEvidence("keeper", entry);
  renderEvidence("challenger", entry);
  renderBasis(entry);
  renderHelp();
  applyMagnification();
  preload();
}

function preload() {
  for (let ahead = 1; ahead <= PRELOAD_AHEAD; ahead += 1) {
    const entry = state.clusters[state.index + ahead];
    if (!entry) break;
    for (const member of entry.members) {
      const url = imageUrl(member.uuid);
      if (preloaded.some((image) => image.dataset.url === url)) continue;
      const image = new Image();
      image.dataset.url = url;
      image.src = url;
      preloaded.push(image);
    }
  }
}

// --- the two overview surfaces ---------------------------------------------------------
//
// Built here rather than inside `overview.js` so that module owns no application state:
// everything it needs to reach back into — where a cluster sits in the walk, what was
// decided about it, how to record a change — arrives as a function, and the compare view
// stays the only thing that moves the cursor.

const overview = createOverview({
  nodes: el,
  getJson,
  postJson,
  imageUrl,
  say,
  indexOf: (key) => state.clusters.findIndex((entry) => entry.cluster_key === key),
  decisionFor: (key) => state.decided.get(key),
  applyDecision,
  onJump: (index) => {
    if (index >= 0) goTo(index);
  },
});

// --- actions -------------------------------------------------------------------------

function goTo(index) {
  state.index = Math.max(0, Math.min(index, state.clusters.length - 1));
  state.challenger = 0;
  state.focus = "challenger";
  // Magnification is per cluster, and that is a decision rather than an omission. A crop
  // means something across the takes of one cluster — they are the same framing seconds
  // apart, so holding one corner still is exactly how two faces get compared. Across
  // clusters it means nothing, and arriving already zoomed into a corner would hide the
  // photograph the owner is being asked to judge.
  state.magnified = false;
  state.crop = { x: 0.5, y: 0.5 };
  say(null);
  render();
}

const actions = {
  previousTake() {
    const others = challengers(current());
    if (!others.length) return;
    state.challenger = (state.challenger - 1 + others.length) % others.length;
    render();
  },

  nextTake() {
    const others = challengers(current());
    if (!others.length) return;
    state.challenger = (state.challenger + 1) % others.length;
    render();
  },

  // `←`/`→` name a side rather than alternating, so `←` is the left pane every time.
  // A toggle on a directional key means the second press undoes the first, which is
  // precisely the "I pressed it and it went back" the owner would hit on a pair.
  focusKeeper() {
    state.focus = "keeper";
    render();
  },

  focusChallenger() {
    state.focus = "challenger";
    render();
  },

  switchPane() {
    state.focus = state.focus === "keeper" ? "challenger" : "keeper";
    render();
  },

  toggleMagnify() {
    if (state.magnified) {
      exitMagnification();
      return;
    }
    const views = liveViews(paneViews());
    if (!canMagnify(views)) {
      // "Further zoom disabled", said rather than silently done. This is the ordinary
      // state of the 24.1% of real clusters with no take above 640px, not an edge — and
      // a key that appears to do nothing is indistinguishable from one that is broken.
      say(
        "These copies are already drawn larger than they are, so this is as much detail" +
          " as there is to see without the full-resolution original.",
        "note"
      );
      return;
    }
    // The crop is not reset here, and that is measured rather than trusted: every path
    // that leaves magnification (`exitMagnification`, `goTo`) already recentres it, so a
    // reset on entry changed nothing observable and no test could tell the two versions
    // apart. Dropped on the same reasoning that retired a redundant guard on the
    // session-lifetime touch handler elsewhere in this project -- keep only redundancy
    // you can actually measure.
    state.magnified = true;
    // Both panes magnify to one scale, capped by the smaller raster — so on a mixed pair
    // the larger take is deliberately not shown at its own detail, and the difference the
    // owner is looking for may belong to the copies rather than to the photographs.
    const sides = views.map((view) => view.naturalWidth);
    say(
      isMixedResolution(views)
        ? `One take here has a ${Math.min(...sides)}px local copy and the other ` +
            `${Math.max(...sides)}px, and both magnify to the same scale — so which ` +
            "looks sharper here is not a fair comparison."
        : null,
      "note"
    );
    applyMagnification();
  },

  toggleMark() {
    const entry = current();
    const member = memberFor(entry, state.focus);
    if (!member) return;
    const marks = marksFor(entry);
    const next = marks[member.uuid] === KEEP ? CULL : KEEP;
    marks[member.uuid] = next;
    // A staged photo cannot also be a favourite — write-back would favourite a photo it
    // is about to put in `Cull/Candidates`. Dropping the favourite here keeps the
    // submission valid instead of failing at the server.
    if (next === CULL) favoritesFor(entry).delete(member.uuid);
    say(null);
    render();
  },

  cullAll() {
    const entry = current();
    const key = entry.cluster_key;
    const marks = marksFor(entry);
    if (Object.values(marks).every((mark) => mark === CULL)) {
      // Put back what was on screen, not the scorer's proposal: the owner may have
      // decided several takes by hand before pressing this, and discarding that
      // silently is worse than the mis-press it exists to undo.
      const previous = state.beforeCullAll.get(key);
      state.marks.set(key, { ...(previous ? previous.marks : entry.proposed) });
      state.favorites.set(key, new Set(previous ? previous.favorites : []));
      state.beforeCullAll.delete(key);
    } else {
      state.beforeCullAll.set(key, {
        marks: { ...marks },
        favorites: new Set(favoritesFor(entry)),
      });
      for (const member of entry.members) marks[member.uuid] = CULL;
      // A staged photo cannot also be a favourite, and that rule still applies when
      // culling the whole cluster — so this has to clear them here, or `F` then `C`
      // builds a submission the server refuses at `Enter`, which is the exact shape
      // of the complaint this key answers.
      favoritesFor(entry).clear();
    }
    say(null);
    render();
  },

  toggleFavorite() {
    const entry = current();
    const member = memberFor(entry, state.focus);
    if (!member) return;
    const favorites = favoritesFor(entry);
    if (favorites.has(member.uuid)) {
      favorites.delete(member.uuid);
      say(null);
    } else if (marksFor(entry)[member.uuid] === CULL) {
      say(
        "That take is staged for culling, so it cannot be favourited. Keep it first."
      );
    } else {
      favorites.add(member.uuid);
      say(null);
    }
    render();
  },

  async submit() {
    const entry = current();
    const key = entry.cluster_key;
    const result = await postJson(`/api/clusters/${encodeURIComponent(key)}/decision`, {
      marks: marksFor(entry),
      favorites: [...favoritesFor(entry)],
    });
    if (!result.ok) {
      say(result.data.detail ?? `The server refused this decision (${result.status}).`);
      return;
    }
    applyDecision(key, result.data);
    state.submitted.push(key);

    if (state.index + 1 < state.clusters.length) {
      goTo(state.index + 1);
    } else {
      render();
      say("That was the last cluster in this session.", "done");
    }
  },

  async undo() {
    const key = state.submitted.pop() ?? current().cluster_key;
    const result = await postJson("/api/undo", { cluster_key: key });
    if (!result.ok) {
      say(result.data.detail ?? `Undo failed (${result.status}).`);
      return;
    }
    const index = state.clusters.findIndex((entry) => entry.cluster_key === key);
    const entry = state.clusters[index];
    if (result.data.batch_id === null || result.data.batch_id === undefined) {
      state.decided.delete(key);
      state.marks.set(key, { ...entry.proposed });
      state.favorites.delete(key);
    } else {
      applyDecision(key, result.data);
    }
    goTo(index);
  },

  toggleHelp() {
    state.helpOpen = !state.helpOpen;
    renderHelp();
  },

  async toggleDashboard() {
    if (overview.dashboardOpen()) {
      overview.closeDashboard();
      return;
    }
    // Opening one overview closes the other. Both are "the whole session at once", and
    // stacking them would leave the owner two Escapes away from the photographs.
    overview.closeSheet();
    await overview.openDashboard();
  },

  async toggleFinalCheck() {
    if (overview.sheetOpen()) {
      overview.closeSheet();
      return;
    }
    overview.closeDashboard();
    await overview.openSheet();
  },

  // Four things answer to `Esc`, so the order is stated here rather than left to whoever
  // reads the file next: whatever is on top closes first. An `Esc` that dropped the
  // magnification underneath a help list that stayed open would be the wrong one every
  // time, and the same holds for the two overviews.
  closeOverlay() {
    if (state.helpOpen) {
      state.helpOpen = false;
      renderHelp();
      return;
    }
    if (overview.sheetOpen()) {
      overview.closeSheet();
      return;
    }
    if (overview.dashboardOpen()) {
      overview.closeDashboard();
      return;
    }
    if (state.magnified) exitMagnification();
  },
};

/** Which actions still work while something is open over the review.
 *
 *  Everything else is swallowed, and that is a correctness rule rather than a nicety:
 *  `Space` under an open contact sheet flips a mark on a cluster the owner cannot see,
 *  and `Enter` records it. The help overlay had the identical hole -- it was merely
 *  harder to reach, because nobody leaves a keyboard-shortcut list open. */
const OVER_THE_REVIEW = new Set([
  "closeOverlay",
  "toggleHelp",
  "toggleDashboard",
  "toggleFinalCheck",
]);

const somethingIsOpen = () =>
  state.helpOpen || overview.dashboardOpen() || overview.sheetOpen();

// --- wiring ----------------------------------------------------------------------------

function onKeyDown(event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const action = actionFor(event.key);
  if (!action) return;
  // Space scrolls and Tab walks the focus ring; both would fight the review otherwise.
  event.preventDefault();
  if (somethingIsOpen() && !OVER_THE_REVIEW.has(action)) return;
  const run = actions[action];
  if (run) Promise.resolve(run()).catch((error) => say(String(error)));
}

async function start() {
  try {
    const doc = await getJson("/api/session");
    state.doc = doc;
    state.clusters = doc.clusters;
    for (const [key, decision] of Object.entries(doc.decided ?? {})) {
      state.decided.set(key, decision);
    }
    if (!state.clusters.length) {
      el.loading.textContent = "This session holds no clusters to review.";
      return;
    }
    el.loading.hidden = true;
    el.app.hidden = false;
    renderHint();
    goTo(firstUndecided());
  } catch (error) {
    el.loading.hidden = true;
    el.failure.hidden = false;
    el.failure.textContent = `Could not load the review session: ${error.message}`;
  }
}

// The evidence card is part of its column as far as the eye is concerned, so clicking it
// focuses that pane like clicking the photograph does.
for (const group of [el.panes, el.evidence]) {
  for (const [side, node] of Object.entries(group)) {
    node.addEventListener("click", () => {
      state.focus = side;
      render();
    });
  }
}
// Dragging a magnified pane pans both. Pointer events rather than mouse events so a
// trackpad, a mouse and a pen all take the same path, and pointer capture so a drag that
// runs off the frame keeps working instead of stopping at the edge.
for (const [side, pane] of Object.entries(el.panes)) {
  const frame = pane.querySelector("[data-frame]");
  let dragging = null;
  frame.addEventListener("pointerdown", (event) => {
    if (!state.magnified) return;
    dragging = { x: event.clientX, y: event.clientY };
    frame.classList.add("dragging");
    frame.setPointerCapture(event.pointerId);
  });
  frame.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    panBy(side, event.clientX - dragging.x, event.clientY - dragging.y);
    dragging = { x: event.clientX, y: event.clientY };
  });
  for (const ending of ["pointerup", "pointercancel"]) {
    frame.addEventListener(ending, (event) => {
      dragging = null;
      frame.classList.remove("dragging");
      if (frame.hasPointerCapture(event.pointerId)) frame.releasePointerCapture(event.pointerId);
    });
  }
  // Chromium drags an `<img>` natively, which swallows every pointer move after it and
  // makes panning look simply broken. This is the guard that stops it; a `preventDefault`
  // on `pointerdown` was tried too and dropped, because with the CSS rule beside it
  // neither could be shown to do anything the other did not.
  frame.addEventListener("dragstart", (event) => event.preventDefault());
}

// The frames are sized from what the window leaves over, so every resize changes the cap.
window.addEventListener("resize", () => applyMagnification());

el.helpButton.addEventListener("click", () => actions.toggleHelp());
el.sheetConfirm.addEventListener("click", () => {
  overview.confirm().catch((error) => say(String(error)));
});
el.sheetWriteback.addEventListener("click", () => {
  overview.writeBack().catch((error) => say(String(error)));
});
el.sheetClose.addEventListener("click", () => overview.closeSheet());
document.addEventListener("keydown", onKeyDown);

window.__photocull = {
  state,
  preloaded,
  KEYMAP,
  actions,
  goTo,
  marksFor,
  favoritesFor,
  overview,
};

start();
