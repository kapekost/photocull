// photocull_album's sequencing UI: pick a trip, drag its photos into print order,
// adjust the crop on any photo whose aspect ratio doesn't already match the print
// size, then hand off to the CLI for the actual (iCloud-triggering) export -- this
// page never calls anything that touches the real Photos library.
//
// `moveBefore` is the same reorder rule pinned in Python as `photocull.albums.
// move_before` (tests/test_albums.py, Task 10 Step 3) -- proven once there so this
// two-line splice doesn't have to be trusted by inspection alone.

const state = {
  token: null,
  printSize: "4x6",
  trips: [],
  sequence: [],
  albumId: null,
  cropNeeds: {}, // uuid -> { needs_crop, offset_x, offset_y }
};

function el(id) {
  return document.getElementById(id);
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${state.token}` },
  });
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`);
  return resp.status === 204 ? null : resp.json();
}

function showSection(name) {
  for (const id of ["trip-picker", "sequencer", "export-result"]) {
    el(id).hidden = id !== name;
  }
}

function setStatus(text) {
  el("status").textContent = text;
}

function setError(text) {
  const node = el("error");
  if (text) {
    node.textContent = text;
    node.hidden = false;
  } else {
    node.hidden = true;
  }
}

function moveBefore(draggedUuid, targetUuid) {
  if (draggedUuid === targetUuid) return;
  const from = state.sequence.indexOf(draggedUuid);
  const to = state.sequence.indexOf(targetUuid);
  if (from === -1 || to === -1) return;
  state.sequence.splice(from, 1);
  state.sequence.splice(state.sequence.indexOf(targetUuid), 0, draggedUuid);
  renderSequence();
  saveSequence();
}

function renderSequence() {
  const grid = el("sequence-grid");
  grid.innerHTML = "";
  state.sequence.forEach((uuid, index) => {
    const card = document.createElement("div");
    card.className = "card";
    card.draggable = true;
    card.dataset.uuid = uuid;
    // `?t=` here, not an Authorization header: a plain <img src> can't set one --
    // see server.py's own docstring on why this one route checks the token this way.
    const src = `/api/image/${uuid}?t=${encodeURIComponent(state.token)}`;
    card.innerHTML = `<img src="${src}" loading="lazy"><span class="index">${index + 1}</span>`;

    card.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", uuid);
      card.classList.add("dragging");
    });
    card.addEventListener("dragend", () => card.classList.remove("dragging"));
    card.addEventListener("dragover", (e) => e.preventDefault());
    card.addEventListener("drop", (e) => {
      e.preventDefault();
      const draggedUuid = e.dataTransfer.getData("text/plain");
      moveBefore(draggedUuid, uuid);
    });

    const need = state.cropNeeds[uuid];
    if (need && need.needs_crop) {
      card.appendChild(buildCropOverlay(uuid, need));
    }

    grid.appendChild(card);
  });
}

// --- crop-adjustment overlay (Task 10 Step 4) ---------------------------------------
//
// Shown only when `GET /api/crop-preview/{uuid}` reports `needs_crop`. A
// mousedown+mousemove-driven pan within the card's own bounds, clamped 0..1 in both
// axes -- the same offset space `compute_crop_rect` consumes -- persisted on mouseup
// via `PUT /api/albums/{album_id}/crop/{uuid}`.

function buildCropOverlay(uuid, need) {
  const overlay = document.createElement("div");
  overlay.className = "crop-overlay";
  const crosshair = document.createElement("div");
  crosshair.className = "crosshair";
  crosshair.style.left = `${need.offset_x * 100}%`;
  crosshair.style.top = `${need.offset_y * 100}%`;
  overlay.appendChild(crosshair);

  let dragging = false;

  function positionFromEvent(e) {
    const rect = overlay.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const y = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    crosshair.style.left = `${x * 100}%`;
    crosshair.style.top = `${y * 100}%`;
    return { x, y };
  }

  overlay.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    positionFromEvent(e);
  });
  window.addEventListener("mousemove", (e) => {
    if (dragging) positionFromEvent(e);
  });
  window.addEventListener("mouseup", async (e) => {
    if (!dragging) return;
    dragging = false;
    const { x, y } = positionFromEvent(e);
    need.offset_x = x;
    need.offset_y = y;
    try {
      await api(`/api/albums/${state.albumId}/crop/${uuid}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ offset_x: x, offset_y: y }),
      });
    } catch (err) {
      setError(String(err));
    }
  });

  return overlay;
}

async function saveSequence() {
  try {
    await api(`/api/albums/${state.albumId}/sequence`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uuids: state.sequence }),
    });
  } catch (err) {
    setError(String(err));
  }
}

// --- trip picker ---------------------------------------------------------------------

async function loadTrips() {
  setStatus("Loading trips…");
  try {
    state.trips = await api("/api/trips");
  } catch (err) {
    setError(String(err));
    return;
  }
  setStatus("");
  renderTripList();
}

function renderTripList() {
  const list = el("trip-list");
  list.innerHTML = "";
  for (const trip of state.trips) {
    const row = document.createElement("li");
    row.className = "trip-row";
    const wideNote = trip.wide_spread ? '<span class="trip-wide">wide spread</span>' : "";
    row.innerHTML =
      `<span class="trip-title">${trip.title}</span>` +
      `<span class="trip-count">${trip.count} photos</span>${wideNote}`;
    row.addEventListener("click", () => startAlbum(trip));
    list.appendChild(row);
  }
}

async function startAlbum(trip) {
  state.printSize = el("print-size").value;
  setError("");
  setStatus("Creating album…");
  try {
    const created = await api("/api/albums", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: trip.title, print_size: state.printSize, uuids: trip.uuids }),
    });
    state.albumId = created.album_id;
    state.sequence = trip.uuids.slice();
  } catch (err) {
    setStatus("");
    setError(String(err));
    return;
  }

  state.cropNeeds = {};
  await Promise.all(state.sequence.map((uuid) => loadCropNeed(uuid)));

  setStatus("");
  el("sequencer-title").textContent = `${trip.title} — ${state.printSize}`;
  showSection("sequencer");
  renderSequence();
}

async function loadCropNeed(uuid) {
  try {
    state.cropNeeds[uuid] = await api(
      `/api/crop-preview/${uuid}?print_size=${encodeURIComponent(state.printSize)}`
    );
  } catch (err) {
    state.cropNeeds[uuid] = { needs_crop: false, offset_x: 0.5, offset_y: 0.5 };
  }
}

function backToTrips() {
  state.albumId = null;
  state.sequence = [];
  state.cropNeeds = {};
  setError("");
  showSection("trip-picker");
}

// --- export handoff --------------------------------------------------------------
//
// This button never calls anything that reaches the real Photos library -- it only
// makes sure the current sequence is saved, then prints the CLI command
// (`photocull albums export`, Task 8) that performs the actual export. Task 11 is
// this project's first-ever real `.export()` call, run from the CLI with the owner
// present; a browser button is not where that gate belongs.

async function exportAlbum() {
  await saveSequence();
  el("export-command").textContent = `photocull albums export --album-id ${state.albumId}`;
  showSection("export-result");
}

el("back-to-trips").addEventListener("click", backToTrips);
el("start-over").addEventListener("click", backToTrips);
el("export-button").addEventListener("click", exportAlbum);

state.token = new URLSearchParams(window.location.search).get("t");
loadTrips();
