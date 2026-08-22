/* The two overview surfaces: the dashboard, and the final check.
 *
 * They live in their own module because they answer a different question from the
 * compare view. That view holds two photographs still beside each other; these two say
 * what has accumulated across thousands of clusters — "a quick overview of what's to be
 * deleted", which is what `staged-set-gets-dashboard-and-final-check` was asked for.
 *
 * **The final check is a gate, not a report.** Nothing reaches `Cull/Candidates` without
 * having been seen here, so the confirm button posts the digest of the set it is looking
 * at. If a decision landed since the screen was drawn, the server refuses rather than
 * confirming whatever is current — see `ReviewSession.confirm_final_check`.
 *
 * **The sheet is windowed, and that is a requirement rather than an optimisation.** At
 * the 0.48 cut the owner's real staged set is ~1,800-2,700 photos. Rendering it whole is
 * ~3,600 `<img>` elements and 1.26 MB of JSON, against a server that reads a file per
 * request, on a screen whose whole job is to be looked at before confirming. Rows arrive
 * a page at a time as the sentinel at the bottom scrolls into view: 60 rows and 41.6 KB,
 * opening in ~90 ms at 1,817 staged.
 *
 * `loading="lazy"` on top of that was measured rather than assumed, and it does less than
 * it looks: at 1440px the cards flow across the width, so one page is about two and a half
 * screens and Chromium fetches all 120 images either way. At one column the same page is
 * ~11,000px tall and it is the difference between **32 requests and 120** — which is the
 * honest division of labour. Windowing bounds the page; lazy bounds what a tall page
 * fetches before it is scrolled to.
 */

/** Staged rows per fetch. Comfortably more than one screen holds, so scrolling never
 *  waits on the network, and far less than the smallest real staged set. */
export const SHEET_PAGE = 60;

/** Plain language for a distance to the keeper, in the vocabulary
 *  `calibrate.write_contact_sheet` established and the owner reviewed 200 clusters in.
 *  Kept identical to `calibrate._similarity_words` deliberately: this screen and that
 *  one describe the same measurement, and two wordings for one number would read as two
 *  different facts. */
export function similarityWords(distance) {
  if (distance < 0.15) return "nearly identical";
  if (distance < 0.25) return "very similar";
  if (distance < 0.33) return "similar";
  return "loosest match in this group";
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** One line of the dashboard: a big number and what it counts. */
function tally(value, label) {
  const line = element("div", "tally");
  line.append(element("b", null, String(value)), element("span", null, ` ${label}`));
  return line;
}

export function createOverview(ctx) {
  const { nodes, getJson, postJson, imageUrl } = ctx;

  //: What the sheet currently holds. Reset on every open, because it is a view of the
  //: staged set at the moment it was asked for — and the confirmation below it names
  //: exactly that set.
  let sheet = { offset: 0, total: 0, digest: "", loading: false, observer: null };

  // --- the dashboard ----------------------------------------------------------------

  async function openDashboard() {
    // Re-read on every open rather than caching: a dashboard that answered with the
    // numbers from ten clusters ago would be wrong in the one direction the owner
    // cannot check, since knowing the true number is why they opened it.
    const data = await getJson("/api/dashboard");
    renderDashboard(data);
    nodes.dashboard.hidden = false;
  }

  function renderDashboard(data) {
    const body = nodes.dashboardBody;
    body.replaceChildren();

    const clusters = element("div", "tallies");
    clusters.append(
      tally(data.clusters.decided, "clusters reviewed"),
      tally(data.clusters.remaining, "to go"),
      tally(data.photos.keep, "photos keeping"),
      tally(data.photos.cull, "photos staging")
    );
    body.append(clusters);

    const undecided = element(
      "p",
      "dim",
      data.next_undecided
        ? `Next undecided: cluster ${data.next_undecided.index + 1}.`
        : "Every cluster in this session has been decided."
    );
    body.append(undecided);

    if (data.largest_staged.length) {
      body.append(element("h3", null, "Largest staged clusters"));
      const list = element("ul", "largest");
      for (const row of data.largest_staged) {
        const index = ctx.indexOf(row.cluster_key);
        const item = element("li");
        const jump = element("button", "jump");
        jump.type = "button";
        jump.textContent =
          `Cluster ${index + 1} · ${row.size} takes · ` +
          `staging ${row.staged}, keeping ${row.kept}`;
        jump.addEventListener("click", () => {
          closeDashboard();
          ctx.onJump(index);
        });
        item.append(jump);
        list.append(item);
      }
      body.append(list);
    } else {
      body.append(
        element("p", "dim", "Nothing is staged yet — decide a cluster and it appears here.")
      );
    }
  }

  function closeDashboard() {
    nodes.dashboard.hidden = true;
  }

  // --- the final check ---------------------------------------------------------------

  async function openSheet() {
    sheet = { offset: 0, total: 0, digest: "", loading: false, observer: sheet.observer };
    nodes.sheetRows.replaceChildren();
    // Shown before the rows arrive, with the wait said out loud. On the real staged set
    // the first page is a round trip, and a screen that stays blank until it lands reads
    // as a key that did nothing — the same complaint `Z` answers with a message.
    nodes.sheetHead.textContent = "Reading what is staged…";
    nodes.sheetLock.textContent = "";
    nodes.sheetConfirm.hidden = true;
    nodes.sheet.dataset.loading = "true";
    nodes.sheet.hidden = false;
    nodes.sheetScroll.scrollTop = 0;
    await loadPage();
    nodes.sheet.dataset.loading = "false";
    watchForMore();
  }

  function closeSheet() {
    nodes.sheet.hidden = true;
    if (sheet.observer) sheet.observer.disconnect();
    sheet.observer = null;
  }

  async function loadPage() {
    if (sheet.loading) return;
    if (sheet.offset && sheet.offset >= sheet.total) return;
    sheet.loading = true;
    try {
      const data = await getJson(
        `/api/final-check?offset=${sheet.offset}&limit=${SHEET_PAGE}`
      );
      sheet.total = data.total;
      sheet.digest = data.staged_digest;
      sheet.offset += data.staged.length;
      appendRows(data.staged);
      renderHead(data.confirmed);
    } finally {
      sheet.loading = false;
    }
  }

  function watchForMore() {
    if (sheet.observer) sheet.observer.disconnect();
    sheet.observer = new IntersectionObserver(
      async (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        if (nodes.sheet.hidden) return;
        const before = sheet.offset;
        await loadPage();
        // Nothing new arrived and nothing is left: stop asking. Without this the
        // sentinel sits visible at the end of a short sheet and re-fires on every
        // scroll for the rest of the session.
        if (sheet.offset === before && sheet.offset >= sheet.total) {
          sheet.observer.disconnect();
        }
      },
      { root: nodes.sheetScroll, rootMargin: "600px" }
    );
    sheet.observer.observe(nodes.sheetMore);
  }

  function renderHead(confirmed) {
    const clusters = new Set(
      [...nodes.sheetRows.querySelectorAll(".sheet-cluster")].map(
        (node) => node.dataset.clusterKey
      )
    );
    const all = sheet.offset >= sheet.total;
    nodes.sheetHead.textContent = sheet.total
      ? `${sheet.total} photos staged for culling, from ${clusters.size} clusters` +
        (all ? "" : ` · showing the first ${countStaged()}`)
      : "Nothing is staged yet.";
    nodes.sheet.dataset.confirmed = confirmed ? "true" : "false";
    // Three states, not two. With nothing staged there is nothing to confirm, and
    // "write-back is not available until you confirm this screen" would send the owner
    // looking for a button that is deliberately absent.
    nodes.sheetLock.textContent = !sheet.total
      ? "Nothing is staged, so there is nothing to write back."
      : confirmed
        ? `Write-back is unlocked for exactly these ${sheet.total} photos.` +
          " Changing any decision locks it again."
        : "Write-back is not available until you confirm this screen.";
    nodes.sheetNote.hidden = !sheet.total;
    nodes.sheetConfirm.hidden = confirmed || !sheet.total;
    nodes.sheetMore.textContent =
      sheet.offset >= sheet.total ? "" : "Scroll for more…";
  }

  /** Rows still staged — a photo pulled back to KEEP stays on screen so the click can be
   *  undone, but it is no longer one of the photos this screen is about to authorise. */
  const countStaged = () =>
    nodes.sheetRows.querySelectorAll('.staged:not([data-restored="true"])').length;

  function appendRows(rows) {
    for (const row of rows) {
      let section = nodes.sheetRows.lastElementChild;
      // A cluster can straddle two pages, so the section is reused rather than
      // reopened — otherwise the 30-take cluster would appear twice under two headings
      // and the owner would count it twice.
      if (!section || section.dataset.clusterKey !== row.cluster_key) {
        section = openSection(row);
        nodes.sheetRows.append(section);
      }
      section.querySelector(".sheet-grid").append(stagedFigure(row));
    }
  }

  function openSection(row) {
    const section = element("section", "sheet-cluster");
    section.dataset.clusterKey = row.cluster_key;
    const index = ctx.indexOf(row.cluster_key);

    const head = element("h3");
    const jump = element("button", "jump");
    jump.type = "button";
    jump.textContent = `Cluster ${index + 1}`;
    jump.addEventListener("click", () => {
      closeSheet();
      ctx.onJump(index);
    });
    head.append(jump);
    section.append(head);

    const grid = element("div", "sheet-grid");
    if (row.keepers.length) {
      for (const keeper of row.keepers) grid.append(keeperFigure(keeper));
    } else {
      // `a-whole-cluster-may-be-culled` makes this an ordinary outcome, and it is
      // exactly the row that deserves the second look: nothing survives this cluster.
      head.append(element("span", "no-keeper", " · no keeper — the whole cluster goes"));
    }
    section.append(grid);
    return section;
  }

  function figureFor(member, className) {
    const figure = element("figure", className);
    figure.dataset.uuid = member.uuid;
    const image = document.createElement("img");
    image.src = imageUrl(member.uuid);
    image.alt = "";
    image.loading = "lazy";
    figure.append(image);
    return figure;
  }

  function keeperFigure(member) {
    const figure = figureFor(member, "keeper");
    figure.append(element("figcaption", "verdict keep", "KEEP — what you kept"));
    return figure;
  }

  function stagedFigure(row) {
    const figure = figureFor(row.photo, "staged");
    figure.dataset.restored = "false";
    figure.tabIndex = 0;
    const caption = element("figcaption");
    caption.append(element("span", "verdict cull", "CULL"));
    const distance = row.photo.distance_to_winner;
    if (distance !== null && distance !== undefined) {
      caption.append(
        element(
          "span",
          "words",
          ` · ${similarityWords(distance)} to the keeper (${distance.toFixed(3)})`
        )
      );
    }
    // Empty until it has been restored: the instruction lives once in the strip above the
    // rows, and what belongs *per photo* is the exception — this one is no longer staged.
    caption.append(element("span", "restore", ""));
    figure.append(caption);
    figure.addEventListener("click", () => restore(figure, row));
    return figure;
  }

  async function restore(figure, row) {
    const staged = figure.dataset.restored === "true";
    const decision = ctx.decisionFor(row.cluster_key);
    if (!decision) return;
    const marks = { ...decision.marks, [row.photo.uuid]: staged ? "cull" : "keep" };
    // Re-staging a photo drops its favourite: `a-favourite-may-not-be-staged` refuses a
    // submission that carries both, and the owner would meet that refusal here as a
    // click that appears to do nothing.
    const favorites = (decision.favorites ?? []).filter(
      (uuid) => !(staged && uuid === row.photo.uuid)
    );
    const result = await postJson(
      `/api/clusters/${encodeURIComponent(row.cluster_key)}/decision`,
      { marks, favorites }
    );
    if (!result.ok) {
      ctx.say(result.data.detail ?? `The server refused that (${result.status}).`);
      return;
    }
    ctx.applyDecision(row.cluster_key, result.data);
    figure.dataset.restored = staged ? "false" : "true";
    // The badge says what the photo *is* now, not what it was when the page was drawn. A
    // row still reading CULL under a green "kept" line is the same two-sources-of-truth
    // shape the compare view avoids by rendering both panes from one mark set.
    const verdict = figure.querySelector(".verdict");
    verdict.textContent = staged ? "CULL" : "KEEP";
    verdict.className = staged ? "verdict cull" : "verdict keep";
    figure.querySelector(".restore").textContent = staged
      ? ""
      : "Kept — click to stage it again";
    await refreshHead();
  }

  /** Re-read the totals without re-rendering the rows: one restore changes the count and
   *  the digest, and the confirmation that names the old digest has just lapsed. */
  async function refreshHead() {
    const data = await getJson("/api/final-check?offset=0&limit=1");
    sheet.total = data.total;
    sheet.digest = data.staged_digest;
    renderHead(data.confirmed);
  }

  async function confirm() {
    const result = await postJson("/api/final-check/confirm", {
      staged_digest: sheet.digest,
    });
    if (!result.ok) {
      ctx.say(result.data.detail ?? `The server refused that (${result.status}).`);
      await refreshHead();
      return;
    }
    renderHead(true);
  }

  return {
    openDashboard,
    closeDashboard,
    openSheet,
    closeSheet,
    confirm,
    dashboardOpen: () => !nodes.dashboard.hidden,
    sheetOpen: () => !nodes.sheet.hidden,
  };
}
