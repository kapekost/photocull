/* The keyboard contract, in one place.
 *
 * This table is the *only* place a key is bound to a behaviour. The dispatcher looks
 * actions up here and the help overlay renders from the same array, so the overlay
 * cannot drift from what the keys actually do — the failure mode where a help screen
 * documents a shortcut that was renamed six commits ago is structurally unavailable.
 *
 * Each row carries two phrasings, and both are rendered from here. `description` is the
 * sentence the help overlay shows. `hint` is the one-word form the always-visible footer
 * shows, and `null` means exactly one thing — this key is not in the footer at all,
 * because it duplicates its neighbour (`←` and `↑`, each folded into its pair's shared
 * entry; `Tab`, which is an alias for `←`/`→`) or only exists inside the overlay it
 * closes (`Esc`). `footerLabel` overrides the key cap for that shared entry.
 *
 * **The arrows select the pane and `↑`/`↓` step the takes**, which is the reverse of
 * where they started. The owner found it on their own library: *"i can't move the
 * selection with the arrow and i need to click."* Stepping the challenger through the
 * other takes is an action that does not exist on the ~71% of clusters that are pairs,
 * so on nearly three clusters in four both arrow keys did nothing, while moving between
 * the two photographs — possible on 100% of clusters — sat on `Tab`, which nobody
 * guesses. `Tab` stays bound as an alias. Writing the footer as free prose was tried first and read as a
 * wall of text wrapping across two lines — the one piece of chrome on screen at all
 * times has to be scannable, not complete.
 *
 * `keys` are compared against `KeyboardEvent.key`. Letters are listed in both cases
 * rather than lower-cased at dispatch time, because lower-casing would silently bind
 * `Shift`ed variants to the same action and this table is meant to be read as the
 * literal truth about which keystrokes do what.
 */

export const KEYMAP = [
  {
    action: "focusKeeper",
    keys: ["ArrowLeft"],
    label: "←",
    description: "Select the proposed keeper, on the left",
    hint: null,
  },
  {
    action: "focusChallenger",
    keys: ["ArrowRight"],
    label: "→",
    description: "Select the challenger, on the right",
    hint: "pane",
    footerLabel: "← →",
  },
  {
    action: "previousTake",
    keys: ["ArrowUp"],
    label: "↑",
    description: "Previous take",
    hint: null,
  },
  {
    action: "nextTake",
    keys: ["ArrowDown"],
    label: "↓",
    description: "Next take",
    hint: "step",
    footerLabel: "↑ ↓",
  },
  {
    action: "switchPane",
    keys: ["Tab"],
    label: "Tab",
    description: "Move between the keeper and the challenger",
    hint: null,
  },
  {
    action: "toggleMagnify",
    keys: ["z", "Z"],
    label: "Z",
    description: "Magnify both takes to the limit of the local copy (Esc, or Z, to fit)",
    hint: "magnify",
  },
  {
    action: "toggleMark",
    keys: [" "],
    label: "Space",
    description: "Keep or cull the pane you are on",
    hint: "keep/cull",
  },
  {
    action: "cullAll",
    keys: ["c", "C"],
    label: "C",
    description: "Stage every take in this cluster (press again to put them back)",
    hint: "cull all",
  },
  {
    action: "toggleFavorite",
    keys: ["f", "F"],
    label: "F",
    description: "Mark favourite for write-back (nothing is written to Photos now)",
    hint: "favourite",
  },
  {
    action: "submit",
    keys: ["Enter"],
    label: "Enter",
    description: "Record this cluster and go to the next",
    hint: "record",
  },
  {
    action: "undo",
    keys: ["u", "U"],
    label: "U",
    description: "Undo the cluster you last recorded",
    hint: "undo",
  },
  {
    action: "toggleDashboard",
    keys: ["d", "D"],
    label: "D",
    description: "Where this review stands — counts, and the largest staged clusters",
    hint: "overview",
  },
  {
    action: "toggleFinalCheck",
    keys: ["v", "V"],
    label: "V",
    description: "The final check: every staged photo beside what it lost to",
    hint: "check",
  },
  {
    action: "toggleHelp",
    keys: ["?"],
    label: "?",
    description: "Show or hide this list",
    hint: "keys",
  },
  {
    action: "closeOverlay",
    keys: ["Escape"],
    label: "Esc",
    description: "Close whatever is open, or return a magnified pair to the fitted view",
    hint: null,
  },
];

/** The action bound to a `KeyboardEvent.key`, or `null` if the key is not ours. */
export function actionFor(key) {
  const entry = KEYMAP.find((row) => row.keys.includes(key));
  return entry ? entry.action : null;
}
