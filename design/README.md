# Handoff: Shiin IME — Visual Design (Variant B · Confidence bars)

## Overview

Shiin IME is an iOS Custom Keyboard Extension that lets users type Japanese
using **consonants only** — no vowels required. A small on-device CoreML model
(Transformer + GRU, ~1.3M params) maps the consonant string to a hiragana
reading, then AzooKey converts the reading to ranked kanji candidates.

This handoff covers the **visual design polish pass** for:

1. **The keyboard surface** (candidate bar + buffer bar + key area) — adopted
   design is **Variant B (Confidence bars)**.
2. **The standalone companion app** — onboarding/hero screen + Settings
   screen (required by iOS because keyboard extensions must ship inside a
   container app).

The functional behavior, input pipeline, and structural layout were already
locked before this design session per `DESIGN_BRIEF.md` (also included). Only
the visual / interactional treatment is new.

---

## About the Design Files

The files in `design_files/` are **design references created in HTML/React** —
prototypes showing intended look and behavior, **not** production code to copy
directly into the iOS project.

The implementation task is to **recreate these designs natively in UIKit
(Swift)** as part of the existing `UIInputViewController`-based keyboard
extension and its companion app. Use the codebase's existing patterns,
typography helpers, and color tokens; promote the values listed in
[Design Tokens](#design-tokens) into Swift constants / asset catalog entries.

The HTML prototypes are interactive — you can open `design_files/index.html`
in a browser and actually type with the keyboard (mock prediction dictionary
in `predict.js`). Useful for confirming timing, marked-text behavior, the
"vowel keys are dimmed and inert" rule, etc.

---

## Fidelity

**High-fidelity.** Final colors, typography, spacing, and animation values
are intended to be implemented exactly as specified. Where the design
references generic system tokens (e.g., `UIColor.systemGray6`), prefer the
native iOS token to a hardcoded hex so Light/Dark and Dynamic Type are
automatic.

---

## Chosen Variant: B · Confidence Bars

Three candidate-bar treatments were explored. **Variant B** was adopted for
its information density and immediate scanability.

| Variant | Treatment | Status |
|---|---|---|
| A · Native | Plain text candidates; rank encoded only via font weight + opacity | rejected — too little affordance for the confidence signal |
| **B · Confidence** | **Kanji text + tiny accent-colored bar (proportional to score) beneath each candidate** | **ADOPTED** |
| C · Filled | Accent fills each candidate cell from the bottom proportional to score | rejected — too visually heavy for typing flow |

A and C remain in `index.html` (under section "他案 (参考)") for reference
only and should not be shipped.

---

## Screens / Views

There are three primary surfaces to implement:

### 1. Keyboard Extension View (UIInputViewController)

**Purpose:** Active typing surface. Replaces the system keyboard when the user
selects Shiin IME from the globe key.

**Total height:** The keyboard view fills the standard iOS keyboard frame
(~291pt on iPhone). All metrics below are in iOS points (pt).

**Vertical stack (top → bottom):**

| # | Section            | Height | Notes |
|---|--------------------|--------|-------|
| 1 | **Candidate bar**  | 50 pt  | Horizontally scrollable. Layout described below. |
| 2 | **Buffer bar**     | 22 pt  | Thin separator strip. Hairline border top + bottom. |
| 3 | **Key area**       | ~210 pt | Four rows of keys + bottom inset. |

Background of the whole keyboard:
- Light: `#D1D4DB`
- Dark:  `#272727`
- Top border: `0.5pt` hairline at `rgba(0,0,0,0.08)` (light) / `rgba(255,255,255,0.08)` (dark)

---

#### 1a. Candidate Bar (Variant B)

Horizontal `UIScrollView` (or `UICollectionView` flow layout, horizontal),
showing up to 6 candidate cells.

**Cell layout (per candidate):**
- Vertical stack, centered.
- Top: kanji/kana text.
  - Font: SF Pro Text (or system) **18 pt**; first candidate weight `Medium (500)`, the rest `Regular (400)`.
  - Color: `#111` (light) / `#FFF` (dark).
- Bottom: confidence bar.
  - **Track:** `3 pt` tall, fully rounded, background `rgba(0,0,0,0.08)` light / `rgba(255,255,255,0.10)` dark.
  - **Track width:** `min(76, cellMinWidth - 18)` pt, so roughly 46pt for narrow cells, ~68pt for the wider top cell.
  - **Fill:** accent color, width = `score × 100%`, animated `width` transition `180ms ease`.
- Vertical gap between text and bar: **4 pt**.

**Cell sizing:**
- Top (rank 0) cell `min-width: 86 pt`, all others `min-width: 64 pt`.
- Horizontal padding inside cell: `12 pt` left/right, `6 pt`/`8 pt` top/bottom.
- Cells separated by a `0.5 pt × 28 pt` vertical hairline divider
  (`rgba(0,0,0,0.08)` / `rgba(255,255,255,0.08)`), vertically centered.

**Empty state** (buffer empty, no candidates):
- Single row showing placeholder text "`type any consonant — 子音だけでOK`"
- Font: `SF Mono 12pt`, color: `rgba(60,60,67,0.45)` / `rgba(235,235,245,0.45)`
- Padding: `0 16pt`.

**Interaction:**
- Tap a cell → commit that candidate's text into the host text field, clear buffer.
- Cells should briefly highlight on touch (standard `UIButton.highlighted`).

---

#### 1b. Buffer Bar

A thin, monospace strip showing the raw consonant buffer.

- Height: **22 pt**, padding `0 16pt`.
- Border-top + border-bottom hairlines `0.5pt rgba(0,0,0,0.06)` / `rgba(255,255,255,0.08)`.
- Hidden visually (transparent text, structure remains for layout) when buffer is empty.

**Content (left → right):**
1. A small **▶** glyph in the accent color, weight 600.
2. The buffer string in **SF Mono 11.5pt** at `rgba(0,0,0,0.78)` / `rgba(255,255,255,0.85)`. Letter-spacing `1.4pt` (extra tracking so consonants read as discrete beats).
3. Right-aligned: the predicted hiragana reading in `Hiragino Sans 11pt`, color `rgba(60,60,67,0.5)` / `rgba(235,235,245,0.45)`, letter-spacing `0.6pt`.

Example, when buffer is `ktk` and the model predicts reading `きたく`:

```
▶ k t k                                    きたく
```

---

#### 1c. Key Area

Four rows of keys. Padding around the area: `8pt 3pt 4pt 3pt`. Row gap: **11 pt**.

| Row | Keys                                              | Layout |
|-----|---------------------------------------------------|--------|
| 1   | `q w e r t y u i o p`                             | 10 equal-flex keys, gap `6pt` |
| 2   | `a s d f g h j k l`                               | 9 equal-flex keys, gap `6pt`, row inset **`18pt`** L/R |
| 3   | `z x c v b n m  ⌫`                                | 18pt left spacer, 7 equal-flex letter keys, gap `6pt`, then `⌫` width **46pt** |
| 4   | `🌐  ‹space›  return`                              | `🌐` width 46pt, space flex, `return` width **82pt** |

**Letter key (standard):**
- Height: **42 pt**.
- Corner radius: **5 pt**.
- Background: `#FFFFFF` (light) / `#6A6A6E` (dark).
- Shadow: `0 1pt 0 rgba(0,0,0,0.22)` (light) / `0 1pt 0 rgba(0,0,0,0.45)` (dark).
- Font: SF Pro **22 pt**, weight `Regular (400)`, color `#000` / `#FFF`.
- Text-transform: lowercase glyph.
- Touch-down press feedback: `transform: scale(0.96)` for ~60ms.

**Vowel keys (`a e i o u`):**
- Same key chrome.
- Glyph is rendered **italic, weight 300**, color `rgba(0,0,0,0.42)` / `rgba(255,255,255,0.55)`.
- Whole key dimmed via `opacity: 0.42`.
- **Tapping a vowel key is a no-op** — the input pipeline must filter out
  vowel inputs entirely (no buffer mutation, no haptic, no preview update).
  This is the most important behavior detail for the consonants-only brand.

**Special keys:**
- `⌫` (delete): background `#ABB0BC` (light) / `#48484A` (dark). SVG glyph rendered in the foreground color.
- `🌐` (globe): same background as `⌫`. Standard iOS globe glyph (3 latitude lines + ellipse) — use SF Symbol `globe` at point size 22.
- `space`:
  - Background same as letter keys (`#FFF` / `#6A6A6E`).
  - Label: `space` when buffer is empty; `確定 · space` (= "commit · space") when buffer is non-empty.
  - Label font: SF Pro **14pt**, weight 400, color `rgba(0,0,0,0.75)` / `rgba(255,255,255,0.85)`.
- `return`:
  - Background: **accent color** (e.g. `#7B5BFF`).
  - Label: `return`, white, SF Pro 14pt weight 500.
  - Width 82pt.

---

### 2. Companion App — Hero / Onboarding Screen

**Purpose:** First-launch screen of the standalone container app. Sells the
"consonants only" idea and links to Settings → Keyboards for enabling the
extension.

**Layout (top → bottom inside the safe area):**

1. **Brand row** — `ShiinMark` (32pt purple rounded square logo) + wordmark "Shiin IME" (SF Pro 15pt weight 600).
2. **Hero headline** (32pt margin top):
   - "Type Japanese" (line 1, color `#0A0A0A` / `#FFF`)
   - "without the vowels." (line 2, **accent color**)
   - Font: SF Pro **36pt, weight 700, line-height 1.05, letter-spacing -1.2pt**, `text-wrap: balance`.
3. **Subtitle** (14pt below headline):
   - Japanese explanation, SF Pro 15.5pt, line-height 1.45, color `rgba(60,60,67,0.62)`.
   - Copy: 「子音だけで打てる、オンデバイスAIキーボード。Transformer + GRU が裏で読みを当て、AzooKey が漢字に変換します。」
4. **Demo card** — rounded card (radius 18pt) showing 3 example transformations:
   - Header row: "KEYSTROKES SAVED" left, "−38%" right (accent, mono, bold).
   - Rows: each row shows romaji (strikethrough) → shiin (accent) above the
     resulting kanji + kana, with a struck-through old keystroke count next to
     the new count on the right.
   - Examples:
     - `watashi` → `wtsh` · 私 (わたし) · 7→4
     - `kitaku`  → `ktk`  · 帰宅 (きたく) · 6→3
     - `genki`   → `gnk`  · 元気 (げんき) · 5→3
   - Footer row: "Fully on-device · no network" left, "1.3M params" mono chip right.
5. **CTA button** (bottom 26pt inset):
   - Full-width, height 52pt, radius 14pt, accent background, white SF Pro 17pt weight 600 "Enable Shiin Keyboard".
   - Drop shadow: `0 10pt 24pt accentColor@32%`.
6. **Helper text** below CTA: "Settings → General → Keyboard → Keyboards" in SF Pro 13pt at 62% gray.

See `design_files/companion-app.jsx → CompanionHero` for exact pixel values.

---

### 3. Companion App — Settings Screen (Japanese)

**Purpose:** Configure the keyboard extension. Standard iOS grouped-list UI.

**Title:** 設定 (SF Pro 32pt weight 700, letter-spacing -0.6pt).

**Status pill (full-width card under title):**
- Background: `accent @ 12%`, border `accent @ 30%`, radius 14pt.
- Left icon: 30×30pt accent square containing ShiinMark in white.
- Title: 「キーボードは有効です」 (14pt, weight 600)
- Subtitle: 「フルアクセス許可済 · 2/2 権限」 (12pt at 60% gray)
- Right: "ON" in accent, SF Mono 11pt weight 700.

**Grouped sections (`UITableViewStyle.insetGrouped` style):**

1. **候補バー**
   - 表示スタイル — "信頼度バー" (value) — chevron
   - 最大候補数 — "6" (value) — chevron
   - 読みを併記する — toggle ON

2. **入力**
   - 母音キー — "ディム" (value, options: Dim / Hide) — chevron
   - インラインプレビュー (上位候補を下線付きで表示) — toggle ON
   - スペースで自動確定 — toggle ON
   - ハプティクス — toggle ON

3. **モデル**
   - 端末内モデル · v2.1 · 1.3M parameters · 4.8 MB — green ✓ badge — "最新" (value)
   - 推論のデバウンス — "60 ms" (value) — chevron
   - 診断情報 — chevron

**Footer text** (centered, 11.5pt at 60% gray, line-height 1.5):

> Shiin IME 0.4.0 (β) · すべて端末内で動作  
> キーストロークは一切外部に送信されません。

---

## Interactions & Behavior

(Reiterating the brief; included here so the developer has everything in one
place.)

| Action                              | Result                                         |
|-------------------------------------|------------------------------------------------|
| Tap a **consonant** key             | Append to buffer; run inference on background queue |
| Tap a **vowel** key                 | **No-op.** No haptic, no buffer mutation, no preview update |
| Tap a **candidate**                 | Commit candidate's text to document; clear buffer; clear marked-text |
| Press **`space`** (buffer present)  | Commit top candidate, then clear buffer |
| Press **`space`** (buffer empty)    | Insert a literal space character |
| Press **`⌫`** (buffer present)      | Remove last consonant from buffer; re-run inference |
| Press **`⌫`** (buffer empty)        | Delete the previous document character (`deleteBackward`) |
| Press **`return`** (buffer present) | Commit top candidate, then `insertText("\n")` |
| Press **`return`** (buffer empty)   | `insertText("\n")` |
| Press **🌐**                         | `advanceToNextInputMode()` |

**Inline marked text:** While the buffer is non-empty, the top candidate
text MUST be set as the input field's marked text (underlined preview),
following standard `UITextInput` marked-text protocol. The underline is
drawn natively by the host app; do not try to render your own underline in
the candidate bar's preview.

**Inference debouncing:** Default 60ms after the last keystroke. Inference
runs on a background thread; the candidate bar updates on the main thread.

**Press animation:** Every key uses a brief touch-down `scale 0.96` for
~60ms (standard iOS press). Letter keys flash to a slightly darker
background on press (iOS standard — use `UIControl.State.highlighted`).

---

## State Management

The keyboard extension's controller (`UIInputViewController` subclass) owns:

```swift
struct ShiinState {
    var buffer: String                  // e.g. "ktk"
    var reading: String                 // e.g. "きたく", from CoreML
    var candidates: [Candidate]         // ranked kanji candidates from AzooKey
    var topCandidate: Candidate? { candidates.first }
}

struct Candidate {
    let text: String        // e.g. "帰宅"
    let score: Float        // 0...1, softmax confidence from the model
}
```

State transitions: see [Interactions & Behavior](#interactions--behavior).

The candidate-bar UICollectionView/UIScrollView should observe changes to
`candidates` and animate the bar-width changes (180ms ease).

---

## Design Tokens

### Color (semantic)

| Token              | Light                          | Dark                                |
|--------------------|--------------------------------|-------------------------------------|
| `keyboardBg`       | `#D1D4DB`                      | `#272727`                           |
| `letterKeyBg`      | `#FFFFFF`                      | `#6A6A6E`                           |
| `specialKeyBg`     | `#ABB0BC`                      | `#48484A`                           |
| `keyShadow`        | `rgba(0,0,0,0.22)`             | `rgba(0,0,0,0.45)`                  |
| `keyGlyph`         | `#000000`                      | `#FFFFFF`                           |
| `vowelGlyph`       | `rgba(0,0,0,0.42)`             | `rgba(255,255,255,0.55)`            |
| `candidateText`    | `#111111`                      | `#FFFFFF`                           |
| `candidateDivider` | `rgba(0,0,0,0.08)`             | `rgba(255,255,255,0.08)`            |
| `confidenceTrack`  | `rgba(0,0,0,0.08)`             | `rgba(255,255,255,0.10)`            |
| `bufferText`       | `rgba(0,0,0,0.78)`             | `rgba(255,255,255,0.85)`            |
| `bufferReading`    | `rgba(60,60,67,0.50)`          | `rgba(235,235,245,0.45)`            |
| `bufferBorder`     | `rgba(0,0,0,0.06)`             | `rgba(255,255,255,0.08)`            |
| `accent` (brand)   | `#7B5BFF`                      | `#7B5BFF`                           |
| `appBg`            | `#F2F2F7` (system grouped)     | `#000000`                           |
| `cardBg`           | `#FFFFFF`                      | `#1C1C1E`                           |

### Accent palette (user-selectable; default first)

`#7B5BFF` · `#0A84FF` · `#FF375F` · `#34C759` · `#FF9F0A` · `#000000`

When `accent = #000000` on dark mode, the `return` key glyph stays white;
the brand color is rendered with a thin white outline so it's still
discoverable. (Implement as `UIColor` resolution with `traitCollection`.)

### Spacing

- Row gap: 11pt
- Key gap: 6pt
- Row 2/3 indent: 18pt
- Side padding (key area): 3pt
- Candidate-bar cell horizontal padding: 12pt

### Radii

- Letter / special key: **5pt**
- Candidate-bar cell (variant B): no radius (text + bar are flush in cell)
- Settings cards: **14pt**
- Hero CTA button: **14pt**
- ShiinMark logo: **8pt**
- Companion-app demo card: **18pt**

### Typography

- **System / UI:** SF Pro Text (`-apple-system`).
- **Japanese:** Hiragino Sans (system Japanese — no custom font, per memory budget).
- **Monospace:** SF Mono (used for the buffer bar and metric chips only).

| Use                   | Family         | Size | Weight | Tracking |
|-----------------------|----------------|------|--------|----------|
| Settings large title  | SF Pro         | 32   | 700    | -0.6     |
| Hero headline         | SF Pro         | 36   | 700    | -1.2     |
| Hero subtitle / body  | SF Pro         | 15.5 | 400    | -0.2     |
| Row title             | SF Pro         | 15.5 | 400    | -0.3     |
| Section header        | SF Pro         | 12   | 500    | 0.5 (uppercase) |
| Candidate text        | Hiragino Sans  | 18   | 500/400 | 0.2     |
| Key glyph             | SF Pro         | 22   | 400    | 0        |
| Buffer (consonants)   | SF Mono        | 11.5 | 400    | 1.4      |
| Buffer (reading)      | Hiragino Sans  | 11   | 400    | 0.6      |
| Mono chip / metric    | SF Mono        | 11   | 500    | 0.4      |

---

## Assets

- **No image or font assets** ship with the keyboard extension (memory budget).
- **ShiinMark** (the 32pt rounded purple square with three dots + a hairline) is rendered as an SVG/CAShapeLayer in code. See `companion-app.jsx → ShiinMark` for the geometry. Suggested implementation: draw with Core Graphics in a `UIView.draw(_:)` override, or as a `UIImage` generated once via `UIGraphicsImageRenderer`.
- **Glyphs** (⌫, 🌐): prefer SF Symbols (`delete.left`, `globe`) at appropriate point sizes.
- **Favorites tiles in Safari mock**: only used in the design prototype to give the keyboard context. **Ignore for implementation** — that's the host app's UI, not ours.

---

## Files

In `design_files/`:

| File                  | What it is |
|-----------------------|------------|
| `index.html`          | Design canvas entry point. Open in a browser to interact with all variants and read the brief card. |
| `predict.js`          | Mock consonant → reading → kanji dictionary used by the prototype. **For demo only** — replace with the real CoreML + AzooKey pipeline. |
| `keyboard.jsx`        | Candidate bar (A/B/C variants), buffer bar, key area. **Primary visual reference for the keyboard surface.** |
| `safari.jsx`          | Safari search-screen mock used as input context. Not part of the deliverable; for reference only. |
| `shiin-phone.jsx`     | Composes Safari + keyboard inside an iPhone frame with shared state. Shows the marked-text behavior in the search field. |
| `companion-app.jsx`   | Hero/onboarding screen, Settings screen, and the ShiinMark logo. |
| `ios-frame.jsx`       | Phone bezel + status bar (cosmetic only, do not implement). |

Top-level: `DESIGN_BRIEF.md` is the original PRD-style brief that scoped this
design session.

---

## Open Questions / Next Steps

These weren't addressed in this design pass and are good follow-ups:

1. **Haptics curve** — what intensity for consonant tap vs. candidate commit
   vs. vowel-tap-rejected? Recommend `.light` / `.medium` / none, respectively.
2. **Long-press behavior on candidates** — currently undefined. Possible:
   peek the full reading + alternate kanji forms.
3. **Buffer overflow** — the prototype caps the buffer at 24 chars. Confirm
   the production limit (related to model context length).
4. **Globe long-press** — should this still surface the iOS keyboard picker
   menu? (UIKit free behavior; just leave it.)
5. **Landscape & iPad** — explicitly out of scope per the brief. Note for
   later: the row-3 indent assumption (`18pt`) needs to scale.
