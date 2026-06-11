# Shiin IME — Design Brief

## What is Shiin IME?

Shiin IME (子音のみIME) is a Japanese input method for iOS that lets users type using **consonants only** — no vowels required.

Standard Japanese romanization (romaji) requires typing both consonants and vowels:
- "watashi" → わたし (6 keystrokes)

Shiin IME requires only the consonants:
- "wtsh" → わたし (4 keystrokes)

The goal is faster Japanese text input with fewer keystrokes, especially for users who know Japanese phonetics intuitively.

---

## How It Works

1. **User types consonants** on a standard QWERTY layout (e.g., `ktk`)
2. **An on-device AI model** (Transformer + GRU, ~1.3M parameters, CoreML) converts the consonant string to a hiragana/katakana reading (e.g., `きたく`)
3. **AzooKey** (an open-source Japanese conversion library) converts the reading to kanji candidates (e.g., `帰宅`, `機宅`, `気宅`)
4. **The user selects a candidate**, or presses space to confirm the top suggestion

The entire inference pipeline runs **fully on-device** with no network requests.

---

## Platform

- **iOS Custom Keyboard Extension** (UIInputViewController)
- Targets iPhone; iPad support is not a current priority
- Implemented in UIKit (Swift)
- Supports Light / Dark mode
- Supports all iOS text input contexts (messages, notes, search fields, etc.)

---

## Current UI Structure

The keyboard is composed of three vertical sections, top to bottom:

### 1. Candidate Bar (top)
- Horizontally scrollable row of conversion candidates
- Each candidate is a tappable button
- Candidates are full kanji words (e.g., `帰宅`, `気持ち`, `今日`)
- The top candidate is also previewed as marked (underlined) text directly in the input field
- Typically shows 3 candidates at a time; may show more depending on model output

### 2. Buffer Bar (middle)
- Thin bar showing the raw consonant input currently being composed
- Displayed as `▶ ktk` while the user is mid-word
- Clears when a candidate is confirmed

### 3. Key Area (bottom)
Four rows of keys:

| Row | Keys |
|-----|------|
| 1 | `q w e r t y u i o p` |
| 2 | `a s d f g h j k l` |
| 3 | `z x c v b n m ⌫` |
| 4 | `🌐` · `space` · `return` |

- Letter keys are equal-width within each row
- `space` fills remaining width in the bottom row
- `🌐` switches to the next system keyboard
- `⌫` deletes the last consonant from the buffer, or (if buffer is empty) deletes the last character from the document

---

## Input Behavior

| Action | Result |
|--------|--------|
| Type a consonant | Appended to buffer; inference runs immediately |
| Tap a candidate | Candidate committed to document; buffer clears |
| Press `space` (with buffer) | Top candidate committed |
| Press `space` (no buffer) | Space character inserted |
| Press `⌫` (with buffer) | Last consonant removed from buffer |
| Press `⌫` (empty buffer) | Document character deleted |
| Press `return` | Top candidate committed + newline inserted |

Inference is debounced and runs on a background thread. The top candidate is shown as iOS **marked text** (underlined preview) in the text field in real time as the user types.

---

## Constraints

- **Keyboard extensions have a strict memory limit** (~50 MB). The CoreML model must stay small (~1.3M parameters).
- No internet access from within a keyboard extension.
- The keyboard must work reliably across all iOS apps and text field types.
- No custom fonts or heavy assets — the extension binary must remain small.

---

## Current State

The AI model is currently being retrained (Phase 2) with a larger corpus and improved architecture. The iOS keyboard code is ready and will be updated to load the new model. The core UX flow described above is stable and will not fundamentally change.

---

## Context for This Session

The goal of this design session is to **visually polish the existing keyboard UI**. The functional behavior described above is fixed. The structure (candidate bar + buffer bar + key area) will remain the same.
