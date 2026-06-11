// predict.js — Fake on-device prediction for Shiin IME demo
// Maps consonant-only strings to a kana reading + ranked kanji candidates.
// Real Shiin IME uses a CoreML Transformer+GRU model; this is a curated
// lookup so the prototype demonstrates the UX faithfully.

// shape: 'ktk' -> { reading, candidates: [{text, score}], partial }
// score is 0..1, the model's softmax confidence

const DICT = {
  // single consonant — show many partials, low confidence
  'k': { reading: 'か', candidates: [
    { text: 'か', score: 0.42 },
    { text: 'が', score: 0.31 },
    { text: 'き', score: 0.18 },
    { text: 'こ', score: 0.09 },
  ]},
  'w': { reading: 'わ', candidates: [
    { text: 'わ', score: 0.54 },
    { text: 'を', score: 0.28 },
    { text: 'は', score: 0.18 },
  ]},

  // common words — high confidence
  'wtsh': { reading: 'わたし', candidates: [
    { text: '私', score: 0.91 },
    { text: 'わたし', score: 0.06 },
    { text: 'ワタシ', score: 0.02 },
    { text: '渡し', score: 0.01 },
  ]},
  'wt': { reading: 'わた', candidates: [
    { text: '私', score: 0.62 },
    { text: '綿', score: 0.18 },
    { text: '渡', score: 0.12 },
    { text: 'わた', score: 0.08 },
  ]},
  'ktk': { reading: 'きたく', candidates: [
    { text: '帰宅', score: 0.78 },
    { text: '北区', score: 0.14 },
    { text: '機宅', score: 0.05 },
    { text: '貴宅', score: 0.03 },
  ]},
  'kt': { reading: 'きた', candidates: [
    { text: '来た', score: 0.41 },
    { text: '北', score: 0.33 },
    { text: '聞いた', score: 0.16 },
    { text: 'きた', score: 0.10 },
  ]},
  'gnk': { reading: 'げんき', candidates: [
    { text: '元気', score: 0.94 },
    { text: '原器', score: 0.03 },
    { text: 'げんき', score: 0.03 },
  ]},
  'nhng': { reading: 'にほんご', candidates: [
    { text: '日本語', score: 0.97 },
    { text: 'にほんご', score: 0.02 },
    { text: 'ニホン語', score: 0.01 },
  ]},
  'nhn': { reading: 'にほん', candidates: [
    { text: '日本', score: 0.92 },
    { text: '二本', score: 0.05 },
    { text: 'にほん', score: 0.03 },
  ]},
  'nh': { reading: 'にほ', candidates: [
    { text: '日本', score: 0.68 },
    { text: '二歩', score: 0.14 },
    { text: 'にほ', score: 0.10 },
    { text: '荷物', score: 0.08 },
  ]},
  'yrsk': { reading: 'よろしく', candidates: [
    { text: 'よろしく', score: 0.55 },
    { text: '宜しく', score: 0.41 },
    { text: 'ヨロシク', score: 0.04 },
  ]},
  'knnch': { reading: 'こんにち', candidates: [
    { text: '今日', score: 0.83 },
    { text: 'こんにち', score: 0.12 },
    { text: '近日', score: 0.05 },
  ]},
  'knnchw': { reading: 'こんにちは', candidates: [
    { text: 'こんにちは', score: 0.96 },
    { text: '今日は', score: 0.03 },
    { text: 'コンニチハ', score: 0.01 },
  ]},
  'knbn': { reading: 'こんばん', candidates: [
    { text: '今晩', score: 0.81 },
    { text: 'こんばん', score: 0.13 },
    { text: '紺盤', score: 0.06 },
  ]},
  'knbnw': { reading: 'こんばんは', candidates: [
    { text: 'こんばんは', score: 0.95 },
    { text: '今晩は', score: 0.04 },
    { text: 'コンバンハ', score: 0.01 },
  ]},
  'rmj': { reading: 'ろまじ', candidates: [
    { text: 'ローマ字', score: 0.88 },
    { text: 'ろまじ', score: 0.09 },
    { text: '路間時', score: 0.03 },
  ]},
  'rgt': { reading: 'ありがと', candidates: [
    { text: 'ありがとう', score: 0.71 },
    { text: 'ありがと', score: 0.21 },
    { text: '有難う', score: 0.08 },
  ]},
  'rgtgzms': { reading: 'ありがとうございます', candidates: [
    { text: 'ありがとうございます', score: 0.93 },
    { text: '有難うございます', score: 0.05 },
    { text: '有り難うございます', score: 0.02 },
  ]},
  'gzms': { reading: 'ございます', candidates: [
    { text: 'ございます', score: 0.96 },
    { text: '御座います', score: 0.04 },
  ]},
  'skd': { reading: 'すきだ', candidates: [
    { text: '好きだ', score: 0.74 },
    { text: 'すきだ', score: 0.20 },
    { text: '隙だ', score: 0.06 },
  ]},
  'sk': { reading: 'すき', candidates: [
    { text: '好き', score: 0.69 },
    { text: 'すき', score: 0.15 },
    { text: '隙', score: 0.10 },
    { text: '鋤', score: 0.06 },
  ]},
  'tnsh': { reading: 'たのし', candidates: [
    { text: '楽し', score: 0.62 },
    { text: '楽しい', score: 0.31 },
    { text: 'たのし', score: 0.07 },
  ]},
  'tnshk': { reading: 'たのしく', candidates: [
    { text: '楽しく', score: 0.88 },
    { text: 'たのしく', score: 0.09 },
    { text: '愉しく', score: 0.03 },
  ]},
  'tnshkt': { reading: 'たのしかった', candidates: [
    { text: '楽しかった', score: 0.91 },
    { text: 'たのしかった', score: 0.06 },
    { text: '愉しかった', score: 0.03 },
  ]},
  // user-facing demo strings
  'shn': { reading: 'しん', candidates: [
    { text: '新', score: 0.39 },
    { text: '心', score: 0.21 },
    { text: '真', score: 0.18 },
    { text: '信', score: 0.13 },
    { text: 'しん', score: 0.09 },
  ]},
  'shnm': { reading: 'しんめ', candidates: [
    { text: '新芽', score: 0.71 },
    { text: '新目', score: 0.18 },
    { text: 'しんめ', score: 0.11 },
  ]},
  // single-consonant + a couple bridges so any combo gives some output
  's': { reading: 'さ', candidates: [
    { text: 'さ', score: 0.30 }, { text: 'し', score: 0.26 },
    { text: 'す', score: 0.20 }, { text: 'そ', score: 0.13 },
    { text: 'せ', score: 0.11 },
  ]},
  't': { reading: 'た', candidates: [
    { text: 'た', score: 0.34 }, { text: 'と', score: 0.27 },
    { text: 'て', score: 0.21 }, { text: 'ち', score: 0.18 },
  ]},
  'n': { reading: 'な', candidates: [
    { text: 'な', score: 0.31 }, { text: 'に', score: 0.27 },
    { text: 'の', score: 0.25 }, { text: 'ね', score: 0.17 },
  ]},
  'r': { reading: 'ら', candidates: [
    { text: 'ら', score: 0.30 }, { text: 'り', score: 0.26 },
    { text: 'る', score: 0.22 }, { text: 'れ', score: 0.13 },
    { text: 'ろ', score: 0.09 },
  ]},
  'h': { reading: 'は', candidates: [
    { text: 'は', score: 0.38 }, { text: 'ひ', score: 0.24 },
    { text: 'ほ', score: 0.18 }, { text: 'へ', score: 0.12 },
    { text: 'ふ', score: 0.08 },
  ]},
  'm': { reading: 'ま', candidates: [
    { text: 'ま', score: 0.32 }, { text: 'み', score: 0.26 },
    { text: 'も', score: 0.22 }, { text: 'め', score: 0.20 },
  ]},
  'y': { reading: 'や', candidates: [
    { text: 'や', score: 0.45 }, { text: 'よ', score: 0.31 },
    { text: 'ゆ', score: 0.24 },
  ]},
  'g': { reading: 'が', candidates: [
    { text: 'が', score: 0.30 }, { text: 'ぎ', score: 0.24 },
    { text: 'ご', score: 0.22 }, { text: 'げ', score: 0.14 },
    { text: 'ぐ', score: 0.10 },
  ]},
  // common partials
  'ny': { reading: 'にゃ', candidates: [
    { text: 'にゃ', score: 0.42 },
    { text: '入', score: 0.31 },
    { text: '荷', score: 0.27 },
  ]},
  'tsk': { reading: 'つき', candidates: [
    { text: '月', score: 0.58 },
    { text: '次', score: 0.20 },
    { text: '突き', score: 0.13 },
    { text: 'つき', score: 0.09 },
  ]},
  'krm': { reading: 'からだ', candidates: [
    { text: '体', score: 0.66 },
    { text: 'からだ', score: 0.21 },
    { text: '躯', score: 0.13 },
  ]},
  'mn': { reading: 'みな', candidates: [
    { text: 'みんな', score: 0.41 },
    { text: '皆', score: 0.36 },
    { text: 'みな', score: 0.14 },
    { text: '港', score: 0.09 },
  ]},
};

function lookup(buf) {
  if (!buf) return null;
  if (DICT[buf]) return DICT[buf];
  // fallback: progressively trim from the end and synthesize
  for (let i = buf.length - 1; i > 0; i--) {
    const head = buf.slice(0, i);
    if (DICT[head]) {
      const base = DICT[head];
      // synthesize "partial" continuation
      return {
        reading: base.reading + buf.slice(i),
        candidates: base.candidates.slice(0, 4).map((c, idx) => ({
          text: c.text + buf.slice(i),
          score: Math.max(0.05, c.score * Math.pow(0.55, buf.length - i) - idx * 0.02),
        })),
        partial: true,
      };
    }
  }
  // no match at all — show consonants as-is
  return {
    reading: buf,
    candidates: [
      { text: buf, score: 0.20 },
    ],
    partial: true,
  };
}

window.ShiinPredict = { lookup };
