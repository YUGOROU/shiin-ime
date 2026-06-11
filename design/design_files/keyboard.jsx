// keyboard.jsx — ShiinKeyboard: candidate bar + buffer bar + key area.
// Variants A/B/C differ ONLY in the candidate bar treatment.
// Owns no input field state — parent passes buffer/candidates and key handlers.

const CONSONANTS = new Set(['q','w','r','t','y','p','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']);
const VOWELS = new Set(['a','e','i','o','u']);

// ─────────────────────────────────────────────────────────
// Candidate bar variations
// ─────────────────────────────────────────────────────────

function CandidatesA({ candidates, dark, accent, onTap }) {
  // Pure-iOS minimal: text only, top candidate bolder; confidence implied via opacity.
  const text = dark ? '#fff' : '#111';
  const sep = dark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.10)';
  if (!candidates.length) {
    return <CandidatesEmpty dark={dark} />;
  }
  // Show top N, sized by score? equal cells, but font-weight encodes ranking
  const top = candidates.slice(0, 6);
  return (
    <div style={{
      display: 'flex', alignItems: 'stretch', height: 44,
      overflowX: 'auto', overflowY: 'hidden',
    }}>
      {top.map((c, i) => (
        <React.Fragment key={i}>
          {i > 0 && <div style={{ width: 0.5, alignSelf: 'center', height: 22, background: sep }} />}
          <button onClick={() => onTap(c)} style={{
            border: 0, background: 'transparent', cursor: 'pointer',
            padding: '0 14px', minWidth: i === 0 ? 88 : 64,
            flex: i === 0 ? '0 0 auto' : '0 0 auto',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: '"Hiragino Sans", "Yu Gothic", -apple-system, system-ui',
            fontSize: 18,
            fontWeight: i === 0 ? 500 : 400,
            color: text,
            opacity: i === 0 ? 1 : (0.85 - i * 0.08),
            letterSpacing: 0.2,
          }}>{c.text}</button>
        </React.Fragment>
      ))}
    </div>
  );
}

function CandidatesB({ candidates, dark, accent, onTap }) {
  // Confidence bars: kanji + tiny accent bar under each cell, sized to score.
  const text = dark ? '#fff' : '#111';
  const sub = dark ? 'rgba(235,235,245,0.5)' : 'rgba(60,60,67,0.55)';
  const sep = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const trackBg = dark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)';

  if (!candidates.length) {
    return <CandidatesEmpty dark={dark} />;
  }
  const top = candidates.slice(0, 6);
  return (
    <div style={{
      display: 'flex', alignItems: 'stretch', height: 50,
      overflowX: 'auto', overflowY: 'hidden',
    }}>
      {top.map((c, i) => (
        <React.Fragment key={i}>
          {i > 0 && <div style={{ width: 0.5, alignSelf: 'center', height: 28, background: sep }} />}
          <button onClick={() => onTap(c)} style={{
            border: 0, background: 'transparent', cursor: 'pointer',
            padding: '6px 12px 8px', minWidth: i === 0 ? 86 : 64,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 4,
          }}>
            <div style={{
              fontFamily: '"Hiragino Sans", "Yu Gothic", -apple-system, system-ui',
              fontSize: 18, fontWeight: i === 0 ? 500 : 400, color: text, lineHeight: 1,
            }}>{c.text}</div>
            <div style={{
              width: Math.max(28, Math.min(76, (i === 0 ? 86 : 64) - 18)),
              height: 3, borderRadius: 2, background: trackBg, position: 'relative', overflow: 'hidden',
            }}>
              <div style={{
                position: 'absolute', left: 0, top: 0, bottom: 0,
                width: `${Math.round(c.score * 100)}%`, background: accent, borderRadius: 2,
                transition: 'width .18s ease',
              }} />
            </div>
          </button>
        </React.Fragment>
      ))}
    </div>
  );
}

function CandidatesC({ candidates, dark, accent, onTap }) {
  // Playful filled chips: accent fills cell from bottom up proportional to score.
  // Top candidate gets a slightly stronger fill + reading peek.
  const text = dark ? '#fff' : '#111';
  const sub = dark ? 'rgba(235,235,245,0.62)' : 'rgba(60,60,67,0.62)';
  const chipBg = dark ? 'rgba(255,255,255,0.07)' : 'rgba(118,118,128,0.10)';

  if (!candidates.length) {
    return <CandidatesEmpty dark={dark} />;
  }
  const top = candidates.slice(0, 6);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      height: 50, padding: '0 6px',
      overflowX: 'auto', overflowY: 'hidden',
    }}>
      {top.map((c, i) => {
        const pct = Math.round(c.score * 100);
        return (
          <button key={i} onClick={() => onTap(c)} style={{
            position: 'relative', overflow: 'hidden',
            border: 0, background: chipBg, borderRadius: 9,
            cursor: 'pointer', flex: '0 0 auto',
            minWidth: i === 0 ? 82 : 58, height: 38, padding: '0 14px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {/* fill from bottom */}
            <div style={{
              position: 'absolute', left: 0, right: 0, bottom: 0,
              height: `${pct}%`,
              background: i === 0
                ? `linear-gradient(180deg, ${withAlpha(accent, 0.30)}, ${withAlpha(accent, 0.55)})`
                : `linear-gradient(180deg, ${withAlpha(accent, 0.10)}, ${withAlpha(accent, 0.22)})`,
              transition: 'height .18s ease',
            }} />
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                fontFamily: '"Hiragino Sans", "Yu Gothic", -apple-system, system-ui',
                fontSize: 18, fontWeight: i === 0 ? 600 : 500, color: text, lineHeight: 1,
              }}>{c.text}</span>
              {i === 0 && (
                <span style={{
                  fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
                  fontSize: 10, fontWeight: 500, color: sub, letterSpacing: 0.4,
                  paddingTop: 2,
                }}>{pct}</span>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function CandidatesEmpty({ dark }) {
  const muted = dark ? 'rgba(235,235,245,0.45)' : 'rgba(60,60,67,0.45)';
  return (
    <div style={{
      height: 44, display: 'flex', alignItems: 'center', padding: '0 16px',
      fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
      fontSize: 12, color: muted, letterSpacing: 0.5,
    }}>type any consonant — 子音だけでOK</div>
  );
}

function withAlpha(hex, a) {
  // accept #rrggbb -> rgba()
  const m = hex.replace('#','');
  const r = parseInt(m.slice(0,2),16), g = parseInt(m.slice(2,4),16), b = parseInt(m.slice(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ─────────────────────────────────────────────────────────
// Buffer bar — monospace minimal
// ─────────────────────────────────────────────────────────
function BufferBar({ buffer, reading, dark, accent }) {
  const muted = dark ? 'rgba(235,235,245,0.45)' : 'rgba(60,60,67,0.5)';
  const text = dark ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.78)';
  const empty = !buffer;
  return (
    <div style={{
      height: 22, padding: '0 16px',
      display: 'flex', alignItems: 'center', gap: 8,
      fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
      fontSize: 11.5, letterSpacing: 0.4,
      color: empty ? 'transparent' : text,
      borderTop: `0.5px solid ${dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'}`,
      borderBottom: `0.5px solid ${dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'}`,
    }}>
      {!empty && (
        <>
          <span style={{ color: accent, fontWeight: 600 }}>▶</span>
          <span style={{ letterSpacing: 1.4 }}>{buffer}</span>
          <span style={{ color: muted, marginLeft: 'auto', fontFamily: '"Hiragino Sans", -apple-system, system-ui', fontSize: 11, letterSpacing: 0.6 }}>
            {reading}
          </span>
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Key area
// ─────────────────────────────────────────────────────────
function Key({ ch, special, dark, accent, onTap, wide, flex, content, dimmed }) {
  const keyBg = dark ? '#6a6a6e' : '#fff';
  const specialBg = dark ? '#48484a' : '#ABB0BC';
  const glyph = dark ? '#fff' : '#000';
  const isVowel = !!ch && VOWELS.has(ch);

  return (
    <button
      onPointerDown={(e) => { e.currentTarget.style.transform = 'scale(0.96)'; }}
      onPointerUp={(e) => { e.currentTarget.style.transform = ''; }}
      onPointerLeave={(e) => { e.currentTarget.style.transform = ''; }}
      onClick={onTap}
      style={{
        height: 42, borderRadius: 5,
        border: 0, padding: 0,
        background: special === 'return' ? accent : (special ? specialBg : keyBg),
        boxShadow: dark
          ? '0 1px 0 rgba(0,0,0,0.45)'
          : '0 1px 0 rgba(0,0,0,0.22)',
        flex: flex ? '1 1 0' : undefined,
        width: wide,
        minWidth: 0,
        cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: '-apple-system, "SF Pro", system-ui',
        fontSize: 22, fontWeight: 400,
        color: special === 'return' ? '#fff' : glyph,
        letterSpacing: 0,
        transition: 'transform 60ms, opacity 120ms',
        opacity: dimmed ? 0.42 : 1,
        userSelect: 'none', WebkitUserSelect: 'none',
        textTransform: 'lowercase',
      }}
    >
      {content ?? (isVowel
        ? <span style={{
            position: 'relative', display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ fontStyle: 'italic', fontWeight: 300, color: dark ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.42)' }}>{ch}</span>
          </span>
        : ch)}
    </button>
  );
}

function KeyArea({ dark, accent, onKey, onSpace, onDelete, onReturn, onGlobe, hasBuffer }) {
  const row1 = ['q','w','e','r','t','y','u','i','o','p'];
  const row2 = ['a','s','d','f','g','h','j','k','l'];
  const row3 = ['z','x','c','v','b','n','m'];

  const delIcon = (
    <svg width="22" height="16" viewBox="0 0 22 16">
      <path d="M6 1h13a2 2 0 012 2v10a2 2 0 01-2 2H6L1 8l5-7z" fill="none" stroke={dark ? '#fff' : '#000'} strokeWidth="1.6" strokeLinejoin="round"/>
      <path d="M10 5l5 5M15 5l-5 5" stroke={dark ? '#fff' : '#000'} strokeWidth="1.6" strokeLinecap="round"/>
    </svg>
  );
  const globeIcon = (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <circle cx="11" cy="11" r="9" stroke={dark?'#fff':'#000'} strokeWidth="1.4"/>
      <ellipse cx="11" cy="11" rx="4" ry="9" stroke={dark?'#fff':'#000'} strokeWidth="1.4"/>
      <line x1="2" y1="11" x2="20" y2="11" stroke={dark?'#fff':'#000'} strokeWidth="1.4"/>
    </svg>
  );

  return (
    <div style={{ padding: '8px 3px 4px', display: 'flex', flexDirection: 'column', gap: 11 }}>
      {/* row 1 */}
      <div style={{ display: 'flex', gap: 6 }}>
        {row1.map(l => <Key key={l} ch={l} dark={dark} accent={accent} onTap={() => onKey(l)} flex dimmed={VOWELS.has(l)} />)}
      </div>
      {/* row 2 — indented */}
      <div style={{ display: 'flex', gap: 6, padding: '0 18px' }}>
        {row2.map(l => <Key key={l} ch={l} dark={dark} accent={accent} onTap={() => onKey(l)} flex dimmed={VOWELS.has(l)} />)}
      </div>
      {/* row 3 */}
      <div style={{ display: 'flex', gap: 6 }}>
        {/* match indent of row2 so letters align */}
        <div style={{ width: 18 }} />
        {row3.map(l => <Key key={l} ch={l} dark={dark} accent={accent} onTap={() => onKey(l)} flex />)}
        <Key special="del" dark={dark} accent={accent} onTap={onDelete} content={delIcon} wide={46} />
      </div>
      {/* row 4 */}
      <div style={{ display: 'flex', gap: 6 }}>
        <Key special="globe" dark={dark} accent={accent} onTap={onGlobe} content={globeIcon} wide={46} />
        <Key special="space" dark={dark} accent={accent} onTap={onSpace} flex
             content={
               <span style={{
                 fontFamily: '-apple-system, system-ui', fontSize: 14, fontWeight: 400,
                 color: dark ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.75)',
                 letterSpacing: 0.3,
               }}>{hasBuffer ? '確定 · space' : 'space'}</span>
             } />
        <Key special="return" dark={dark} accent={accent} onTap={onReturn} wide={82}
             content={
               <span style={{
                 fontFamily: '-apple-system, system-ui', fontSize: 14, fontWeight: 500, color: '#fff',
               }}>return</span>
             } />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Main keyboard
// ─────────────────────────────────────────────────────────
function ShiinKeyboard({
  variant = 'a',
  dark = false,
  accent = '#7B5BFF',
  buffer = '',
  candidates = [],
  reading = '',
  onKey, onSpace, onDelete, onReturn, onGlobe, onCandidate,
}) {
  const kbBg = dark ? '#272727' : '#D1D4DB';

  const Candidates = variant === 'b' ? CandidatesB : variant === 'c' ? CandidatesC : CandidatesA;

  return (
    <div style={{
      background: kbBg,
      paddingBottom: 8,
      borderTop: dark ? '0.5px solid rgba(255,255,255,0.08)' : '0.5px solid rgba(0,0,0,0.08)',
    }}>
      <Candidates candidates={candidates} dark={dark} accent={accent} onTap={onCandidate} />
      <BufferBar buffer={buffer} reading={reading} dark={dark} accent={accent} />
      <KeyArea
        dark={dark} accent={accent}
        onKey={onKey} onSpace={onSpace} onDelete={onDelete} onReturn={onReturn} onGlobe={onGlobe}
        hasBuffer={!!buffer}
      />
    </div>
  );
}

window.ShiinKeyboard = ShiinKeyboard;
window.ShiinIsConsonant = (ch) => CONSONANTS.has(ch);
window.withAlpha = withAlpha;
