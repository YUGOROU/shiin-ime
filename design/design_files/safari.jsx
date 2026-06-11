// safari.jsx — minimal Safari search UI as input context for the keyboard.
// Shows the URL/search bar with editable text + marked (underlined) preview.

function SafariMock({ dark, value, marked, accent, onTapField }) {
  const bg = dark ? '#000' : '#F2F2F7';
  const chromeBg = dark ? 'rgba(28,28,30,0.92)' : 'rgba(245,245,247,0.92)';
  const inputBg = dark ? 'rgba(118,118,128,0.24)' : 'rgba(118,118,128,0.12)';
  const text = dark ? '#fff' : '#000';
  const muted = dark ? 'rgba(235,235,245,0.55)' : 'rgba(60,60,67,0.55)';
  const sectionTitle = dark ? 'rgba(235,235,245,0.55)' : 'rgba(60,60,67,0.6)';
  const cardBg = dark ? '#1c1c1e' : '#fff';
  const border = dark ? 'rgba(84,84,88,0.45)' : 'rgba(60,60,67,0.12)';

  // favorites tiles
  const faves = [
    { l: 'A', label: 'Apple', c: '#000' },
    { l: 'G', label: 'Google', c: '#4285F4' },
    { l: 'Y', label: 'YouTube', c: '#FF0000' },
    { l: 'X', label: 'X', c: '#000' },
    { l: 'W', label: 'Wikipedia', c: '#888' },
    { l: '青', label: '青空文庫', c: '#3a86ff' },
    { l: 'N', label: 'NHK', c: '#003F7F' },
    { l: '読', label: '読売', c: '#c4282b' },
  ];

  // build display content with marked text inline
  const renderField = () => {
    if (!value && !marked) {
      return (
        <span style={{ color: muted, fontSize: 17, letterSpacing: -0.4 }}>
          Search or enter website
        </span>
      );
    }
    return (
      <span style={{ color: text, fontSize: 17, letterSpacing: -0.3, fontFamily: '-apple-system, system-ui' }}>
        {value}
        {marked && (
          <span style={{
            color: text,
            borderBottom: `1.5px solid ${accent}`,
            paddingBottom: 1,
          }}>{marked}</span>
        )}
        <span style={{
          display: 'inline-block', width: 2, height: 18,
          background: accent, marginLeft: 1, verticalAlign: '-3px',
          animation: 'shiinCaret 1s steps(2) infinite',
        }} />
      </span>
    );
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: bg, overflow: 'hidden' }}>
      {/* below status bar — start search results area */}
      <div style={{ height: 54 }} />

      <div style={{ flex: 1, padding: '0 16px', overflow: 'hidden' }}>
        {/* Favorites grid (peek under the search bar) */}
        <div style={{
          fontFamily: '-apple-system, system-ui', fontSize: 13, fontWeight: 600,
          color: sectionTitle, letterSpacing: -0.08, textTransform: 'none',
          padding: '4px 4px 10px',
        }}>Favorites</div>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px 10px',
        }}>
          {faves.map((f) => (
            <div key={f.label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
              <div style={{
                width: 60, height: 60, borderRadius: 14, background: cardBg,
                border: `0.5px solid ${border}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: '-apple-system, system-ui', fontWeight: 600, fontSize: 26,
                color: f.c,
              }}>{f.l}</div>
              <div style={{
                fontFamily: '-apple-system, system-ui', fontSize: 11.5,
                color: text, letterSpacing: -0.1,
              }}>{f.label}</div>
            </div>
          ))}
        </div>

        {/* Privacy report card */}
        <div style={{
          marginTop: 20, padding: '12px 14px',
          background: cardBg, borderRadius: 14, border: `0.5px solid ${border}`,
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <div style={{
            width: 28, height: 28, borderRadius: 7,
            background: dark ? 'rgba(120,120,128,0.24)' : 'rgba(120,120,128,0.16)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: '-apple-system, system-ui', fontWeight: 600,
            color: muted, fontSize: 14,
          }}>!</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontFamily: '-apple-system, system-ui', fontSize: 15, fontWeight: 500, color: text, letterSpacing: -0.3 }}>
              Privacy Report
            </div>
            <div style={{ fontFamily: '-apple-system, system-ui', fontSize: 12.5, color: muted, letterSpacing: -0.1 }}>
              0 trackers prevented in the last 30 days
            </div>
          </div>
          <svg width="8" height="14" viewBox="0 0 8 14">
            <path d="M1 1l6 6-6 6" stroke={muted} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>

      {/* search bar (floating above keyboard) */}
      <div style={{
        background: chromeBg,
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        backdropFilter: 'blur(20px) saturate(180%)',
        borderTop: `0.5px solid ${border}`,
        padding: '10px 12px 10px',
      }}>
        <div onClick={onTapField} style={{
          background: inputBg, borderRadius: 12, padding: '7px 12px 7px 10px',
          display: 'flex', alignItems: 'center', gap: 6, minHeight: 36,
        }}>
          {/* aA / lock icon */}
          <span style={{
            fontFamily: '-apple-system, system-ui', fontSize: 13, fontWeight: 600,
            color: muted, padding: '0 4px',
          }}>aA</span>
          <div style={{ flex: 1, minHeight: 22, display: 'flex', alignItems: 'center' }}>
            {renderField()}
          </div>
          {/* mic */}
          <svg width="14" height="20" viewBox="0 0 14 20">
            <rect x="4" y="2" width="6" height="11" rx="3" fill={muted}/>
            <path d="M1 9c0 3.3 2.7 6 6 6s6-2.7 6-6M7 15v3" stroke={muted} strokeWidth="1.6" fill="none" strokeLinecap="round"/>
          </svg>
        </div>
      </div>
    </div>
  );
}

window.SafariMock = SafariMock;
