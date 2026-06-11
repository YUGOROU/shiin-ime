// companion-app.jsx — Standalone iOS app screens (onboarding + settings)
// The keyboard extension lives inside this app per iOS requirements.

// ─────────────────────────────────────────────────────────
// Hero / Onboarding: explains the consonants-only concept
// ─────────────────────────────────────────────────────────
function CompanionHero({ dark, accent, width = 393, height = 760 }) {
  const bg = dark ? '#000' : '#fafafa';
  const text = dark ? '#fff' : '#0a0a0a';
  const sub = dark ? 'rgba(235,235,245,0.62)' : 'rgba(60,60,67,0.62)';
  const cardBg = dark ? '#1c1c1e' : '#fff';
  const border = dark ? 'rgba(84,84,88,0.4)' : 'rgba(60,60,67,0.10)';

  // Demonstration row: shows romaji → shiin transform
  const Row = ({ romaji, shiin, kana, kanji, keystrokesA, keystrokesB }) => (
    <div style={{
      padding: '14px 16px',
      display: 'flex', alignItems: 'center', gap: 12,
      borderBottom: `0.5px solid ${border}`,
    }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 5 }}>
        <div style={{
          fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
          fontSize: 13, letterSpacing: 0.6, color: sub,
        }}>
          <span style={{ textDecoration: 'line-through', textDecorationColor: sub, opacity: 0.55 }}>{romaji}</span>
          <span style={{ margin: '0 8px' }}>→</span>
          <span style={{ color: accent, fontWeight: 600 }}>{shiin}</span>
        </div>
        <div style={{
          fontFamily: '"Hiragino Sans", system-ui', fontSize: 19, color: text, letterSpacing: 0.3,
        }}>{kanji} <span style={{ color: sub, fontSize: 13, marginLeft: 6 }}>{kana}</span></div>
      </div>
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: 4,
        fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
      }}>
        <span style={{ color: sub, fontSize: 12, textDecoration: 'line-through' }}>{keystrokesA}</span>
        <span style={{ color: text, fontSize: 22, fontWeight: 600 }}>{keystrokesB}</span>
      </div>
    </div>
  );

  return (
    <div style={{
      width, height, borderRadius: 48, overflow: 'hidden',
      position: 'relative', background: bg,
      boxShadow: '0 30px 60px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.10)',
      fontFamily: '-apple-system, system-ui',
    }}>
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10 }}>
        <IOSStatusBar dark={dark} />
      </div>
      <div style={{
        position: 'absolute', top: 11, left: '50%', transform: 'translateX(-50%)',
        width: 124, height: 36, borderRadius: 22, background: '#000', zIndex: 50,
      }} />

      <div style={{ paddingTop: 78, height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* mark / logo */}
        <div style={{ padding: '0 24px', display: 'flex', alignItems: 'center', gap: 10 }}>
          <ShiinMark accent={accent} size={32} />
          <div style={{
            fontFamily: '-apple-system, system-ui', fontSize: 15, fontWeight: 600,
            color: text, letterSpacing: -0.1,
          }}>Shiin IME</div>
        </div>

        {/* Hero */}
        <div style={{ padding: '28px 24px 18px' }}>
          <div style={{
            fontFamily: '-apple-system, system-ui',
            fontSize: 36, fontWeight: 700, lineHeight: 1.05, color: text,
            letterSpacing: -1.2, textWrap: 'balance',
          }}>
            Type Japanese<br />
            <span style={{ color: accent }}>without the vowels.</span>
          </div>
          <div style={{
            marginTop: 14,
            fontFamily: '-apple-system, system-ui',
            fontSize: 15.5, lineHeight: 1.45, color: sub, letterSpacing: -0.2,
            textWrap: 'pretty', maxWidth: 320,
          }}>
            子音だけで打てる、オンデバイスAIキーボード。Transformer + GRU が裏で読みを当て、AzooKey が漢字に変換します。
          </div>
        </div>

        {/* Demo card */}
        <div style={{
          margin: '4px 16px 0',
          background: cardBg, borderRadius: 18,
          border: `0.5px solid ${border}`,
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '12px 16px 8px',
            fontFamily: '-apple-system, system-ui', fontSize: 12, fontWeight: 600,
            color: sub, textTransform: 'uppercase', letterSpacing: 0.8,
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <span>Keystrokes saved</span>
            <span style={{
              fontFamily: 'ui-monospace, "SF Mono", monospace',
              color: accent, fontWeight: 700, letterSpacing: 0.4,
            }}>−38%</span>
          </div>
          <Row romaji="watashi" shiin="wtsh" kana="わたし" kanji="私" keystrokesA="7" keystrokesB="4" />
          <Row romaji="kitaku"  shiin="ktk"  kana="きたく" kanji="帰宅" keystrokesA="6" keystrokesB="3" />
          <Row romaji="genki"   shiin="gnk"  kana="げんき" kanji="元気" keystrokesA="5" keystrokesB="3" />
          <div style={{
            padding: '10px 16px 12px',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            fontFamily: '-apple-system, system-ui', fontSize: 12, color: sub, letterSpacing: -0.1,
          }}>
            <span>Fully on-device · no network</span>
            <span style={{
              fontFamily: 'ui-monospace, "SF Mono", monospace',
              fontSize: 11, color: sub, padding: '3px 7px', borderRadius: 4,
              background: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)',
            }}>1.3M params</span>
          </div>
        </div>

        <div style={{ flex: 1 }} />

        {/* CTA */}
        <div style={{ padding: '0 16px 26px' }}>
          <button style={{
            width: '100%', height: 52, borderRadius: 14, border: 0,
            background: accent, color: '#fff', cursor: 'pointer',
            fontFamily: '-apple-system, system-ui', fontSize: 17, fontWeight: 600,
            letterSpacing: -0.2,
            boxShadow: `0 10px 24px ${withAlpha(accent, 0.32)}`,
          }}>Enable Shiin Keyboard</button>
          <div style={{
            marginTop: 12, textAlign: 'center',
            fontFamily: '-apple-system, system-ui', fontSize: 13, color: sub, letterSpacing: -0.1,
          }}>
            Settings → General → Keyboard → Keyboards
          </div>
        </div>
      </div>

      {/* home indicator */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 60,
        height: 26, display: 'flex', justifyContent: 'center', alignItems: 'flex-end',
        paddingBottom: 7, pointerEvents: 'none',
      }}>
        <div style={{
          width: 134, height: 5, borderRadius: 100,
          background: dark ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.28)',
        }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Settings screen
// ─────────────────────────────────────────────────────────
function CompanionSettings({ dark, accent, width = 393, height = 760 }) {
  const bg = dark ? '#000' : '#F2F2F7';
  const text = dark ? '#fff' : '#0a0a0a';
  const sub = dark ? 'rgba(235,235,245,0.62)' : 'rgba(60,60,67,0.6)';
  const cardBg = dark ? '#1c1c1e' : '#fff';
  const border = dark ? 'rgba(84,84,88,0.4)' : 'rgba(60,60,67,0.10)';

  const Row = ({ icon, title, detail, value, kind = 'chevron', isLast }) => (
    <div style={{
      display: 'flex', alignItems: 'center', minHeight: 48,
      padding: '0 14px', position: 'relative',
    }}>
      {icon && (
        <div style={{
          width: 28, height: 28, borderRadius: 7, background: icon.bg,
          marginRight: 12, flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#fff', fontWeight: 700, fontSize: 14,
        }}>{icon.glyph}</div>
      )}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: '-apple-system, system-ui', fontSize: 15.5, color: text, letterSpacing: -0.3 }}>{title}</div>
        {detail && <div style={{ fontFamily: '-apple-system, system-ui', fontSize: 12, color: sub, letterSpacing: -0.1 }}>{detail}</div>}
      </div>
      {kind === 'value' && value && (
        <div style={{ fontFamily: '-apple-system, system-ui', fontSize: 15, color: sub, marginRight: 6, letterSpacing: -0.2 }}>{value}</div>
      )}
      {kind === 'toggle' && (
        <div style={{
          width: 50, height: 30, borderRadius: 15,
          background: value ? accent : (dark ? '#39393D' : '#E9E9EA'),
          position: 'relative', transition: 'background .15s',
        }}>
          <div style={{
            position: 'absolute', top: 2, left: value ? 22 : 2,
            width: 26, height: 26, borderRadius: 13, background: '#fff',
            boxShadow: '0 2px 4px rgba(0,0,0,0.15), 0 0 0 0.5px rgba(0,0,0,0.05)',
            transition: 'left .15s',
          }} />
        </div>
      )}
      {kind === 'chevron' && (
        <svg width="8" height="14" viewBox="0 0 8 14" style={{ flexShrink: 0 }}>
          <path d="M1 1l6 6-6 6" stroke={sub} strokeOpacity={0.5} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )}
      {!isLast && (
        <div style={{
          position: 'absolute', bottom: 0, right: 0,
          left: icon ? 54 : 14, height: 0.5, background: border,
        }} />
      )}
    </div>
  );

  const SectionLabel = ({ children }) => (
    <div style={{
      fontFamily: '-apple-system, system-ui', fontSize: 12, fontWeight: 500,
      color: sub, textTransform: 'uppercase', letterSpacing: 0.5,
      padding: '24px 32px 8px',
    }}>{children}</div>
  );
  const Card = ({ children }) => (
    <div style={{
      background: cardBg, borderRadius: 14, margin: '0 16px',
      border: `0.5px solid ${border}`, overflow: 'hidden',
    }}>{children}</div>
  );

  return (
    <div style={{
      width, height, borderRadius: 48, overflow: 'hidden',
      position: 'relative', background: bg,
      boxShadow: '0 30px 60px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.10)',
      fontFamily: '-apple-system, system-ui',
    }}>
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10 }}>
        <IOSStatusBar dark={dark} />
      </div>
      <div style={{
        position: 'absolute', top: 11, left: '50%', transform: 'translateX(-50%)',
        width: 124, height: 36, borderRadius: 22, background: '#000', zIndex: 50,
      }} />

      <div style={{ paddingTop: 56, paddingBottom: 36, height: '100%', overflow: 'auto', boxSizing: 'border-box' }}>
        {/* large title */}
        <div style={{ padding: '8px 18px 12px' }}>
          <div style={{
            fontFamily: '-apple-system, system-ui', fontSize: 32, fontWeight: 700,
            color: text, letterSpacing: -0.6,
          }}>設定</div>
        </div>

        {/* Status pill */}
        <div style={{
          margin: '0 16px 6px',
          padding: '12px 14px',
          background: withAlpha(accent, 0.12),
          border: `0.5px solid ${withAlpha(accent, 0.30)}`,
          borderRadius: 14,
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8, background: accent,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <ShiinMark accent="#fff" size={18} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontFamily: '-apple-system, system-ui', fontSize: 14, fontWeight: 600, color: text, letterSpacing: -0.2 }}>
              キーボードは有効です
            </div>
            <div style={{ fontFamily: '-apple-system, system-ui', fontSize: 12, color: sub, letterSpacing: -0.1 }}>
              フルアクセス許可済 · 2/2 権限
            </div>
          </div>
          <div style={{
            fontFamily: 'ui-monospace, "SF Mono", monospace',
            fontSize: 11, color: accent, fontWeight: 700, letterSpacing: 0.6,
          }}>ON</div>
        </div>

        <SectionLabel>候補バー</SectionLabel>
        <Card>
          <Row title="表示スタイル" detail="候補の見せ方" value="信頼度バー" kind="value" />
          <Row title="最大候補数" value="6" kind="value" />
          <Row title="読みを併記する" kind="toggle" value={true} isLast />
        </Card>

        <SectionLabel>入力</SectionLabel>
        <Card>
          <Row title="母音キー" detail="使わない母音キーをディム / 非表示" value="ディム" kind="value" />
          <Row title="インラインプレビュー" detail="上位候補を下線付きで表示" kind="toggle" value={true} />
          <Row title="スペースで自動確定" kind="toggle" value={true} />
          <Row title="ハプティクス" kind="toggle" value={true} isLast />
        </Card>

        <SectionLabel>モデル</SectionLabel>
        <Card>
          <Row icon={{bg:'#34C759', glyph:'✓'}} title="端末内モデル" detail="v2.1 · 1.3M parameters · 4.8 MB" kind="value" value="最新" />
          <Row title="推論のデバウンス" value="60 ms" kind="value" />
          <Row title="診断情報" kind="chevron" isLast />
        </Card>

        <div style={{
          padding: '24px 24px 24px',
          fontFamily: '-apple-system, system-ui', fontSize: 11.5, color: sub,
          textAlign: 'center', letterSpacing: -0.05, lineHeight: 1.5,
        }}>
          Shiin IME 0.4.0 (β) · すべて端末内で動作<br/>
          キーストロークは一切外部に送信されません。
        </div>
      </div>

      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 60,
        height: 26, display: 'flex', justifyContent: 'center', alignItems: 'flex-end',
        paddingBottom: 7, pointerEvents: 'none',
      }}>
        <div style={{
          width: 134, height: 5, borderRadius: 100,
          background: dark ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.28)',
        }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Brand mark — abstract "consonant-only" glyph
// Three small circles (consonant beats) + a thin underline (the absent vowel)
// ─────────────────────────────────────────────────────────
function ShiinMark({ accent = '#7B5BFF', size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" style={{ display: 'block' }}>
      <rect x="0" y="0" width="32" height="32" rx="8" fill={accent} />
      {/* three "consonants" */}
      <circle cx="9" cy="14" r="2.6" fill="#fff" />
      <circle cx="16" cy="14" r="2.6" fill="#fff" />
      <circle cx="23" cy="14" r="2.6" fill="#fff" />
      {/* ghost vowel underline */}
      <rect x="6" y="22" width="20" height="1.6" rx="0.8" fill="#fff" fillOpacity="0.45" />
    </svg>
  );
}

window.CompanionHero = CompanionHero;
window.CompanionSettings = CompanionSettings;
window.ShiinMark = ShiinMark;
