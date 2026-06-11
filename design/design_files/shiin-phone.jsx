// shiin-phone.jsx — composes Safari + keyboard inside an iPhone frame.
// Each instance owns its own input state.

function ShiinPhone({ variant = 'a', dark = false, accent = '#7B5BFF', width = 393, height = 760, label }) {
  const [committed, setCommitted] = React.useState('');
  const [buffer, setBuffer] = React.useState('');

  const pred = React.useMemo(() => buffer ? window.ShiinPredict.lookup(buffer) : null, [buffer]);
  const candidates = pred ? pred.candidates : [];
  const reading = pred ? pred.reading : '';
  const topCandidate = candidates[0] || null;

  const onKey = (ch) => {
    if (!window.ShiinIsConsonant(ch)) {
      // vowel pressed — give a subtle nope (no buffer change)
      return;
    }
    setBuffer(b => (b + ch).slice(0, 24));
  };
  const onDelete = () => {
    if (buffer) setBuffer(b => b.slice(0, -1));
    else setCommitted(c => c.slice(0, -1));
  };
  const onSpace = () => {
    if (buffer && topCandidate) {
      setCommitted(c => c + topCandidate.text);
      setBuffer('');
    } else {
      setCommitted(c => c + ' ');
    }
  };
  const onReturn = () => {
    if (buffer && topCandidate) {
      setCommitted(c => c + topCandidate.text);
      setBuffer('');
    }
    // search action — flash nothing
  };
  const onGlobe = () => {};
  const onCandidate = (c) => {
    setCommitted(committed + c.text);
    setBuffer('');
  };
  const clear = () => { setCommitted(''); setBuffer(''); };

  // status bar tint
  return (
    <div style={{
      width, height, borderRadius: 48, overflow: 'hidden',
      position: 'relative',
      background: dark ? '#000' : '#F2F2F7',
      boxShadow: '0 30px 60px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.10)',
      fontFamily: '-apple-system, system-ui, sans-serif',
      WebkitFontSmoothing: 'antialiased',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* dynamic island */}
      <div style={{
        position: 'absolute', top: 11, left: '50%', transform: 'translateX(-50%)',
        width: 124, height: 36, borderRadius: 22, background: '#000', zIndex: 50,
      }} />
      {/* status bar */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10 }}>
        <IOSStatusBar dark={dark} />
      </div>
      {/* safari content */}
      <SafariMock
        dark={dark}
        value={committed}
        marked={buffer ? (topCandidate ? topCandidate.text : buffer) : ''}
        accent={accent}
        onTapField={clear}
      />
      {/* keyboard */}
      <ShiinKeyboard
        variant={variant}
        dark={dark}
        accent={accent}
        buffer={buffer}
        candidates={candidates}
        reading={reading}
        onKey={onKey}
        onSpace={onSpace}
        onDelete={onDelete}
        onReturn={onReturn}
        onGlobe={onGlobe}
        onCandidate={onCandidate}
      />
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

window.ShiinPhone = ShiinPhone;
