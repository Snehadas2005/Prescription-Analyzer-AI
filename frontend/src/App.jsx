import React, { useState, useCallback, useEffect, useRef } from 'react';
import { 
  Upload, User, Pill, Star, Activity, Stethoscope, 
  Loader2, FileText, Sparkles, ShieldCheck, Zap,
  Volume2, VolumeX, Globe, ChevronDown, Languages,
  X, Check, AlertCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

// ─── TTS utility ────────────────────────────────────────────────────────────
const speak = (text, lang = 'en-US') => {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = lang;
  utt.rate = 0.9;
  utt.pitch = 1;
  window.speechSynthesis.speak(utt);
};

const stopSpeaking = () => {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
};

const LANG_MAP = {
  en: { code: 'en-US', label: 'English', flag: '🇬🇧' },
  hi: { code: 'hi-IN', label: 'हिंदी',   flag: '🇮🇳' }
};

const TRANSLATIONS = {
  en: {
    brand: "VaidyaScan",
    badge: "Gemini 2.0 Flash Vision · Hindi + English",
    heroTitle: "Read any prescription.",
    heroAccent: "In any language.",
    heroSubtitle: "Upload a handwritten or printed prescription — Hindi, English, or mixed — and get structured data with TTS readback in seconds.",
    stateless: "Stateless & Private",
    fast: "Sub-3s Extraction",
    regional: "Hindi + Regional Scripts",
    voice: "Voice Readback",
    drop: "Drop prescription here",
    browse: "or click to browse — JPG, PNG, TIFF supported",
    pillEn: "English prescriptions",
    pillHi: "Hindi prescriptions",
    pillOk: "Handwritten OK",
    footer: "© 2026 VaidyaScan — Developed by Sneha Das · Powered by Gemini Vision"
  },
  hi: {
    brand: "वैद्यस्कैन",
    badge: "जेमिनी 2.0 फ्लैश विजन · हिंदी + अंग्रेजी",
    heroTitle: "किसी भी प्रिस्क्रिप्शन को पढ़ें।",
    heroAccent: "किसी भी भाषा में।",
    heroSubtitle: "हाथ से लिखा हुआ या मुद्रित प्रिस्क्रिप्शन अपलोड करें - हिंदी, अंग्रेजी या मिश्रित - और सेकंडों में टीटीएस रीडबैक के साथ संरचित डेटा प्राप्त करें।",
    stateless: "स्टेटलेस और निजी",
    fast: "3 सेकेंड से कम में एक्सट्रैक्शन",
    regional: "हिंदी + क्षेत्रीय लिपियाँ",
    voice: "वॉयस रीडबैक",
    drop: "यहाँ प्रिस्क्रिप्शन ड्रॉप करें",
    browse: "या ब्राउज़ करने के लिए क्लिक करें — JPG, PNG, TIFF समर्थित",
    pillEn: "अंग्रेजी प्रिस्क्रिप्शन",
    pillHi: "हिंदी प्रिस्क्रिप्शन",
    pillOk: "हस्तलिखित ठीक है",
    footer: "© 2026 वैद्यस्कैन — स्नेहा दास द्वारा विकसित · जेमिनी विजन द्वारा संचालित"
  }
};

// ─── Build TTS script from result ───────────────────────────────────────────
const buildTTSScript = (result, lang) => {
  const isHindi = lang === 'hi';

  if (isHindi) {
    let s = '';
    if (result.patient?.name) s += `मरीज़ का नाम: ${result.patient.name}. `;
    if (result.patient?.age)  s += `उम्र: ${result.patient.age} साल. `;
    if (result.doctor?.name)  s += `डॉक्टर: ${result.doctor.name}. `;
    if (result.diagnosis?.length) s += `निदान: ${result.diagnosis.join(', ')}. `;
    if (result.medicines?.length) {
      s += `दवाइयाँ: `;
      result.medicines.forEach((m, i) => {
        s += `${i+1}. ${m.name}. `;
        if (m.dosage)     s += `मात्रा: ${m.dosage}. `;
        if (m.frequency)  s += `आवृत्ति: ${m.frequency}. `;
        if (m.duration)   s += `अवधि: ${m.duration}. `;
      });
    }
    return s;
  }

  let s = '';
  if (result.patient?.name) s += `Patient: ${result.patient.name}. `;
  if (result.patient?.age)  s += `Age: ${result.patient.age}. `;
  if (result.doctor?.name)  s += `Doctor: ${result.doctor.name}. `;
  if (result.diagnosis?.length) s += `Diagnosis: ${result.diagnosis.join(', ')}. `;
  if (result.medicines?.length) {
    s += `Medications: `;
    result.medicines.forEach((m, i) => {
      s += `${i+1}. ${m.name}. `;
      if (m.dosage)    s += `Dosage: ${m.dosage}. `;
      if (m.frequency) s += `${m.frequency}. `;
      if (m.duration)  s += `For ${m.duration}. `;
    });
  }
  return s;
};

// ─── Inline SVG icon ────────────────────────────────────────────────────────
const HeartECG = ({ size = 28, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 28 28" fill="none">
    <path d="M14 24S4 18 4 10.5A5.5 5.5 0 0 1 14 7a5.5 5.5 0 0 1 10 3.5C24 18 14 24 14 24z"
      stroke={color} strokeWidth="1.8" fill="none" strokeLinejoin="round"/>
    <polyline points="6,13 9,13 11,9 13,17 15,13 17,13 19,13 21,13"
      stroke={color} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// ─── Noise texture overlay ───────────────────────────────────────────────────
const Noise = () => (
  <svg className="noise-overlay">
    <filter id="noise"><feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3"/><feColorMatrix type="saturate" values="0"/></filter>
    <rect width="100%" height="100%" filter="url(#noise)"/>
  </svg>
);

// ─── Language Selector ───────────────────────────────────────────────────────
const LangSelector = ({ value, onChange }) => {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position:'relative', userSelect:'none' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="lang-select-btn"
      >
        <Globe size={14}/> {LANG_MAP[value]?.flag} {LANG_MAP[value]?.label} <ChevronDown size={12}/>
      </button>
      {open && (
        <div style={{
          position:'absolute', top:'110%', right:0,
          background:'var(--white)', border:'1.5px solid var(--secondary)',
          borderRadius:'14px', padding:'6px', zIndex:100,
          minWidth:'160px', boxShadow:'var(--shadow-expensive)'
        }}>
          {Object.entries(LANG_MAP).map(([k,v]) => (
            <button key={k} onClick={() => { onChange(k); setOpen(false); }}
              style={{
                display:'flex', alignItems:'center', gap:'10px',
                width:'100%', padding:'9px 12px', borderRadius:'9px',
                border:'none', background: value===k ? 'var(--bg-secondary)' : 'transparent',
                color:'var(--dark)', fontSize:'13px', cursor:'pointer',
                fontFamily:'inherit', textAlign:'left'
              }}>
              {v.flag} {v.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── TTSButton ───────────────────────────────────────────────────────────────
const TTSButton = ({ result, lang }) => {
  const [speaking, setSpeaking] = useState(false);

  const toggle = () => {
    if (speaking) { stopSpeaking(); setSpeaking(false); return; }
    const script = buildTTSScript(result, lang);
    speak(script, LANG_MAP[lang]?.code || 'en-US');
    setSpeaking(true);
    const utt = window.speechSynthesis;
    const check = setInterval(() => {
      if (!utt.speaking) { clearInterval(check); setSpeaking(false); }
    }, 300);
  };

  return (
    <button 
      onClick={toggle} 
      title={speaking ? 'Stop reading' : 'Read aloud'}
      className={`tts-btn ${speaking ? 'active' : ''}`}
    >
      {speaking ? <VolumeX size={15}/> : <Volume2 size={15}/>}
      {speaking ? 'Stop' : 'Read Aloud'}
    </button>
  );
};

// ─── Drag-drop upload zone ───────────────────────────────────────────────────
const UploadZone = ({ onFile, lang }) => {
  const [dragging, setDragging] = useState(false);
  const ref = useRef();
  const t = TRANSLATIONS[lang] || TRANSLATIONS.en;

  const process = (f) => {
    if (f && f.type.startsWith('image/')) onFile(f);
  };

  return (
    <div
      ref={ref}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); process(e.dataTransfer.files[0]); }}
      onClick={() => document.getElementById('file-upload').click()}
      className={`upload-zone-wrapper ${dragging ? 'dragging' : ''}`}
    >
      <div className="upload-icon-box">
        <Upload size={28} />
      </div>
      <p style={{ color:'var(--dark)', fontWeight:'700', fontSize:'18px', marginBottom:'8px' }}>
        {t.drop}
      </p>
      <p style={{ color:'var(--text-secondary)', fontSize:'14px', marginBottom:'24px' }}>
        {t.browse}
      </p>
      <div style={{ display:'flex', justifyContent:'center', gap:'10px', flexWrap:'wrap' }}>
        {[t.pillEn, t.pillHi, t.pillOk].map(text => (
          <span key={text} className="feature-pill" style={{ padding: '6px 14px', fontSize: '12px' }}>{text}</span>
        ))}
      </div>
      <input type="file" id="file-upload" style={{ display:'none' }}
        accept="image/*" onChange={e => process(e.target.files[0])}/>
    </div>
  );
};

// ─── Confidence ring ─────────────────────────────────────────────────────────
const ConfRing = ({ value }) => {
  const pct = Math.round(value * 100);
  const r = 36, c = 2*Math.PI*r;
  const dash = c - (pct/100)*c;
  return (
    <div style={{ position:'relative', width:'96px', height:'96px' }}>
      <svg width="96" height="96" style={{ transform:'rotate(-90deg)' }}>
        <circle cx="48" cy="48" r={r} fill="none" stroke="var(--bg-secondary)" strokeWidth="8"/>
        <circle cx="48" cy="48" r={r} fill="none" stroke="var(--dark)" strokeWidth="8"
          strokeDasharray={c} strokeDashoffset={dash}
          style={{ transition:'stroke-dashoffset 1s ease', strokeLinecap:'round' }}/>
      </svg>
      <div style={{
        position:'absolute', inset:0, display:'flex', flexDirection:'column',
        alignItems:'center', justifyContent:'center'
      }}>
        <span style={{ fontSize:'20px', fontWeight:'900', color:'var(--dark)', lineHeight:1 }}>{pct}</span>
        <span style={{ fontSize:'10px', color:'var(--text-muted)', fontWeight:'600' }}>%</span>
      </div>
    </div>
  );
};

// ─── Field ───────────────────────────────────────────────────────────────────
const Field = ({ label, value, accent }) => (
  <div className="field-group">
    <div className="field-label">{label}</div>
    <div className={`field-value ${value ? (accent ? 'accent' : '') : 'empty'}`}>
      {value || '—'}
    </div>
  </div>
);

// ─── Medicine card ───────────────────────────────────────────────────────────
const MedCard = ({ med, idx }) => {
  const ref = useRef();
  useEffect(() => {
    gsap.fromTo(ref.current,
      { opacity:0, y:24 },
      { opacity:1, y:0, duration:0.5, delay:idx*0.07, ease:'power2.out' }
    );
  }, []);

  return (
    <div ref={ref} className="med-card">
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:'16px' }}>
        <h4 style={{ fontSize:'16px', fontWeight:'800', color:'var(--dark)', margin:0 }}>{med.name}</h4>
        <span style={{
          padding:'4px 12px', borderRadius:'20px',
          background:'var(--white)', color:'var(--dark)',
          fontSize:'11px', fontWeight:'800', border: '1.px solid var(--primary)'
        }}>#{idx+1}</span>
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'12px' }}>
        <Field label="Dosage"     value={med.dosage}/>
        <Field label="Frequency"  value={med.frequency}/>
        <Field label="Duration"   value={med.duration}/>
        <Field label="Timing"     value={med.timing}/>
      </div>
    </div>
  );
};


// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [phase, setPhase]           = useState('upload'); // upload | analyzing | results
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState(null);
  const [lang, setLang]             = useState('en');
  const [langDetected, setLangDetected] = useState(null);

  const headerRef  = useRef();
  const featRef    = useRef();
  const resultRef  = useRef();
  const particlesRef = useRef([]);

  const API_URL = `${import.meta.env.VITE_API_URL}/api/v1`;

  // ── Header entrance ──
  useEffect(() => {
    if (!headerRef.current) return;
    gsap.fromTo(
      [...headerRef.current.children],
      { opacity:0, y:-20 },
      { opacity:1, y:0, duration:0.7, stagger:0.12, ease:'power3.out', delay:0.1 }
    );
  }, []);

  // ── Feature pills entrance ──
  useEffect(() => {
    if (!featRef.current || phase !== 'upload') return;
    gsap.fromTo(
      [...featRef.current.children],
      { opacity:0, scale:0.85 },
      { opacity:1, scale:1, duration:0.4, stagger:0.08, ease:'back.out(1.4)', delay:0.6 }
    );
  }, [phase]);

  // ── Results entrance ──
  useEffect(() => {
    if (phase !== 'results' || !resultRef.current) return;
    gsap.fromTo(
      resultRef.current,
      { opacity:0, y:40 },
      { opacity:1, y:0, duration:0.65, ease:'power3.out' }
    );
  }, [phase]);

  // ── Detect Hindi in raw text ──
  const detectLang = (raw = '') => {
    const hindiChars = (raw.match(/[\u0900-\u097F]/g) || []).length;
    if (hindiChars > 20) { setLangDetected('hi'); setLang('hi'); }
    else { setLangDetected('en'); }
  };

  const handleFile = useCallback((f) => {
    setFile(f);
    setError(null);
    setResult(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(f);
  }, []);

  const analyze = async () => {
    if (!file) return;
    setPhase('analyzing');
    setError(null);

    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_URL}/upload`, { method:'POST', body:fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const data = json.data || json;
      if (json.success) {
        detectLang(data.raw_text || '');
        setResult(data);
        setTimeout(() => setPhase('results'), 800);
      } else {
        setError(json.error || 'Analysis failed');
        setPhase('upload');
      }
    } catch (e) {
      setError(e.message);
      setPhase('upload');
    }
  };

  const reset = () => {
    stopSpeaking();
    setFile(null); setPreview(null);
    setResult(null); setError(null);
    setLangDetected(null);
    setPhase('upload');
  };

  const t = TRANSLATIONS[lang] || TRANSLATIONS.en;

  return (
    <div className="app-container">
      <Noise/>

      {/* ── Header ── */}
      <header ref={headerRef} className="main-header">
        <div className="logo-group">
          <div className="logo-box">
            <HeartECG size={20} color="white"/>
          </div>
          <span className="logo-text">
            {t.brand.slice(0, -4)}<span className="logo-accent">{t.brand.slice(-4)}</span>
          </span>
        </div>

        <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
          <LangSelector value={lang} onChange={setLang}/>
          {langDetected && (
            <span className="feature-pill" style={{ borderColor: 'var(--primary)' }}>
              <Languages size={11} style={{ display:'inline', marginRight:4 }}/>
              {LANG_MAP[langDetected]?.label} detected
            </span>
          )}
        </div>
      </header>

      <main className="max-width-container">

        {/* ── UPLOAD PHASE ── */}
        <AnimatePresence mode="wait">
        {phase === 'upload' && (
          <motion.div key="upload"
            initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0, y:-20 }}>

            {/* Hero */}
            <div className="hero-section">
              <div className="hero-badge">
                {t.badge}
              </div>

              <h1 className="hero-title">
                {t.heroTitle}<br/>
                <span className="hero-gradient-text">{t.heroAccent}</span>
              </h1>

              <p className="hero-subtitle">
                {t.heroSubtitle}
              </p>

              {/* Feature pills */}
              <div ref={featRef} className="feature-pills">
                {[
                  { icon:<ShieldCheck size={14}/>, t: t.stateless },
                  { icon:<Zap size={14}/>,         t: t.fast },
                  { icon:<Globe size={14}/>,        t: t.regional },
                  { icon:<Volume2 size={14}/>,      t: t.voice },
                ].map(({ icon, t }) => (
                  <span key={t} className="feature-pill">{icon}{t}</span>
                ))}
              </div>
            </div>

            {/* Upload card */}
            <div className="upload-card">
              {!file ? (
                <UploadZone onFile={handleFile} lang={lang}/>
              ) : (
                <div>
                  <div style={{
                    position:'relative', marginBottom:'28px',
                    borderRadius:'16px', overflow:'hidden',
                    border:'1px solid var(--secondary)'
                  }}>
                    <img src={preview} alt="Preview"
                      style={{ width:'100%', maxHeight:'280px', objectFit:'cover', display:'block' }}/>
                    <button onClick={reset} style={{
                      position:'absolute', top:'10px', right:'10px',
                      width:'32px', height:'32px', borderRadius:'50%',
                      background:'rgba(28, 39, 76, 0.7)', border:'none',
                      color:'white', cursor:'pointer', display:'flex',
                      alignItems:'center', justifyContent:'center'
                    }}>
                      <X size={16}/>
                    </button>
                  </div>
                  <div style={{ display:'flex', gap:'12px' }}>
                    <button onClick={reset} className="btn-base btn-outline" style={{ flex: 1 }}>Change</button>
                    <button onClick={analyze} className="btn-base btn-primary" style={{ flex: 2 }}>
                      Analyze Prescription →
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <motion.div initial={{ opacity:0, y:10 }} animate={{ opacity:1, y:0 }}
                style={{
                  maxWidth:'640px', margin:'20px auto 0',
                  padding:'14px 18px', borderRadius:'14px',
                  background:'rgba(239,68,68,0.05)', border:'1px solid rgba(239,68,68,0.15)',
                  color:'var(--error)', fontSize:'13px',
                  display:'flex', alignItems:'center', gap:'10px', fontWeight: '600'
                }}>
                <AlertCircle size={15}/> {error}
              </motion.div>
            )}
          </motion.div>
        )}

        {/* ── ANALYZING PHASE ── */}
        {phase === 'analyzing' && (
          <motion.div key="analyzing"
            initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
            style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:'50vh', textAlign:'center' }}>

            <div style={{ position:'relative', marginBottom:'40px' }}>
              {/* Spinner rings */}
              {[80,96,112].map((s,i) => (
                <div key={i} className="spinner-ring" style={{
                  position: i===0 ? 'relative' : 'absolute',
                  top: i===0 ? 0 : `${-(s-80)/2}px`,
                  left: i===0 ? 0 : `${-(s-80)/2}px`,
                  width:`${s}px`, height:`${s}px`,
                  borderTopColor:`rgba(28, 39, 76, ${0.9-i*0.25})`,
                  animationDuration: `${1+i*0.3}s`
                }}/>
              ))}
              <div style={{
                position:'absolute', inset:0,
                display:'flex', alignItems:'center', justifyContent:'center'
              }}>
                <HeartECG size={32} color="var(--dark)"/>
              </div>
            </div>

            <h2 style={{ fontSize:'28px', fontWeight:'900', color: 'var(--dark)', marginBottom:'10px' }}>
              Reading your prescription…
            </h2>
            <p style={{ color:'var(--text-secondary)', fontSize:'16px' }}>
              Gemini Vision is decoding handwriting and extracting structured data
            </p>
          </motion.div>
        )}

        {/* ── RESULTS PHASE ── */}
        {phase === 'results' && result && (
          <motion.div key="results" ref={resultRef}
            initial={{ opacity:0 }} animate={{ opacity:1 }}>

            {/* Top bar */}
            <div className="results-header-bar">
              <div className="status-indicator">
                <div className="status-dot" />
                <span style={{ fontWeight:'700', color:'var(--dark)' }}>Analysis complete</span>
              </div>

              <div style={{ display:'flex', gap:'12px', flexWrap:'wrap', alignItems: 'center' }}>
                <TTSButton result={result} lang={lang}/>
                <LangSelector value={lang} onChange={setLang}/>
                <button onClick={reset} className="btn-base btn-outline">New scan</button>
              </div>
            </div>

            {/* Summary strip */}
            <div className="summary-strip">
              {[
                { label:'Confidence', value:`${Math.round((result.confidence || 0.95)*100)}%` },
                { label:'Medicines found', value:result.medicines?.length || 0 },
                { label:'Diagnoses', value:result.diagnosis?.length || 0 },
                { label:'Language', value:LANG_MAP[langDetected||'en']?.label },
              ].map(({ label, value }) => (
                <div key={label} className="summary-stat-card">
                  <div className="stat-label">{label}</div>
                  <div className="stat-value">{value}</div>
                </div>
              ))}
            </div>

            {/* Main content */}
            <div className="dashboard-grid">

              {/* Left column */}
              <div style={{ display:'flex', flexDirection:'column', gap:'18px' }}>

                {/* Patient */}
                <div className="info-card">
                  <div className="section-title-group">
                    <User size={16} className="section-icon"/>
                    <span className="section-label">Patient</span>
                  </div>
                  <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
                    <Field label="Full Name" value={result.patient?.name} accent/>
                    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'14px' }}>
                      <Field label="Age" value={result.patient?.age}/>
                      <Field label="Gender" value={result.patient?.gender}/>
                    </div>
                  </div>
                </div>

                {/* Doctor */}
                <div className="info-card">
                  <div className="section-title-group">
                    <Stethoscope size={16} className="section-icon"/>
                    <span className="section-label">Doctor</span>
                  </div>
                  <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
                    <Field label="Name" value={result.doctor?.name} accent/>
                    <Field label="Specialization" value={result.doctor?.specialization}/>
                    <Field label="Registration" value={result.doctor?.registration}/>
                  </div>
                </div>

                {/* Diagnosis */}
                {result.diagnosis?.length > 0 && (
                  <div className="info-card">
                    <div className="section-title-group">
                      <Activity size={16} className="section-icon"/>
                      <span className="section-label">Diagnosis</span>
                    </div>
                    <div style={{ display:'flex', flexWrap:'wrap', gap:'8px' }}>
                      {result.diagnosis.map((d,i) => (
                        <span key={i} className="feature-pill" style={{ color: 'var(--dark)' }}>{d}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Confidence ring */}
                <div className="info-card" style={{ display:'flex', alignItems:'center', gap:'20px' }}>
                  <ConfRing value={result.confidence || 0.95}/>
                  <div>
                    <div style={{ fontSize:'14px', fontWeight:'800', color:'var(--dark)', marginBottom:'4px' }}>AI Confidence</div>
                    <div style={{ fontSize:'12px', color:'var(--text-secondary)', lineHeight:1.5 }}>
                      {(result.confidence || 0.95) > 0.9 ? 'High — safe to review' : 'Moderate — verify with pharmacist'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Right column — Medicines */}
              <div className="info-card">
                <div className="section-title-group">
                  <Pill size={16} className="section-icon"/>
                  <span className="section-label">Medications</span>
                  <span style={{
                    marginLeft:'auto', padding:'4px 12px', borderRadius:'20px',
                    background:'var(--bg-secondary)', color:'var(--dark)',
                    fontSize:'11px', fontWeight:'800'
                  }}>{result.medicines?.length || 0} found</span>
                </div>

                {result.medicines?.length ? (
                  <div className="medicine-list">
                    {result.medicines.map((m, i) => <MedCard key={i} med={m} idx={i}/>)}
                  </div>
                ) : (
                  <div style={{ textAlign:'center', padding:'60px 0', color:'var(--text-muted)' }}>
                    <Pill size={40} style={{ marginBottom:'12px', opacity:0.3 }}/>
                    <p style={{ fontSize:'14px' }}>No medicines extracted</p>
                  </div>
                )}
              </div>
            </div>

            {/* Raw text toggle */}
            {result.raw_text && <RawTextPanel text={result.raw_text}/>}
          </motion.div>
        )}
        </AnimatePresence>
      </main>

      <footer className="main-footer">
        {t.footer}
      </footer>

      {/* Global responsive grid override */}
      <style>{`
        @media (max-width: 768px) {
          .results-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}

// ── Raw text collapsible ──────────────────────────────────────────────────────
const RawTextPanel = ({ text }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="raw-text-panel">
      <button onClick={() => setOpen(o => !o)} className="raw-text-toggle">
        <span style={{ display:'flex', alignItems:'center', gap:'8px' }}>
          <FileText size={14}/> Raw OCR Text
        </span>
        <ChevronDown size={18} style={{ transform: open ? 'rotate(180deg)' : 'none', transition:'transform 0.2s' }}/>
      </button>
      {open && (
        <pre className="raw-text-content">{text}</pre>
      )}
    </div>
  );
};