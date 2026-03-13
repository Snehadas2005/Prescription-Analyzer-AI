import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Upload, User, Pill, Activity, Stethoscope, FileText,
  ShieldCheck, Zap, Volume2, VolumeX, Globe, ChevronDown, X, AlertCircle, Languages
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import gsap from 'gsap';
const LANGS = {
  en: { ttsCode: 'en-US', label: 'English', flag: '🇬🇧' },
  hi: { ttsCode: 'hi-IN', label: 'हिंदी',   flag: '🇮🇳' },
};

const UI_STRINGS = {
  en: {
    badge: 'Gemini 2.0 Flash Vision · Hindi + English',
    h1a: 'Read any prescription.', h1b: 'In any language.',
    sub: 'Upload a handwritten or printed prescription — Hindi, English, or mixed — and get structured data with voice readback in seconds.',
    feat1: 'Stateless & Private', feat2: 'Sub-3s Extraction',
    feat3: 'Hindi + Regional Scripts', feat4: 'Voice Readback',
    dropTitle: 'Drop prescription here',
    dropSub: 'or click to browse · JPG, PNG, TIFF supported',
    tag1: 'English prescriptions', tag2: 'Hindi prescriptions', tag3: 'Handwritten OK',
    change: 'Change', analyze: 'Analyze Prescription →',
    analyzing: 'Reading your prescription…',
    analyzingSub: 'Gemini Vision is decoding handwriting and extracting structured data',
    done: 'Analysis complete', newScan: 'New scan',
    confidence: 'Confidence', medicines: 'Medicines',
    diagnoses: 'Diagnoses', scriptLabel: 'Script',
    patient: 'Patient', fullName: 'Full Name', age: 'Age', gender: 'Gender',
    doctor: 'Doctor', name: 'Name', spec: 'Specialization', reg: 'Registration',
    diagnosisSec: 'Diagnosis',
    aiConf: 'AI Confidence', highConf: 'High — safe to review',
    modConf: 'Moderate — verify with pharmacist',
    medications: 'Medications', found: 'found', noMeds: 'No medicines extracted',
    dosage: 'Dosage', frequency: 'Frequency', duration: 'Duration', timing: 'Timing',
    rawText: 'Raw OCR Text', readAloud: 'Read Aloud', stop: 'Stop',
    detected: (l) => `${LANGS[l]?.flag || ''} ${LANGS[l]?.label || l} script detected`,
    footer: '© 2026 VaidyaScan — Developed by Sneha Das · Powered by Gemini Vision',
  },
  hi: {
    badge: 'जेमिनी 2.0 फ्लैश विज़न · हिंदी + अंग्रेज़ी',
    h1a: 'कोई भी पर्चा पढ़ें।', h1b: 'किसी भी भाषा में।',
    sub: 'हस्तलिखित या मुद्रित पर्चा अपलोड करें — हिंदी, अंग्रेज़ी या मिश्रित — और कुछ ही सेकंड में संरचित डेटा प्राप्त करें।',
    feat1: 'गोपनीय डेटा', feat2: 'तेज़ विश्लेषण',
    feat3: 'हिंदी + क्षेत्रीय भाषाएँ', feat4: 'आवाज़ में पढ़ें',
    dropTitle: 'पर्चा यहाँ छोड़ें',
    dropSub: 'या ब्राउज़ करने के लिए क्लिक करें · JPG, PNG, TIFF',
    tag1: 'अंग्रेज़ी पर्चे', tag2: 'हिंदी पर्चे', tag3: 'हस्तलेख ठीक है',
    change: 'बदलें', analyze: 'पर्चा विश्लेषण करें →',
    analyzing: 'आपका पर्चा पढ़ा जा रहा है…',
    analyzingSub: 'जेमिनी विज़न हस्तलेख समझ रहा है और डेटा निकाल रहा है',
    done: 'विश्लेषण पूर्ण', newScan: 'नया स्कैन',
    confidence: 'विश्वसनीयता', medicines: 'दवाइयाँ',
    diagnoses: 'निदान', scriptLabel: 'लिपि',
    patient: 'मरीज़', fullName: 'पूरा नाम', age: 'उम्र', gender: 'लिंग',
    doctor: 'डॉक्टर', name: 'नाम', spec: 'विशेषज्ञता', reg: 'पंजीकरण',
    diagnosisSec: 'निदान',
    aiConf: 'AI विश्वसनीयता', highConf: 'उच्च — समीक्षा के लिए सुरक्षित',
    modConf: 'मध्यम — फार्मासिस्ट से जाँचें',
    medications: 'दवाइयाँ', found: 'मिलीं', noMeds: 'कोई दवाई नहीं मिली',
    dosage: 'मात्रा', frequency: 'आवृत्ति', duration: 'अवधि', timing: 'समय',
    rawText: 'OCR टेक्स्ट', readAloud: 'आवाज़ में सुनें', stop: 'रोकें',
    detected: (l) => `${LANGS[l]?.flag || ''} ${LANGS[l]?.label || l} लिपि पहचानी गई`,
    footer: '© 2026 VaidyaScan — स्नेहा दास द्वारा विकसित · जेमिनी विज़न द्वारा संचालित',
  },
};

const S = (lang, key, ...args) => {
  const dict = UI_STRINGS[lang] || UI_STRINGS.en;
  const val  = dict[key] ?? UI_STRINGS.en[key];
  if (typeof val === 'function') return val(...args);
  return val ?? key;
};

const speak = (text, ttsCode = 'en-US') => {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utt  = new SpeechSynthesisUtterance(text);
  utt.lang   = ttsCode;
  utt.rate   = 0.9;
  utt.pitch  = 1;
  window.speechSynthesis.speak(utt);
};
const stopSpeaking = () => window.speechSynthesis?.cancel();

const buildTTSScript = (result, lang) => {
  if (lang === 'hi') {
    let s = '';
    if (result.patient?.name)          s += `मरीज़ का नाम ${result.patient.name}. `;
    if (result.patient?.age)           s += `उम्र ${result.patient.age} साल. `;
    if (result.doctor?.name)           s += `डॉक्टर ${result.doctor.name}. `;
    if (result.doctor?.specialization) s += `विशेषज्ञता ${result.doctor.specialization}. `;
    if (result.diagnosis?.length)      s += `निदान: ${result.diagnosis.join('। ')}. `;
    if (result.medicines?.length) {
      s += `${result.medicines.length} दवाइयाँ निर्धारित हैं। `;
      result.medicines.forEach((m, i) => {
        s += `${i + 1}. ${m.name}. `;
        if (m.dosage)    s += `मात्रा ${m.dosage}. `;
        if (m.frequency) s += `${m.frequency}. `;
        if (m.duration)  s += `${m.duration} के लिए. `;
      });
    }
    return s;
  }
  
  let s = '';
  if (result.patient?.name)          s += `Patient ${result.patient.name}. `;
  if (result.patient?.age)           s += `Age ${result.patient.age}. `;
  if (result.doctor?.name)           s += `Doctor ${result.doctor.name}. `;
  if (result.doctor?.specialization) s += `Specialization ${result.doctor.specialization}. `;
  if (result.diagnosis?.length)      s += `Diagnosis: ${result.diagnosis.join(', ')}. `;
  if (result.medicines?.length) {
    s += `${result.medicines.length} medications prescribed. `;
    result.medicines.forEach((m, i) => {
      s += `${i + 1}. ${m.name}. `;
      if (m.dosage)    s += `Dosage ${m.dosage}. `;
      if (m.frequency) s += `${m.frequency}. `;
      if (m.duration)  s += `For ${m.duration}. `;
    });
  }
  return s;
};


const Noise = () => (
  <svg className="noise-overlay">
    <filter id="noise">
      <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3"/>
      <feColorMatrix type="saturate" values="0"/>
    </filter>
    <rect width="100%" height="100%" filter="url(#noise)"/>
  </svg>
);

const LangSelector = ({ value, onChange }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef();
  useEffect(() => {
    const close = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, []);
  return (
    <div ref={ref} style={{ position:'relative', userSelect:'none' }}>
      <button onClick={() => setOpen(o => !o)} className="lang-select-btn">
        <Globe size={14}/> {LANGS[value]?.flag} {LANGS[value]?.label} <ChevronDown size={12}/>
      </button>
      {open && (
        <div className="lang-dropdown">
          {Object.entries(LANGS).map(([k, v]) => (
            <button key={k} onClick={() => { onChange(k); setOpen(false); }} 
              className={`lang-option ${value === k ? 'active' : ''}`}>
              {v.flag} {v.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

const TTSButton = ({ result, lang }) => {
  const [speaking, setSpeaking] = useState(false);

  useEffect(() => {
    if (speaking) { stopSpeaking(); setSpeaking(false); }
  }, [lang]); 

  const toggle = () => {
    if (speaking) { stopSpeaking(); setSpeaking(false); return; }
    const script = buildTTSScript(result, lang);
    speak(script, LANGS[lang]?.ttsCode || 'en-US');
    setSpeaking(true);
    const check = setInterval(() => {
      if (!window.speechSynthesis.speaking) { clearInterval(check); setSpeaking(false); }
    }, 300);
  };

  return (
    <button onClick={toggle} className={`tts-btn ${speaking ? 'active' : ''}`}>
      {speaking ? <VolumeX size={15}/> : <Volume2 size={15}/>}
      {speaking ? S(lang, 'stop') : S(lang, 'readAloud')}
    </button>
  );
};

const UploadZone = ({ onFile, lang }) => {
  const [dragging, setDragging] = useState(false);
  const process = (f) => { if (f && f.type.startsWith('image/')) onFile(f); };
  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); process(e.dataTransfer.files[0]); }}
      onClick={() => document.getElementById('file-upload').click()}
      className={`upload-zone-wrapper ${dragging ? 'dragging' : ''}`}
    >
      <div className="upload-icon-box">
        <Upload size={28} />
      </div>
      <p style={{ color:'var(--dark)', fontWeight:'700', fontSize:'17px', marginBottom:'8px' }}>
        {S(lang, 'dropTitle')}
      </p>
      <p style={{ color:'var(--text-secondary)', fontSize:'13px', marginBottom:'20px' }}>
        {S(lang, 'dropSub')}
      </p>
      <div style={{ display:'flex', justifyContent:'center', gap:'8px', flexWrap:'wrap' }}>
        {['tag1','tag2','tag3'].map(k => (
          <span key={k} className="feature-pill" style={{ padding: '5px 12px', fontSize: '12px' }}>
            {S(lang, k)}
          </span>
        ))}
      </div>
      <input type="file" id="file-upload" style={{ display:'none' }}
        accept="image/*" onChange={e => process(e.target.files[0])}/>
    </div>
  );
};

const ConfRing = ({ value }) => {
  const pct = Math.round(value * 100);
  const r = 36, c = 2 * Math.PI * r;
  return (
    <div className="conf-ring">
      <svg width="96" height="96" className="conf-ring-svg">
        <circle cx="48" cy="48" r={r} fill="none" stroke="var(--bg-secondary)" strokeWidth="7"/>
        <circle cx="48" cy="48" r={r} fill="none" stroke="var(--dark)" strokeWidth="7"
          strokeDasharray={c} strokeDashoffset={c - (pct / 100) * c}
          style={{ transition:'stroke-dashoffset 1s ease', strokeLinecap:'round' }}/>
      </svg>
      <div className="conf-ring-text-container">
        <span style={{ fontSize:'20px', fontWeight:'900', color:'var(--dark)', lineHeight:1 }}>{pct}</span>
        <span style={{ fontSize:'10px', color:'var(--text-muted)', fontWeight:'600' }}>%</span>
      </div>
    </div>
  );
};

const Field = ({ label, value, accent }) => (
  <div className="field-group">
    <div className="field-label">{label}</div>
    <div className={`field-value ${value ? (accent ? 'accent' : '') : 'empty'}`}>
      {value || '—'}
    </div>
  </div>
);

const MedCard = ({ med, idx, lang }) => {
  const ref = useRef();
  useEffect(() => {
    gsap.fromTo(ref.current, { opacity:0, y:24 }, { opacity:1, y:0, duration:0.5, delay:idx*0.07, ease:'power2.out' });
  }, []);
  return (
    <div ref={ref} className="med-card">
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:'16px' }}>
        <h4 style={{ fontSize:'16px', fontWeight:'800', color:'var(--dark)', margin:0 }}>{med.name}</h4>
        <span style={{ 
          padding:'4px 12px', borderRadius:'20px', 
          background:'var(--bg-secondary)', color:'var(--dark)', 
          fontSize:'11px', fontWeight:'800' 
        }}>#{idx+1}</span>
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'12px' }}>
        <Field label={S(lang,'dosage')}    value={med.dosage}/>
        <Field label={S(lang,'frequency')} value={med.frequency}/>
        <Field label={S(lang,'duration')}  value={med.duration}/>
        <Field label={S(lang,'timing')}    value={med.timing}/>
      </div>
    </div>
  );
};

const RawTextPanel = ({ text, lang }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="raw-text-panel">
      <button onClick={() => setOpen(o => !o)} className="raw-text-toggle">
        <span style={{ display:'flex', alignItems:'center', gap:'8px' }}>
          <FileText size={14}/> {S(lang,'rawText')}
        </span>
        <ChevronDown size={15} style={{ transform: open ? 'rotate(180deg)' : 'none', transition:'transform 0.2s' }}/>
      </button>
      {open && (
        <pre className="raw-text-content">{text}</pre>
      )}
    </div>
  );
};

export default function App() {
  const [file, setFile]           = useState(null);
  const [preview, setPreview]     = useState(null);
  const [phase, setPhase]         = useState('upload');
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [lang, setLang]           = useState('en');   
  const [prescLang, setPrescLang] = useState(null);   

  const headerRef = useRef();
  const featRef   = useRef();
  const API_URL   = `${import.meta.env.VITE_API_URL}/api/v1`;

  const s = (key, ...args) => S(lang, key, ...args);

  useEffect(() => {
    if (!headerRef.current) return;
    gsap.fromTo([...headerRef.current.children],
      { opacity:0, y:-20 },
      { opacity:1, y:0, duration:0.7, stagger:0.12, ease:'power3.out', delay:0.1 }
    );
  }, []);

  useEffect(() => {
    if (!featRef.current || phase !== 'upload') return;
    gsap.fromTo([...featRef.current.children],
      { opacity:0, scale:0.85 },
      { opacity:1, scale:1, duration:0.4, stagger:0.08, ease:'back.out(1.4)', delay:0.6 }
    );
  }, [phase]);

  const handleFile = useCallback((f) => {
    setFile(f); setError(null); setResult(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(f);
  }, []);

  const reset = () => {
    setFile(null); setPreview(null); setResult(null);
    setError(null); setPrescLang(null); setPhase('upload');
    stopSpeaking();
  };

  const analyze = async () => {
    if (!file) return;
    setPhase('analyzing'); setError(null); stopSpeaking();
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res  = await fetch(`${API_URL}/upload`, { method:'POST', body:fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const data = json.data || json;
      if (json.success) {
        setPrescLang(data.detected_language || 'en');
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

  return (
    <div className="app-container">
      <Noise/>

      {/* ── HEADER ── */}
      <header ref={headerRef} className="main-header">
        <div className="logo-group">
          <div className="logo-box">
            <Activity size={20} color="white"/>
          </div>
          <span className="logo-text">
            VaidyaScan
          </span>
        </div>

        <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
          {prescLang && (
            <span className="detected-badge">
              <Languages size={11}/>
              {s('detected', prescLang)}
            </span>
          )}
          <LangSelector value={lang} onChange={setLang}/>
        </div>
      </header>

      <main className="max-width-container">
        <AnimatePresence mode="wait">

          {/* ── UPLOAD ── */}
          {phase === 'upload' && (
            <motion.div key="upload" initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0, y:-20 }}>

              <div className="hero-section">
                <div className="hero-badge">
                  {s('badge')}
                </div>

                <h1 className="hero-title">
                  {s('h1a')}<br/>
                  <span className="hero-gradient-text">
                    {s('h1b')}
                  </span>
                </h1>

                <p className="hero-subtitle">
                  {s('sub')}
                </p>

                <div ref={featRef} className="feature-pills">
                  {[
                    { icon:<ShieldCheck size={13}/>, key:'feat1' },
                    { icon:<Zap size={13}/>,         key:'feat2' },
                    { icon:<Globe size={13}/>,        key:'feat3' },
                    { icon:<Volume2 size={13}/>,      key:'feat4' },
                  ].map(({ icon, key }) => (
                    <span key={key} className="feature-pill">{icon}{s(key)}</span>
                  ))}
                </div>
              </div>

              <div className="upload-card">
                {!file ? (
                  <UploadZone onFile={handleFile} lang={lang}/>
                ) : (
                  <div>
                    <div style={{ position:'relative', marginBottom:'28px', borderRadius:'16px', overflow:'hidden', border:'1px solid var(--secondary)' }}>
                      <img src={preview} alt="Preview" style={{ width:'100%', maxHeight:'280px', objectFit:'cover', display:'block' }}/>
                      <button onClick={reset} className="btn-base" style={{
                        position:'absolute', top:'10px', right:'10px',
                        width:'32px', height:'32px', borderRadius:'50%',
                        background:'rgba(28, 39, 76, 0.7)', border:'none',
                        color:'white', cursor:'pointer', display:'flex', alignItems:'center', justifyContent:'center'
                      }}><X size={16}/></button>
                    </div>
                    <div style={{ display:'flex', gap:'12px' }}>
                      <button onClick={reset} className="btn-base btn-outline" style={{ flex: 1 }}>{s('change')}</button>
                      <button onClick={analyze} className="btn-base btn-primary" style={{ flex: 2 }}>{s('analyze')}</button>
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <motion.div initial={{ opacity:0, y:10 }} animate={{ opacity:1, y:0 }} style={{
                  maxWidth:'640px', margin:'20px auto 0', padding:'14px 18px', borderRadius:'14px',
                  background:'rgba(239,68,68,0.08)', border:'1px solid rgba(239,68,68,0.25)',
                  color:'#f87171', fontSize:'13px', display:'flex', alignItems:'center', gap:'10px'
                }}>
                  <AlertCircle size={15}/> {error}
                </motion.div>
              )}
            </motion.div>
          )}

          {/* ── ANALYZING ── */}
          {phase === 'analyzing' && (
            <motion.div key="analyzing" initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
              style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:'50vh', textAlign:'center' }}>
              <div className="spinner-container">
                {[80, 96, 112].map((sz, i) => (
                  <div key={i} className="spinner-ring" style={{
                    top:`${(112-sz)/2}px`, left:`${(112-sz)/2}px`,
                    width:`${sz}px`, height:`${sz}px`,
                    borderTopColor:`rgba(28, 39, 76, ${0.9-i*0.25})`,
                    animation:`spin ${1+i*0.3}s linear infinite`
                  }}/>
                ))}
                <div style={{ position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center' }}>
                  <Activity size={32} color="var(--dark)"/>
                </div>
              </div>
              <h2 style={{ fontSize:'28px', fontWeight:'900', color: 'var(--dark)', marginBottom:'10px' }}>{s('analyzing')}</h2>
              <p style={{ color:'var(--text-secondary)', fontSize:'16px' }}>{s('analyzingSub')}</p>
            </motion.div>
          )}

          {/* ── RESULTS ── */}
          {phase === 'results' && result && (
            <motion.div key="results" initial={{ opacity:0, y:30 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.5 }}>

              {/* Top bar */}
              <div className="results-header-bar">
                <div className="status-indicator">
                  <div className="status-dot" />
                  <span style={{ fontWeight:'700', color:'var(--dark)' }}>{s('done')}</span>
                </div>
                <div style={{ display:'flex', gap:'12px', flexWrap:'wrap', alignItems: 'center' }}>
                  <TTSButton result={result} lang={lang}/>
                  <button onClick={reset} className="btn-base btn-outline">{s('newScan')}</button>
                </div>
              </div>

              {/* Summary strip */}
              <div className="summary-strip">
                {[
                  { k:'confidence', v:`${Math.round((result.confidence||0.95)*100)}%`,  c:'var(--dark)' },
                  { k:'medicines',  v: result.medicines?.length || 0,                    c:'var(--dark)' },
                  { k:'diagnoses',  v: result.diagnosis?.length || 0,                    c:'var(--dark)' },
                  { k:'scriptLabel',v: LANGS[prescLang||'en']?.label || '—',            c:'var(--dark)' },
                ].map(({ k, v, c }) => (
                  <div key={k} className="summary-stat-card">
                    <div className="stat-label">{s(k)}</div>
                    <div className="stat-value" style={{ color: c }}>{v}</div>
                  </div>
                ))}
              </div>

              {/* Content grid */}
              <div className="dashboard-grid">

                <div style={{ display:'flex', flexDirection:'column', gap:'18px' }}>

                  {/* Patient card */}
                  <div className="info-card">
                    <div className="section-title-group">
                      <User size={16} className="section-icon"/>
                      <span className="section-label">{s('patient')}</span>
                    </div>
                    <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
                      <Field label={s('fullName')} value={result.patient?.name} accent/>
                      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'14px' }}>
                        <Field label={s('age')}    value={result.patient?.age}/>
                        <Field label={s('gender')} value={result.patient?.gender}/>
                      </div>
                    </div>
                  </div>

                  {/* Doctor card */}
                  <div className="info-card">
                    <div className="section-title-group">
                      <Stethoscope size={16} className="section-icon"/>
                      <span className="section-label">{s('doctor')}</span>
                    </div>
                    <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
                      <Field label={s('name')} value={result.doctor?.name} accent/>
                      <Field label={s('spec')} value={result.doctor?.specialization}/>
                      <Field label={s('reg')}  value={result.doctor?.registration}/>
                    </div>
                  </div>

                  {/* Diagnosis */}
                  {result.diagnosis?.length > 0 && (
                    <div className="info-card">
                      <div className="section-title-group">
                        <Activity size={16} className="section-icon"/>
                        <span className="section-label">{s('diagnosisSec')}</span>
                      </div>
                      <div style={{ display:'flex', flexWrap:'wrap', gap:'8px' }}>
                        {result.diagnosis.map((d, i) => (
                          <span key={i} className="feature-pill" style={{ color: 'var(--dark)' }}>{d}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Confidence ring */}
                  <div className="info-card" style={{ display:'flex', alignItems:'center', gap:'20px' }}>
                    <ConfRing value={result.confidence || 0.95}/>
                    <div>
                      <div style={{ fontSize:'14px', fontWeight:'800', color:'var(--dark)', marginBottom:'4px' }}>{s('aiConf')}</div>
                      <div style={{ fontSize:'12px', color:'var(--text-secondary)', lineHeight:1.5 }}>
                        {(result.confidence || 0.95) > 0.9 ? s('highConf') : s('modConf')}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Medicines */}
                <div className="info-card">
                  <div className="section-title-group">
                    <Pill size={16} className="section-icon"/>
                    <span className="section-label">{s('medications')}</span>
                    <span style={{ 
                      marginLeft:'auto', padding:'4px 12px', borderRadius:'20px', 
                      background:'var(--bg-secondary)', color:'var(--dark)', 
                      fontSize:'11px', fontWeight:'800' 
                    }}>{result.medicines?.length || 0} {s('found')}</span>
                  </div>
                  {result.medicines?.length ? (
                    <div className="medicine-list">
                      {result.medicines.map((m, i) => <MedCard key={i} med={m} idx={i} lang={lang}/>)}
                    </div>
                  ) : (
                    <div style={{ textAlign:'center', padding:'60px 0', color:'var(--text-muted)' }}>
                      <Pill size={40} style={{ marginBottom:'12px', opacity:0.3 }}/>
                      <p style={{ fontSize:'14px' }}>{s('noMeds')}</p>
                    </div>
                  )}
                </div>
              </div>

              {result.raw_text && <RawTextPanel text={result.raw_text} lang={lang}/>}
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      <footer className="main-footer">
        {s('footer')}
      </footer>

      <style>{`
        @media (max-width: 768px) { .results-grid { grid-template-columns: 1fr !important; } }
      `}</style>
    </div>
  );
}