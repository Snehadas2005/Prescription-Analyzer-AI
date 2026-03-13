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
    footer: '© 2026 RxScan AI — Developed by Sneha Das · Powered by Gemini Vision',
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
    footer: '© 2026 वैद्यस्कैन — स्नेहा दास द्वारा विकसित · जेमिनी विज़न द्वारा संचालित',
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

const HeartECG = ({ size = 28, color = 'currentColor' }) => (
  <svg width={size} height={size} viewBox="0 0 28 28" fill="none">
    <path d=  "M14 24S4 18 4 10.5A5.5 5.5 0 0 1 14 7a5.5 5.5 0 0 1 10 3.5C24 18 14 24 14 24z"
      stroke={color} strokeWidth="1.8" fill="none" strokeLinejoin="round"/>
    <polyline points="6,13 9,13 11,9 13,17 15,13 17,13 19,13 21,13"
      stroke={color} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const Noise = () => (
  <svg style={{ position:'fixed', inset:0, width:'100%', height:'100%', pointerEvents:'none', zIndex:0, opacity:0.03 }}>
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
      <button onClick={() => setOpen(o => !o)} style={{
        display:'flex', alignItems:'center', gap:'6px',
        padding:'8px 14px', borderRadius:'12px',
        border:'1.5px solid #2a2a3e', background:'#16162a',
        color:'#c8c8e8', fontSize:'13px', fontWeight:'600',
        cursor:'pointer', fontFamily:'inherit'
      }}>
        <Globe size={14}/> {LANGS[value]?.flag} {LANGS[value]?.label} <ChevronDown size={12}/>
      </button>
      {open && (
        <div style={{
          position:'absolute', top:'110%', right:0,
          background:'#1a1a2e', border:'1.5px solid #2a2a3e',
          borderRadius:'14px', padding:'6px', zIndex:100,
          minWidth:'170px', boxShadow:'0 20px 60px rgba(0,0,0,0.5)'
        }}>
          {Object.entries(LANGS).map(([k, v]) => (
            <button key={k} onClick={() => { onChange(k); setOpen(false); }} style={{
              display:'flex', alignItems:'center', gap:'10px',
              width:'100%', padding:'9px 12px', borderRadius:'9px',
              border:'none', background: value === k ? '#2a2a4a' : 'transparent',
              color:'#c8c8e8', fontSize:'13px', cursor:'pointer',
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
    <button onClick={toggle} style={{
      display:'flex', alignItems:'center', gap:'8px',
      padding:'10px 18px', borderRadius:'12px',
      border:'1.5px solid', borderColor: speaking ? '#7c6bdb' : '#2a2a3e',
      background: speaking ? 'rgba(124,107,219,0.15)' : '#16162a',
      color: speaking ? '#a89af0' : '#8888aa',
      fontSize:'13px', fontWeight:'600', cursor:'pointer',
      fontFamily:'inherit', transition:'all 0.2s'
    }}>
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
      style={{
        border:`2px dashed ${dragging ? '#7c6bdb' : '#2a2a3e'}`,
        borderRadius:'24px', padding:'64px 32px', textAlign:'center', cursor:'pointer',
        background: dragging ? 'rgba(124,107,219,0.06)' : 'transparent', transition:'all 0.25s'
      }}
    >
      <div style={{
        width:'72px', height:'72px', borderRadius:'20px',
        background:'linear-gradient(135deg,#7c6bdb22,#a89af022)',
        border:'1.5px solid #2a2a3e', display:'flex', alignItems:'center',
        justifyContent:'center', margin:'0 auto 20px'
      }}>
        <Upload size={28} color="#7c6bdb"/>
      </div>
      <p style={{ color:'#c8c8e8', fontWeight:'700', fontSize:'17px', marginBottom:'8px' }}>
        {S(lang, 'dropTitle')}
      </p>
      <p style={{ color:'#555570', fontSize:'13px', marginBottom:'20px' }}>
        {S(lang, 'dropSub')}
      </p>
      <div style={{ display:'flex', justifyContent:'center', gap:'8px', flexWrap:'wrap' }}>
        {['tag1','tag2','tag3'].map(k => (
          <span key={k} style={{
            padding:'5px 12px', borderRadius:'20px',
            border:'1px solid #2a2a3e', color:'#6666aa', fontSize:'12px'
          }}>{S(lang, k)}</span>
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
    <div style={{ position:'relative', width:'96px', height:'96px', flexShrink:0 }}>
      <svg width="96" height="96" style={{ transform:'rotate(-90deg)' }}>
        <circle cx="48" cy="48" r={r} fill="none" stroke="#1e1e32" strokeWidth="7"/>
        <circle cx="48" cy="48" r={r} fill="none" stroke="#7c6bdb" strokeWidth="7"
          strokeDasharray={c} strokeDashoffset={c - (pct / 100) * c}
          style={{ transition:'stroke-dashoffset 1s ease', strokeLinecap:'round' }}/>
      </svg>
      <div style={{ position:'absolute', inset:0, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center' }}>
        <span style={{ fontSize:'20px', fontWeight:'900', color:'#c8c8e8', lineHeight:1 }}>{pct}</span>
        <span style={{ fontSize:'10px', color:'#555570', fontWeight:'600' }}>%</span>
      </div>
    </div>
  );
};

const Field = ({ label, value, accent }) => (
  <div>
    <div style={{ fontSize:'10px', fontWeight:'800', color:'#44445a', letterSpacing:'0.08em', textTransform:'uppercase', marginBottom:'5px' }}>{label}</div>
    <div style={{ fontSize:'14px', fontWeight:'600', color: value ? (accent || '#c8c8e8') : '#333350' }}>{value || '—'}</div>
  </div>
);

const MedCard = ({ med, idx, lang }) => {
  const ref = useRef();
  useEffect(() => {
    gsap.fromTo(ref.current, { opacity:0, y:24 }, { opacity:1, y:0, duration:0.5, delay:idx*0.07, ease:'power2.out' });
  }, []);
  return (
    <div ref={ref} style={{
      background:'#111125', border:'1px solid #1e1e38',
      borderRadius:'20px', padding:'22px', borderLeft:'3px solid #7c6bdb'
    }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:'16px' }}>
        <h4 style={{ fontSize:'15px', fontWeight:'800', color:'#a89af0', margin:0 }}>{med.name}</h4>
        <span style={{ padding:'3px 10px', borderRadius:'20px', background:'rgba(124,107,219,0.12)', color:'#7c6bdb', fontSize:'11px', fontWeight:'700' }}>#{idx+1}</span>
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
    <div style={{ marginTop:'24px', background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'20px', overflow:'hidden' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width:'100%', padding:'16px 24px', display:'flex', alignItems:'center',
        justifyContent:'space-between', background:'transparent', border:'none',
        cursor:'pointer', color:'#555570', fontFamily:'inherit',
        fontWeight:'700', fontSize:'12px', textTransform:'uppercase', letterSpacing:'0.07em'
      }}>
        <span style={{ display:'flex', alignItems:'center', gap:'8px' }}>
          <FileText size={13}/> {S(lang,'rawText')}
        </span>
        <ChevronDown size={15} style={{ transform: open ? 'rotate(180deg)' : 'none', transition:'transform 0.2s' }}/>
      </button>
      {open && (
        <pre style={{
          padding:'0 24px 24px', margin:0, fontFamily:"'JetBrains Mono','Fira Mono',monospace",
          fontSize:'12px', color:'#44445a', lineHeight:1.7,
          whiteSpace:'pre-wrap', wordBreak:'break-word', maxHeight:'260px', overflowY:'auto'
        }}>{text}</pre>
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
    <div style={{
      minHeight:'100vh', background:'#0a0a1a', color:'#c8c8e8',
      fontFamily:'"Inter","Noto Sans Devanagari","Noto Sans Bengali","Noto Sans Telugu",sans-serif',
      position:'relative'
    }}>
      <Noise/>

      {/* ── HEADER ── */}
      <header ref={headerRef} style={{
        display:'flex', alignItems:'center', justifyContent:'space-between',
        padding:'20px 32px', borderBottom:'1px solid #0f0f20',
        position:'sticky', top:0, zIndex:50,
        background:'rgba(10,10,26,0.92)', backdropFilter:'blur(16px)'
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
          <div style={{
            width:'36px', height:'36px', borderRadius:'10px',
            background:'linear-gradient(135deg,#7c6bdb,#c084fc)',
            display:'flex', alignItems:'center', justifyContent:'center'
          }}>
            <HeartECG size={20} color="white"/>
          </div>
          <span style={{ fontWeight:'800', fontSize:'16px', letterSpacing:'-0.02em' }}>
            {lang === 'hi' ? 'वैद्यस्कैन' : 'RxScan AI'}
          </span>
        </div>

        <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
          {/* Prescription script badge — info only, doesn't drive anything */}
          {prescLang && (
            <span style={{
              padding:'5px 12px', borderRadius:'20px',
              background:'rgba(124,107,219,0.1)', border:'1px solid #2a2a4a',
              color:'#7c6bdb', fontSize:'11px', fontWeight:'700',
              display:'flex', alignItems:'center', gap:'5px'
            }}>
              <Languages size={11}/>
              {s('detected', prescLang)}
            </span>
          )}
          <LangSelector value={lang} onChange={setLang}/>
        </div>
      </header>

      <main style={{ maxWidth:'1100px', margin:'0 auto', padding:'60px 24px 120px' }}>
        <AnimatePresence mode="wait">

          {/* ── UPLOAD ── */}
          {phase === 'upload' && (
            <motion.div key="upload" initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0, y:-20 }}>

              <div style={{ textAlign:'center', marginBottom:'64px' }}>
                <div style={{
                  display:'inline-block', padding:'6px 16px', borderRadius:'20px',
                  background:'rgba(124,107,219,0.1)', border:'1px solid rgba(124,107,219,0.3)',
                  color:'#a89af0', fontSize:'12px', fontWeight:'700',
                  letterSpacing:'0.06em', marginBottom:'28px'
                }}>
                  {s('badge')}
                </div>

                <h1 style={{
                  fontSize:'clamp(36px,6vw,70px)', fontWeight:'900',
                  lineHeight:1.05, letterSpacing:'-0.03em', color:'#e8e8f0', marginBottom:'22px'
                }}>
                  {s('h1a')}<br/>
                  <span style={{ background:'linear-gradient(90deg,#7c6bdb,#c084fc)', WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent' }}>
                    {s('h1b')}
                  </span>
                </h1>

                <p style={{ fontSize:'17px', color:'#555570', maxWidth:'520px', margin:'0 auto 48px', lineHeight:1.6 }}>
                  {s('sub')}
                </p>

                <div ref={featRef} style={{ display:'flex', justifyContent:'center', gap:'10px', flexWrap:'wrap', marginBottom:'56px' }}>
                  {[
                    { icon:<ShieldCheck size={13}/>, key:'feat1' },
                    { icon:<Zap size={13}/>,         key:'feat2' },
                    { icon:<Globe size={13}/>,        key:'feat3' },
                    { icon:<Volume2 size={13}/>,      key:'feat4' },
                  ].map(({ icon, key }) => (
                    <span key={key} style={{
                      display:'inline-flex', alignItems:'center', gap:'6px',
                      padding:'7px 14px', borderRadius:'20px',
                      border:'1px solid #1e1e38', background:'#111125',
                      color:'#7788aa', fontSize:'12px', fontWeight:'600'
                    }}>{icon}{s(key)}</span>
                  ))}
                </div>
              </div>

              <div style={{ background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'28px', padding:'40px', maxWidth:'640px', margin:'0 auto' }}>
                {!file ? (
                  <UploadZone onFile={handleFile} lang={lang}/>
                ) : (
                  <div>
                    <div style={{ position:'relative', marginBottom:'28px', borderRadius:'16px', overflow:'hidden', border:'1px solid #1e1e38' }}>
                      <img src={preview} alt="Preview" style={{ width:'100%', maxHeight:'280px', objectFit:'cover', display:'block' }}/>
                      <button onClick={reset} style={{
                        position:'absolute', top:'10px', right:'10px',
                        width:'32px', height:'32px', borderRadius:'50%',
                        background:'rgba(0,0,0,0.7)', border:'none',
                        color:'white', cursor:'pointer', display:'flex', alignItems:'center', justifyContent:'center'
                      }}><X size={16}/></button>
                    </div>
                    <div style={{ display:'flex', gap:'12px' }}>
                      <button onClick={reset} style={{
                        flex:1, padding:'13px', borderRadius:'14px',
                        border:'1.5px solid #2a2a3e', background:'transparent',
                        color:'#6666aa', fontWeight:'700', cursor:'pointer', fontFamily:'inherit'
                      }}>{s('change')}</button>
                      <button onClick={analyze} style={{
                        flex:2, padding:'13px', borderRadius:'14px', border:'none',
                        background:'linear-gradient(135deg,#7c6bdb,#9b87f0)',
                        color:'white', fontWeight:'800', cursor:'pointer',
                        fontSize:'15px', fontFamily:'inherit',
                        boxShadow:'0 8px 30px rgba(124,107,219,0.4)'
                      }}>{s('analyze')}</button>
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
              <div style={{ position:'relative', width:'112px', height:'112px', marginBottom:'40px' }}>
                {[80, 96, 112].map((sz, i) => (
                  <div key={i} style={{
                    position:'absolute', top:`${(112-sz)/2}px`, left:`${(112-sz)/2}px`,
                    width:`${sz}px`, height:`${sz}px`, borderRadius:'50%',
                    border:'2px solid transparent',
                    borderTopColor:`rgba(124,107,219,${0.9-i*0.25})`,
                    animation:`spin ${1+i*0.3}s linear infinite`
                  }}/>
                ))}
                <div style={{ position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center' }}>
                  <HeartECG size={28} color="#7c6bdb"/>
                </div>
              </div>
              <h2 style={{ fontSize:'24px', fontWeight:'800', marginBottom:'10px' }}>{s('analyzing')}</h2>
              <p style={{ color:'#555570', fontSize:'15px' }}>{s('analyzingSub')}</p>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </motion.div>
          )}

          {/* ── RESULTS ── */}
          {phase === 'results' && result && (
            <motion.div key="results" initial={{ opacity:0, y:30 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.5 }}>

              {/* Top bar */}
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'36px', flexWrap:'wrap', gap:'12px' }}>
                <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
                  <div style={{ width:'10px', height:'10px', borderRadius:'50%', background:'#4ade80', boxShadow:'0 0 8px #4ade80' }}/>
                  <span style={{ fontWeight:'700', color:'#c8c8e8' }}>{s('done')}</span>
                </div>
                <div style={{ display:'flex', gap:'10px', flexWrap:'wrap' }}>
                  <TTSButton result={result} lang={lang}/>
                  <button onClick={reset} style={{
                    padding:'10px 18px', borderRadius:'12px', border:'1.5px solid #2a2a3e',
                    background:'transparent', color:'#6666aa', fontSize:'13px',
                    fontWeight:'700', cursor:'pointer', fontFamily:'inherit'
                  }}>{s('newScan')}</button>
                </div>
              </div>

              {/* Summary strip */}
              <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(160px,1fr))', gap:'16px', marginBottom:'32px' }}>
                {[
                  { k:'confidence', v:`${Math.round((result.confidence||0.95)*100)}%`,  c:'#7c6bdb' },
                  { k:'medicines',  v: result.medicines?.length || 0,                    c:'#a89af0' },
                  { k:'diagnoses',  v: result.diagnosis?.length || 0,                    c:'#c084fc' },
                  { k:'scriptLabel',v: LANGS[prescLang||'en']?.label || '—',            c:'#818cf8' },
                ].map(({ k, v, c }) => (
                  <div key={k} style={{ background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'18px', padding:'20px 22px' }}>
                    <div style={{ fontSize:'11px', color:'#44445a', fontWeight:'800', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:'6px' }}>{s(k)}</div>
                    <div style={{ fontSize:'24px', fontWeight:'900', color:c }}>{v}</div>
                  </div>
                ))}
              </div>

              {/* Content grid */}
              <div style={{ display:'grid', gridTemplateColumns:'340px 1fr', gap:'24px' }} className="results-grid">

                <div style={{ display:'flex', flexDirection:'column', gap:'18px' }}>

                  {/* Patient card */}
                  <div style={{ background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'20px', padding:'24px' }}>
                    <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'18px', paddingBottom:'14px', borderBottom:'1px solid #14142a' }}>
                      <User size={15} color="#7c6bdb"/>
                      <span style={{ fontSize:'12px', fontWeight:'800', color:'#555570', textTransform:'uppercase', letterSpacing:'0.07em' }}>{s('patient')}</span>
                    </div>
                    <div style={{ display:'flex', flexDirection:'column', gap:'14px' }}>
                      <Field label={s('fullName')} value={result.patient?.name} accent="#c8c8e8"/>
                      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'14px' }}>
                        <Field label={s('age')}    value={result.patient?.age}/>
                        <Field label={s('gender')} value={result.patient?.gender}/>
                      </div>
                    </div>
                  </div>

                  {/* Doctor card */}
                  <div style={{ background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'20px', padding:'24px' }}>
                    <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'18px', paddingBottom:'14px', borderBottom:'1px solid #14142a' }}>
                      <Stethoscope size={15} color="#7c6bdb"/>
                      <span style={{ fontSize:'12px', fontWeight:'800', color:'#555570', textTransform:'uppercase', letterSpacing:'0.07em' }}>{s('doctor')}</span>
                    </div>
                    <div style={{ display:'flex', flexDirection:'column', gap:'14px' }}>
                      <Field label={s('name')} value={result.doctor?.name} accent="#c8c8e8"/>
                      <Field label={s('spec')} value={result.doctor?.specialization}/>
                      <Field label={s('reg')}  value={result.doctor?.registration}/>
                    </div>
                  </div>

                  {/* Diagnosis */}
                  {result.diagnosis?.length > 0 && (
                    <div style={{ background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'20px', padding:'24px' }}>
                      <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'18px', paddingBottom:'14px', borderBottom:'1px solid #14142a' }}>
                        <Activity size={15} color="#7c6bdb"/>
                        <span style={{ fontSize:'12px', fontWeight:'800', color:'#555570', textTransform:'uppercase', letterSpacing:'0.07em' }}>{s('diagnosisSec')}</span>
                      </div>
                      <div style={{ display:'flex', flexWrap:'wrap', gap:'8px' }}>
                        {result.diagnosis.map((d, i) => (
                          <span key={i} style={{
                            padding:'5px 12px', borderRadius:'20px',
                            background:'rgba(124,107,219,0.08)', border:'1px solid rgba(124,107,219,0.2)',
                            color:'#a89af0', fontSize:'13px', fontWeight:'600'
                          }}>{d}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Confidence ring */}
                  <div style={{ background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'20px', padding:'24px', display:'flex', alignItems:'center', gap:'20px' }}>
                    <ConfRing value={result.confidence || 0.95}/>
                    <div>
                      <div style={{ fontSize:'13px', fontWeight:'800', color:'#c8c8e8', marginBottom:'4px' }}>{s('aiConf')}</div>
                      <div style={{ fontSize:'12px', color:'#555570', lineHeight:1.5 }}>
                        {(result.confidence || 0.95) > 0.9 ? s('highConf') : s('modConf')}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Medicines */}
                <div style={{ background:'#0e0e20', border:'1px solid #1a1a30', borderRadius:'20px', padding:'28px' }}>
                  <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'24px', paddingBottom:'16px', borderBottom:'1px solid #14142a' }}>
                    <Pill size={15} color="#7c6bdb"/>
                    <span style={{ fontSize:'12px', fontWeight:'800', color:'#555570', textTransform:'uppercase', letterSpacing:'0.07em' }}>{s('medications')}</span>
                    <span style={{ marginLeft:'auto', padding:'3px 10px', borderRadius:'20px', background:'rgba(124,107,219,0.1)', color:'#7c6bdb', fontSize:'11px', fontWeight:'700' }}>
                      {result.medicines?.length || 0} {s('found')}
                    </span>
                  </div>
                  {result.medicines?.length ? (
                    <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(260px,1fr))', gap:'16px' }}>
                      {result.medicines.map((m, i) => <MedCard key={i} med={m} idx={i} lang={lang}/>)}
                    </div>
                  ) : (
                    <div style={{ textAlign:'center', padding:'60px 0', color:'#333350' }}>
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

      <footer style={{ textAlign:'center', padding:'40px 24px', borderTop:'1px solid #0f0f20', color:'#333350', fontSize:'13px', fontWeight:'600' }}>
        {s('footer')}
      </footer>

      <style>{`
        @media (max-width: 768px) { .results-grid { grid-template-columns: 1fr !important; } }
      `}</style>
    </div>
  );
}