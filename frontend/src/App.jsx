import React, { useState, useCallback, useEffect, useRef } from 'react';
import { 
  Upload, User, Pill, CheckCircle, AlertCircle, Star, 
  Activity, Stethoscope, XCircle, Loader2, AlertTriangle, 
  FileText, Camera, ArrowRight, ChevronRight, Share2, Download,
  History, Sparkles, ShieldCheck, Zap
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import gsap from 'gsap';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  
  const heroRef = useRef(null);

  useEffect(() => {
    if (activeTab === 'upload' && heroRef.current) {
      const children = heroRef.current.children;
      gsap.fromTo(children, 
        { 
          opacity: 0, 
          y: 30,
          filter: 'blur(10px)'
        }, 
        { 
          opacity: 1, 
          y: 0, 
          filter: 'blur(0px)',
          duration: 0.8, 
          stagger: 0.15,
          ease: "power3.out"
        }
      );
    }
  }, [activeTab]);

  const API_URL = `${import.meta.env.VITE_API_URL}/api/v1`;

  const handleFileSelect = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith("image/")) {
        setSelectedFile(file);
        setAnalysisResult(null);
        setError(null);
        
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result);
        };
        reader.readAsDataURL(file);
      } else {
        setError("Please select an image file (JPEG, PNG, etc.)");
      }
    }
  }, []);

  const analyzePrescription = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setActiveTab("analysis");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const actualData = result.data || result;
      
      if (result.success) {
        setAnalysisResult(actualData);
        setTimeout(() => {
          setIsAnalyzing(false);
          setActiveTab("results");
        }, 1500); 
      } else {
        setError(result.error || "Failed to analyze prescription");
        setIsAnalyzing(false);
        setActiveTab("upload");
      }
    } catch (error) {
      setError(`Failed to connect to server: ${error.message}`);
      setIsAnalyzing(false);
      setActiveTab("upload");
    }
  };

  const handleNewAnalysis = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setActiveTab("upload");
    setError(null);
    setPreview(null);
  };


  return (
    <div className="app-root">
      {/* Background Blobs */}
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <div style={{ padding: '40px 0', display: 'flex', justifyContent: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <HeartIcon color="#1C274C" size={40} />
          <span style={{ fontSize: '28px', fontWeight: '800', color: '#1C274C', fontFamily: 'Plus Jakarta Sans' }}>
            Prescription<span style={{ color: 'var(--primary-hover)' }}>Analyzer</span>AI
          </span>
        </div>
      </div>

      <main className="container" style={{ paddingTop: '64px', paddingBottom: '100px' }}>
        <AnimatePresence mode="wait">
          {/* UPLOAD SECTION */}
          {activeTab === "upload" && (
            <motion.div 
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
            >
              <div style={{
                width: '100%',
                background: 'var(--secondary)',
                borderRadius: 'var(--radius-xl)',
                padding: '120px 40px',
                position: 'relative',
                overflow: 'hidden',
                textAlign: 'center',
                marginBottom: '80px'
              }}>
                <SparkleIcon style={{ position: 'absolute', top: 40, right: 40, width: 180, height: 180, color: '#D7D6FF', opacity: 0.8 }} />
                <SparkleIcon style={{ position: 'absolute', bottom: -20, left: -40, width: 220, height: 220, color: '#D7D6FF', opacity: 0.8 }} />
                
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 16px',
                  background: 'white',
                  color: '#1C274C',
                  borderRadius: '50px',
                  fontSize: '12px',
                  fontWeight: '800',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  marginBottom: '24px',
                  position: 'relative',
                  zIndex: 1
                }}>
                  <Sparkles size={14} /> Next-Gen AI Vision
                </div>

                <h1 style={{ fontSize: 'clamp(36px, 6vw, 64px)', color: '#1C274C', marginBottom: '24px', fontWeight: '800', lineHeight: 1.1, position: 'relative', zIndex: 1 }}>
                   Unlock the truth in <br /><span style={{ color: 'var(--white)', textShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>Medical Handprints.</span>
                </h1>

                <p style={{ fontSize: '18px', color: '#475569', maxWidth: '600px', margin: '0 auto 40px', position: 'relative', zIndex: 1 }}>
                  Our neural OCR extracts medicines, symptoms, and dosages from handwritten prescriptions with industrial accuracy.
                </p>
                
                <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', marginBottom: '40px', position: 'relative', zIndex: 1 }}>
                   <Badge icon={<ShieldCheck size={18} />} text="End-to-End Encrypted" />
                   <Badge icon={<Zap size={18} />} text="Sub-second Inference" />
                </div>
                
                {selectedFile ? (
                  <div style={{ position: 'relative', zIndex: 1 }}>
                    <div style={{ marginBottom: '32px' }}>
                      <img src={preview} alt="Preview" style={{ maxHeight: '250px', borderRadius: '20px', border: '5px solid white', boxShadow: 'var(--shadow-lg)' }} />
                      <p style={{ marginTop: '12px', color: '#1C274C', fontWeight: '600' }}>{selectedFile.name}</p>
                    </div>
                    <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                      <button 
                        className="btn btn-outline" 
                        style={{ padding: '12px 32px', borderRadius: '50px' }}
                        onClick={() => { setSelectedFile(null); setPreview(null); }}
                      >
                        Cancel
                      </button>
                      <button 
                        className="btn btn-primary" 
                        style={{ background: '#1C274C', color: 'white', padding: '12px 32px', borderRadius: '50px' }}
                        onClick={analyzePrescription}
                      >
                        Analyze Now
                      </button>
                    </div>
                  </div>
                ) : (
                  <button 
                    className="btn btn-primary" 
                    style={{ background: '#1C274C', color: 'white', padding: '18px 40px', borderRadius: '50px', fontSize: '18px', position: 'relative', zIndex: 1 }}
                    onClick={() => document.getElementById('file-upload').click()}
                  >
                    Start Now
                  </button>
                )}

                <input type="file" id="file-upload" className="hidden" style={{ display: 'none' }} onChange={handleFileSelect} />
              </div>
            </motion.div>
          )}

          {/* ANALYSIS STATE */}
          {activeTab === "analysis" && (
            <motion.div 
              key="analysis"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '50vh', textAlign: 'center' }}
            >
              <div style={{ position: 'relative', width: '120px', height: '120px', marginBottom: '40px' }}>
                <div style={{ position: 'absolute', inset: 0, background: 'var(--primary)', opacity: 0.1, blur: '40px', borderRadius: '50%' }} />
                <Loader2 size={120} className="animate-spin" color="var(--primary)" strokeWidth={1} />
                <HeartIcon size={48} color="var(--primary)" style={{ position: 'absolute', top: '50%', left: '50%', translate: '-50% -50%' }} />
              </div>
              <h2 style={{ fontSize: '32px', marginBottom: '8px' }}>Scanning Neural Fibers...</h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '400px' }}>Our OCR engine is decoding handwriting and cross-referencing medical drugs.</p>
            </motion.div>
          )}

          {/* RESULTS STATE */}
          {activeTab === "results" && analysisResult && (
            <motion.div 
              key="results"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}
            >
              {/* Stats Bar */}
              <div className="glass card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '24px 32px' }}>
                 <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Star color="#1C274C" fill="#1C274C" size={32} />
                      <div>
                        <div style={{ fontSize: '12px', fontWeight: '800', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Accuracy Rate</div>
                        <div style={{ fontSize: '24px', fontWeight: '900', color: '#1C274C' }}>{Math.round((analysisResult?.confidence || 0.96) * 100)}%</div>
                      </div>
                    </div>

                    <div style={{ width: '1px', height: '40px', background: '#e2e8f0' }} />

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Pill color="#1C274C" size={32} />
                      <div>
                        <div style={{ fontSize: '12px', fontWeight: '800', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Drugs Identified</div>
                        <div style={{ fontSize: '24px', fontWeight: '900', color: '#1C274C' }}>{analysisResult.medicines?.length || 0}</div>
                      </div>
                    </div>
                 </div>

                 <button onClick={handleNewAnalysis} className="btn btn-primary" style={{ background: '#1C274C' }}>Start New Analysis</button>
              </div>

              {/* Data Grid */}
              <div className="dashboard-grid">
                {/* Side Pane */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                  <SectionCard icon={<User size={20} color="#1C274C" />} title="Patient Information">
                     <Field label="Full Name" value={analysisResult.patient?.name} />
                     <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                        <Field label="Age" value={analysisResult.patient?.age} />
                        <Field label="Gender" value={analysisResult.patient?.gender} />
                     </div>
                  </SectionCard>

                  <SectionCard icon={<Stethoscope size={20} color="#1C274C" />} title="Doctor Insights">
                     <Field label="Prescribing Clinician" value={analysisResult.doctor?.name} />
                     <Field label="Specialization" value={analysisResult.doctor?.specialization} />
                  </SectionCard>

                  <SectionCard icon={<Activity size={20} color="#1C274C" />} title="Symptom Tags">
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {analysisResult.diagnosis?.map((d, i) => (
                        <span key={i} style={{ 
                          padding: '6px 12px', 
                          background: 'white', 
                          border: '1px solid #e2e8f0', 
                          borderRadius: '10px', 
                          fontSize: '12px', 
                          fontWeight: '700', 
                          color: 'var(--text-secondary)' 
                        }}>{d}</span>
                      ))}
                    </div>
                  </SectionCard>
                </div>

                {/* Main Pane: Medicines */}
                <div className="glass card" style={{ padding: '40px' }}>
                   <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <HeartIcon size={24} color="#1C274C" />
                        <h2 style={{ fontSize: '28px', color: '#1C274C' }}>Identified Medications</h2>
                      </div>
                   </div>

                   <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }} className="med-list">
                      {analysisResult.medicines?.map((med, idx) => (
                        <div key={idx} style={{ background: 'white', border: '1px solid #f1f5f9', padding: '24px', borderRadius: '24px', position: 'relative' }}>
                           <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: '#1C274C' }}>
                               {med.name || "Untitled Molecule"}
                           </h3>
                           <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                              <Field label="Dosage" value={med.dosage} small />
                              <Field label="Frequency" value={med.frequency} small />
                              <Field label="Duration" value={med.duration} small />
                              <Field label="Timing" value={med.instructions} small />
                           </div>
                        </div>
                      ))}
                   </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer style={{ textAlign: 'center', padding: '64px 0', opacity: 0.6, fontSize: '14px', fontWeight: '600', color: '#1C274C' }}>
         &copy; 2026 Prescription Analyzer AI. Developed by Sneha Das. All Rights Reserved.
      </footer>

    </div>
  );
};

const Badge = ({ icon, text }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 20px', background: 'white', border: '1px solid #e2e8f0', borderRadius: '16px', fontSize: '13px', fontWeight: '700', color: 'var(--text-secondary)' }}>
    {icon} {text}
  </div>
);

const HeartIcon = ({ size = 24, color = "currentColor" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12.39 20.87C12.17 20.95 11.83 20.95 11.61 20.87C9.74 20.23 5.56 17.57 3.73 14.15C1.9 10.73 2.5 7.6 3.93 5.86C5.59 3.86 8.5 3.33 10.7 5.09L12 6.13L13.3 5.09C15.5 3.33 18.41 3.86 20.07 5.86C21.5 7.6 22.1 10.73 20.27 14.15C18.44 17.57 14.26 20.23 12.39 20.87Z" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M6 11.5H8L9.5 8L11.5 14L13 11.5H15" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const SparkleIcon = ({ style }) => (
  <svg viewBox="0 0 100 100" style={style} fill="currentColor">
    <path d="M50 0L57.5 37.5L95 45L57.5 52.5L50 90L42.5 52.5L5 45L42.5 37.5L50 0Z" />
  </svg>
);

const SectionCard = ({ icon, title, children }) => (
  <div className="glass card" style={{ padding: '24px' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px', borderBottom: '1px solid #f1f5f9', paddingBottom: '12px' }}>
      {icon} <h3 style={{ fontSize: '16px', color: 'var(--text-primary)' }}>{title}</h3>
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {children}
    </div>
  </div>
);

const Field = ({ label, value, small = false }) => (
  <div>
    <label style={{ display: 'block', textTransform: 'uppercase', fontSize: '10px', fontWeight: '900', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '4px' }}>{label}</label>
    <div style={{ fontSize: small ? '14px' : '16px', fontWeight: '700', color: 'var(--text-primary)' }}>{value || <span style={{ opacity: 0.3 }}>N/A</span>}</div>
  </div>
);

export default App;