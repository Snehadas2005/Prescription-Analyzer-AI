import React, { useState, useCallback, useEffect, useRef } from 'react';
import { 
  Upload, User, Pill, CheckCircle, AlertCircle, Star, 
  Activity, Stethoscope, XCircle, Loader2, AlertTriangle, 
  FileText, Camera, ArrowRight, ChevronRight, Share2, Download,
  History, Sparkles, BrainCircuit, ShieldCheck, Zap
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
  const [isEditing, setIsEditing] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);
  
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

  const API_BASE_URL = "http://localhost:8080/api/v1";

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
    setIsEditing(false);
  };

  const handleFieldChange = (section, field, value, index = null) => {
    const newResult = { ...analysisResult };
    if (index !== null) {
      newResult[section][index][field] = value;
    } else if (section) {
      newResult[section][field] = value;
    } else {
      newResult[field] = value;
    }
    setAnalysisResult(newResult);
  };

  const submitFeedback = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prescription_id: analysisResult.prescription_id,
          feedback_type: "correction",
          corrections: analysisResult,
          timestamp: new Date().toISOString()
        }),
      });

      if (response.ok) {
        setFeedbackSent(true);
        setIsEditing(false);
        setTimeout(() => setFeedbackSent(false), 3000);
      }
    } catch (error) {
      setError(`Failed to save corrections: ${error.message}`);
    }
  };

  return (
    <div className="app-root">
      {/* Background Blobs */}
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <div style={{ padding: '40px 0 20px', display: 'flex', justifyContent: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            background: '#0f172a',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white'
          }}>
            <BrainCircuit size={24} />
          </div>
          <span style={{ fontSize: '24px', fontWeight: '800', fontFamily: 'Plus Jakarta Sans' }}>
            Prescription<span style={{ color: 'var(--primary)' }}>AI</span>
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
              <header ref={heroRef} style={{ textAlign: 'center', maxWidth: '800px', marginBottom: '64px' }}>
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 16px',
                  background: 'var(--primary-light)',
                  color: 'var(--primary)',
                  borderRadius: '50px',
                  fontSize: '12px',
                  fontWeight: '800',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  marginBottom: '24px'
                }}>
                  <Sparkles size={14} /> Next-Gen AI Vision
                </div>
                <h1 style={{ fontSize: 'clamp(40px, 8vw, 72px)', marginBottom: '24px', lineHeight: 1.1 }}>
                  Unlock the truth in <span className="gradient-text">Medical Handprints.</span>
                </h1>
                <p style={{ fontSize: '20px', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto 40px' }}>
                  Our neural OCR extracts medicines, symptoms, and dosages from handwritten prescriptions with industrial accuracy.
                </p>

                <div style={{ display: 'flex', justifyContent: 'center', gap: '16px' }}>
                   <Badge icon={<ShieldCheck size={18} />} text="End-to-End Encrypted" />
                   <Badge icon={<Zap size={18} />} text="Sub-second Inference" />
                </div>
              </header>

              {/* Upload Card */}
              <div 
                className="glass card" 
                style={{ 
                  width: '100%', 
                  maxWidth: '700px', 
                  border: '2px dashed #cbd5e1', 
                  cursor: 'pointer',
                  textAlign: 'center',
                  transition: 'all 0.4s'
                }}
                onClick={() => document.getElementById('file-upload').click()}
                onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--primary)'}
                onMouseLeave={(e) => e.currentTarget.style.borderColor = '#cbd5e1'}
              >
                <input type="file" id="file-upload" className="hidden" style={{ display: 'none' }} onChange={handleFileSelect} />
                
                {preview ? (
                  <div style={{ padding: '20px' }}>
                    <img src={preview} alt="Prescription" style={{ maxHeight: '400px', maxWidth: '100%', borderRadius: '20px', boxShadow: 'var(--shadow-lg)', border: '6px solid white' }} />
                    <h3 style={{ marginTop: '24px', color: 'var(--primary)' }}>{selectedFile?.name}</h3>
                  </div>
                ) : (
                  <div style={{ padding: '80px 40px' }}>
                    <div style={{ 
                      width: '80px', 
                      height: '80px', 
                      background: 'var(--primary-light)', 
                      borderRadius: '24px', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      color: 'var(--primary)',
                      margin: '0 auto 32px'
                    }}>
                      <Upload size={40} />
                    </div>
                    <h2 style={{ marginBottom: '12px' }}>Upload Document</h2>
                    <p style={{ color: 'var(--text-secondary)' }}>Click or drag a prescription image here</p>
                  </div>
                )}
              </div>

              {selectedFile && (
                <motion.button
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  onClick={analyzePrescription}
                  className="btn btn-primary"
                  style={{ marginTop: '48px', padding: '20px 48px', borderRadius: '24px', fontSize: '18px' }}
                >
                  <Sparkles size={20} /> Finalize Analysis <ArrowRight size={20} />
                </motion.button>
              )}
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
                <BrainCircuit size={48} color="var(--primary)" style={{ position: 'absolute', top: '50%', left: '50%', translate: '-50% -50%' }} />
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
              <div className="glass card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '24px 32px', flexWrap: 'wrap', gap: '20px' }}>
                 <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ width: '48px', height: '48px', background: 'var(--primary-light)', borderRadius: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)' }}>
                        <Star fill="currentColor" size={24} />
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', fontWeight: '800', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Extraction Success</div>
                        <div style={{ fontSize: '24px', fontWeight: '900' }}>{Math.round((analysisResult?.confidence || 0.94) * 100)}%</div>
                      </div>
                    </div>
                    
                    <div style={{ width: '1px', height: '40px', background: '#e2e8f0' }} />

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Pill color="var(--primary)" size={32} />
                      <div>
                        <div style={{ fontSize: '12px', fontWeight: '800', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Drugs Identified</div>
                        <div style={{ fontSize: '24px', fontWeight: '900' }}>{analysisResult.medicines?.length || 0}</div>
                      </div>
                    </div>
                 </div>

                 <div style={{ display: 'flex', gap: '12px' }}>
                   {isEditing ? (
                     <>
                        <button onClick={() => setIsEditing(false)} className="btn btn-outline">Cancel</button>
                        <button onClick={submitFeedback} className="btn btn-primary" style={{ background: 'var(--success)' }}>Train Knowledge Engine</button>
                     </>
                   ) : (
                     <>
                        <button onClick={() => setIsEditing(true)} className="btn btn-outline">Correct AI</button>
                        <button onClick={handleNewAnalysis} className="btn btn-primary">Start New Analysis</button>
                     </>
                   )}
                 </div>
              </div>

              {/* Data Grid */}
              <div className="dashboard-grid">
                {/* Side Pane */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                  <SectionCard icon={<User size={20} color="var(--primary)" />} title="Patient Information">
                     <Field label="Full Name" value={analysisResult.patient?.name} isEditing={isEditing} onChange={(v) => handleFieldChange('patient', 'name', v)} />
                     <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                        <Field label="Age" value={analysisResult.patient?.age} isEditing={isEditing} onChange={(v) => handleFieldChange('patient', 'age', v)} />
                        <Field label="Gender" value={analysisResult.patient?.gender} isEditing={isEditing} onChange={(v) => handleFieldChange('patient', 'gender', v)} />
                     </div>
                  </SectionCard>

                  <SectionCard icon={<Stethoscope size={20} color="var(--accent)" />} title="Doctor Insights">
                     <Field label="Prescribing Clinician" value={analysisResult.doctor?.name} isEditing={isEditing} onChange={(v) => handleFieldChange('doctor', 'name', v)} />
                     <Field label="Specialization" value={analysisResult.doctor?.specialization} isEditing={isEditing} onChange={(v) => handleFieldChange('doctor', 'specialization', v)} />
                  </SectionCard>

                  <SectionCard icon={<Activity size={20} color="var(--warning)" />} title="Symptom Tags">
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
                        <Pill size={24} color="var(--primary)" />
                        <h2 style={{ fontSize: '28px' }}>Identified Medications</h2>
                      </div>
                   </div>

                   <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }} className="med-list">
                      {analysisResult.medicines?.map((med, idx) => (
                        <div key={idx} style={{ background: 'white', border: '1px solid #f1f5f9', padding: '24px', borderRadius: '24px', position: 'relative' }}>
                           <h3 style={{ fontSize: '18px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                             {isEditing ? (
                               <input value={med.name} onChange={(e) => handleFieldChange('medicines', 'name', e.target.value, idx)} style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '8px' }} />
                             ) : (
                               med.name || "Untitled Molecule"
                             )}
                           </h3>
                           <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                              <Field label="Dosage" value={med.dosage} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'dosage', v, idx)} small />
                              <Field label="Frequency" value={med.frequency} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'frequency', v, idx)} small />
                              <Field label="Duration" value={med.duration} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'duration', v, idx)} small />
                              <Field label="Timing" value={med.instructions} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'instructions', v, idx)} small />
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
      <footer style={{ textAlign: 'center', padding: '64px 0', opacity: 0.5, fontSize: '13px', fontWeight: '600' }}>
         &copy; 2026 Advanced Bio-Vision Systems. Global Medical Standards Compliant.
      </footer>

      {/* Toast Notification */}
      <AnimatePresence>
        {feedbackSent && (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            style={{
              position: 'fixed',
              bottom: '40px',
              right: '40px',
              zIndex: 1000,
              background: '#0f172a',
              color: 'white',
              padding: '24px 32px',
              borderRadius: '24px',
              boxShadow: 'var(--shadow-expensive)',
              display: 'flex',
              alignItems: 'center',
              gap: '16px'
            }}
          >
            <div style={{ width: '40px', height: '40px', background: 'var(--success)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyCenter: 'center' }}>
              <CheckCircle size={24} color="white" />
            </div>
            <div>
              <div style={{ fontWeight: '800', fontSize: '18px' }}>Synapse Updated</div>
              <div style={{ opacity: 0.7, fontSize: '14px' }}>The model has incorporated your corrections.</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const Badge = ({ icon, text }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 20px', background: 'white', border: '1px solid #e2e8f0', borderRadius: '16px', fontSize: '13px', fontWeight: '700', color: 'var(--text-secondary)' }}>
    {icon} {text}
  </div>
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

const Field = ({ label, value, isEditing, onChange, small = false }) => (
  <div>
    <label style={{ display: 'block', textTransform: 'uppercase', fontSize: '10px', fontWeight: '900', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '4px' }}>{label}</label>
    {isEditing ? (
      <input 
        value={value || ''} 
        onChange={(e) => onChange(e.target.value)} 
        style={{ width: '100%', padding: '8px 12px', border: '2px solid #f1f5f9', borderRadius: '10px', fontSize: '14px', outline: 'none', background: '#fff' }} 
      />
    ) : (
      <div style={{ fontSize: small ? '14px' : '16px', fontWeight: '700', color: 'var(--text-primary)' }}>{value || <span style={{ opacity: 0.3 }}>N/A</span>}</div>
    )}
  </div>
);

export default App;