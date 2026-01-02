import React, { useState, useCallback } from 'react';
import { Upload, User, Pill, CheckCircle, AlertCircle, Star, Activity, Stethoscope, XCircle, Loader2, AlertTriangle, FileText, Camera } from 'lucide-react';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);

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
        setActiveTab("results");
      } else {
        setError(result.error || "Failed to analyze prescription");
      }
    } catch (error) {
      setError(`Failed to connect to server: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
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

  const getConfidence = (result) => {
    return result?.confidence || result?.confidence_score || 0;
  };

  return (
    <div style={{
      minHeight: '100vh',
      width: '100%',
      background: 'linear-gradient(135deg, #FFEFCA 0%, #F8D450 25%, #F89F1C 50%, #F8A80E 75%, #EA4235 100%)',
      backgroundAttachment: 'fixed',
      fontFamily: '"Inter", sans-serif',
      display: 'flex',
      flexDirection: 'column',
      margin: 0,
      padding: 0,
    }}>
      {/* Header */}
      <header style={{
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        padding: '24px 20px',
        boxShadow: '0 4px 25px rgba(0, 0, 0, 0.15)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        width: '100%',
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
        }}>
          <div>
            <h1 style={{
              fontSize: 'clamp(28px, 5vw, 42px)',
              fontWeight: '800',
              fontFamily: '"Outfit", sans-serif',
              background: 'linear-gradient(135deg, #F89F1C, #EA4235)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              margin: 0,
              letterSpacing: '-1px',
              textTransform: 'uppercase',
            }}>
              Prescription Analyzer AI
            </h1>
            <p style={{
              fontSize: 'clamp(14px, 2vw, 17px)',
              color: '#555',
              marginTop: '8px',
              fontWeight: '500',
            }}>
              Intelligent Medicine Extraction & Self-Learning AI
            </p>
          </div>
        </div>
      </header>

      <main style={{ 
        maxWidth: '1200px', 
        width: '100%',
        margin: '0 auto', 
        padding: '40px 20px',
        flex: 1,
      }}>
        {/* Error Alert */}
        {error && (
          <div style={{
            background: 'rgba(234, 66, 53, 0.1)',
            border: '2px solid rgba(234, 66, 53, 0.3)',
            borderRadius: '24px',
            padding: '20px',
            marginBottom: '30px',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
            animation: 'slideDown 0.4s ease-out',
          }}>
            <AlertTriangle size={28} color="#EA4235" />
            <p style={{ margin: 0, color: '#EA4235', flex: 1, fontSize: '16px', fontWeight: '500' }}>{error}</p>
            <button
              onClick={() => setError(null)}
              style={{
                background: 'rgba(234, 66, 53, 0.1)',
                border: 'none',
                cursor: 'pointer',
                padding: '8px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <XCircle size={24} color="#EA4235" />
            </button>
          </div>
        )}

        {/* Feedback Success Toast */}
        {feedbackSent && (
          <div style={{
            position: 'fixed',
            top: '100px',
            right: '20px',
            background: '#10b981',
            color: 'white',
            padding: '16px 24px',
            borderRadius: '16px',
            boxShadow: '0 10px 25px rgba(16, 185, 129, 0.3)',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            zIndex: 1000,
            animation: 'slideIn 0.5s ease-out',
          }}>
            <CheckCircle size={24} />
            <span style={{ fontWeight: '700' }}>AI Knowledge Updated! Thanks for your feedback.</span>
          </div>
        )}

        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          marginBottom: '40px',
        }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(20px)',
            borderRadius: '30px',
            padding: '10px',
            display: 'flex',
            gap: '12px',
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.12)',
            border: '1px solid rgba(255, 255, 255, 0.5)',
          }}>
            {['upload', 'analysis', 'results'].map((tab) => (
              <button
                key={tab}
                onClick={() => !isAnalyzing && (tab !== 'results' || analysisResult) && setActiveTab(tab)}
                disabled={tab === 'results' && !analysisResult}
                style={{
                  padding: '14px 36px',
                  borderRadius: '22px',
                  border: 'none',
                  fontSize: '16px',
                  fontWeight: '700',
                  cursor: (tab === 'results' && !analysisResult) ? 'not-allowed' : 'pointer',
                  background: activeTab === tab
                    ? 'linear-gradient(135deg, #F89F1C, #EA4235)'
                    : 'transparent',
                  color: activeTab === tab ? '#fff' : '#666',
                  transition: 'all 0.3s ease',
                  opacity: tab === 'results' && !analysisResult ? 0.5 : 1,
                  textTransform: 'capitalize',
                  transform: activeTab === tab ? 'scale(1.05)' : 'scale(1)',
                }}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        {/* Main Content Area */}
        {activeTab === "upload" && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(15px)',
            borderRadius: '40px',
            padding: 'clamp(30px, 8vw, 60px)',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.2)',
            maxWidth: '900px',
            margin: '0 auto',
            animation: 'fadeIn 0.6s ease-out',
          }}>
            <div
              onClick={() => document.getElementById("file-upload").click()}
              style={{
                border: `4px dashed ${selectedFile ? '#10b981' : '#F89F1C'}`,
                borderRadius: '32px',
                padding: 'clamp(40px, 10vw, 80px)',
                textAlign: 'center',
                cursor: 'pointer',
                background: selectedFile
                  ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.15))'
                  : 'linear-gradient(135deg, rgba(248, 159, 28, 0.08), rgba(248, 168, 14, 0.15))',
                transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            >
              <input
                type="file"
                id="file-upload"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              
              {preview ? (
                <div>
                  <img
                    src={preview}
                    alt="Preview"
                    style={{
                      maxHeight: '450px',
                      maxWidth: '100%',
                      borderRadius: '24px',
                      marginBottom: '30px',
                      boxShadow: '0 15px 40px rgba(0, 0, 0, 0.2)',
                      border: '8px solid white',
                    }}
                  />
                  <h3 style={{ color: '#10b981', fontWeight: '700' }}>{selectedFile.name}</h3>
                </div>
              ) : (
                <div>
                  <Upload size={50} color="#F89F1C" style={{ marginBottom: '20px' }} />
                  <h3 style={{ fontSize: '24px', fontWeight: '800' }}>Upload Prescription</h3>
                  <p>Drag & drop or click to select image</p>
                </div>
              )}
            </div>

            {selectedFile && (
              <div style={{ textAlign: 'center', marginTop: '40px' }}>
                <button
                  onClick={analyzePrescription}
                  disabled={isAnalyzing}
                  style={{
                    padding: '20px 60px',
                    borderRadius: '24px',
                    border: 'none',
                    fontSize: '20px',
                    fontWeight: '800',
                    cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                    background: 'linear-gradient(135deg, #F89F1C, #EA4235)',
                    color: '#fff',
                    boxShadow: '0 12px 35px rgba(234, 66, 53, 0.4)',
                  }}
                >
                  {isAnalyzing ? "Processing AI..." : "Start AI Analysis"}
                </button>
              </div>
            )}
          </div>
        )}

        {activeTab === "analysis" && (
          <div style={{
            background: 'white',
            borderRadius: '40px',
            padding: '80px 40px',
            textAlign: 'center',
            maxWidth: '700px',
            margin: '0 auto',
            boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
            animation: 'fadeIn 0.5s ease-out',
          }}>
            <div style={{ position: 'relative', width: '120px', height: '120px', margin: '0 auto 40px' }}>
              <Loader2 size={120} color="#F89F1C" style={{ animation: 'spin 2s linear infinite' }} />
              <Activity size={40} color="#EA4235" style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', animation: 'pulse 1.5s ease-in-out infinite' }} />
            </div>
            <h2 style={{ fontSize: '32px', fontWeight: '800' }}>Analyzing Prescription...</h2>
            <p style={{ color: '#666', fontSize: '18px' }}>Our AI models are extracting information and learning from your document.</p>
          </div>
        )}

        {activeTab === "results" && analysisResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px', animation: 'fadeIn 0.6s ease-out' }}>
            {/* Header / Summary Box */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.95)',
              borderRadius: '30px',
              padding: '30px',
              boxShadow: '0 10px 40px rgba(0, 0, 0, 0.1)',
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '20px',
            }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                  <Star size={24} color="#F8A80E" fill="#F8A80E" />
                  <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '800' }}>Analysis Results</h2>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <div style={{ width: '150px', height: '8px', background: '#f0f0f0', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: `${getConfidence(analysisResult) * 100}%`, height: '100%', background: 'linear-gradient(90deg, #F89F1C, #EA4235)' }} />
                  </div>
                  <span style={{ fontWeight: '800', color: '#EA4235' }}>{Math.round(getConfidence(analysisResult) * 100)}% Confidence</span>
                </div>
              </div>
              
              <div style={{ display: 'flex', gap: '12px' }}>
                {isEditing ? (
                  <>
                    <button onClick={() => setIsEditing(false)} style={{ padding: '12px 24px', borderRadius: '15px', border: '2px solid #ddd', background: 'white', fontWeight: '700' }}>Cancel</button>
                    <button onClick={submitFeedback} style={{ padding: '12px 24px', borderRadius: '15px', border: 'none', background: '#10b981', color: 'white', fontWeight: '700' }}>Update & Learn</button>
                  </>
                ) : (
                  <>
                    <button onClick={() => setIsEditing(true)} style={{ padding: '12px 24px', borderRadius: '15px', border: 'none', background: '#F89F1C', color: 'white', fontWeight: '700' }}>Correct Results</button>
                    <button onClick={handleNewAnalysis} style={{ padding: '12px 24px', borderRadius: '15px', border: 'none', background: '#333', color: 'white', fontWeight: '700' }}>New Analysis</button>
                  </>
                )}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '30px' }}>
              {/* Patient Card */}
              <div style={{ background: 'white', borderRadius: '30px', padding: '30px', boxShadow: '0 10px 30px rgba(0, 0, 0, 0.05)' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px', borderBottom: '2px solid #f8f9fa', paddingBottom: '12px' }}>
                  <User color="#3b82f6" /> Patient Info
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <EditableField label="Name" value={analysisResult.patient?.name} isEditing={isEditing} onChange={(v) => handleFieldChange('patient', 'name', v)} />
                  <EditableField label="Age" value={analysisResult.patient?.age} isEditing={isEditing} onChange={(v) => handleFieldChange('patient', 'age', v)} />
                  <EditableField label="Gender" value={analysisResult.patient?.gender} isEditing={isEditing} onChange={(v) => handleFieldChange('patient', 'gender', v)} />
                </div>
              </div>

              {/* Doctor Card */}
              <div style={{ background: 'white', borderRadius: '30px', padding: '30px', boxShadow: '0 10px 30px rgba(0, 0, 0, 0.05)' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px', borderBottom: '2px solid #f8f9fa', paddingBottom: '12px' }}>
                  <Stethoscope color="#10b981" /> Doctor Info
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <EditableField label="Name" value={analysisResult.doctor?.name} isEditing={isEditing} onChange={(v) => handleFieldChange('doctor', 'name', v)} />
                  <EditableField label="Specialization" value={analysisResult.doctor?.specialization} isEditing={isEditing} onChange={(v) => handleFieldChange('doctor', 'specialization', v)} />
                  <EditableField label="License" value={analysisResult.doctor?.registration || analysisResult.doctor?.registration_number} isEditing={isEditing} onChange={(v) => handleFieldChange('doctor', 'registration', v)} />
                </div>
              </div>
            </div>

            {/* Diagnosis / Symptoms Section */}
            {(analysisResult.diagnosis?.length > 0 || isEditing) && (
              <div style={{ background: 'white', borderRadius: '30px', padding: '30px', boxShadow: '0 10px 30px rgba(0, 0, 0, 0.05)' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', borderBottom: '2px solid #f8f9fa', paddingBottom: '12px' }}>
                  <Activity color="#ea4235" /> Symptoms / Diagnosis
                </h3>
                {isEditing ? (
                  <textarea
                    value={Array.isArray(analysisResult.diagnosis) ? analysisResult.diagnosis.join(', ') : analysisResult.diagnosis || ''}
                    onChange={(e) => handleFieldChange(null, 'diagnosis', e.target.value.split(',').map(s => s.trim()))}
                    placeholder="Enter symptoms or diagnosis (comma separated)"
                    style={{
                      width: '100%',
                      padding: '15px',
                      borderRadius: '15px',
                      border: '2px solid #eee',
                      minHeight: '100px',
                      fontFamily: 'inherit',
                      outline: 'none',
                      fontSize: '14px'
                    }}
                  />
                ) : (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                    {Array.isArray(analysisResult.diagnosis) && analysisResult.diagnosis.length > 0 ? (
                      analysisResult.diagnosis.map((d, i) => (
                        <span key={i} style={{ background: '#fef2f2', color: '#ea4235', padding: '8px 16px', borderRadius: '12px', fontWeight: '700', fontSize: '14px', border: '1px solid #fee2e2' }}>
                          {d}
                        </span>
                      ))
                    ) : (
                      <span style={{ color: '#999', fontStyle: 'italic' }}>No specific diagnosis detected</span>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Medicines List */}
            <div style={{ background: 'white', borderRadius: '30px', padding: '40px', boxShadow: '0 15px 50px rgba(0, 0, 0, 0.08)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: 0 }}>
                  <Pill color="#8b5cf6" /> Medicines Detected
                </h3>
                {isEditing && (
                  <button 
                    onClick={() => {
                      const newRes = { ...analysisResult };
                      newRes.medicines = [...(newRes.medicines || [])];
                      newRes.medicines.push({ name: '', dosage: '', frequency: '', duration: '', instructions: '' });
                      setAnalysisResult(newRes);
                    }}
                    style={{ background: '#f5f3ff', border: 'none', padding: '10px 20px', borderRadius: '12px', color: '#8b5cf6', fontWeight: '700' }}
                  >
                    + Add Medicine
                  </button>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '25px' }}>
                {analysisResult.medicines?.map((med, idx) => (
                  <div key={idx} style={{ 
                    background: '#fcfcfc', 
                    padding: '24px', 
                    borderRadius: '24px', 
                    border: '1px solid #f0f0f0',
                    transition: 'all 0.3s'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '18px', gap: '10px' }}>
                      {isEditing ? (
                        <input 
                          value={med.name} 
                          onChange={(e) => handleFieldChange('medicines', 'name', e.target.value, idx)}
                          placeholder="Medicine Name"
                          style={{ fontWeight: '800', fontSize: '18px', border: '2px solid #eee', borderRadius: '10px', padding: '6px 12px', width: '100%', outline: 'none' }}
                        />
                      ) : (
                        <span style={{ fontWeight: '800', fontSize: '20px', color: '#333' }}>{med.name || `Medicine ${idx + 1}`}</span>
                      )}
                      {isEditing && (
                        <button onClick={() => {
                          const newRes = { ...analysisResult };
                          newRes.medicines.splice(idx, 1);
                          setAnalysisResult(newRes);
                        }} style={{ color: '#ea4235', background: 'rgba(234, 66, 53, 0.1)', border: 'none', padding: '8px', borderRadius: '10px' }}>
                          <XCircle size={18} />
                        </button>
                      )}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                      <EditableField label="Dosage" value={med.dosage} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'dosage', v, idx)} small />
                      <EditableField label="Frequency" value={med.frequency} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'frequency', v, idx)} small />
                      <EditableField label="Duration" value={med.duration} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'duration', v, idx)} small />
                      <EditableField label="Notes" value={med.instructions || med.timing} isEditing={isEditing} onChange={(v) => handleFieldChange('medicines', 'instructions', v, idx)} small />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Raw Text Toggle (Debug) */}
            <details style={{ background: 'rgba(0,0,0,0.05)', padding: '20px', borderRadius: '20px', cursor: 'pointer' }}>
              <summary style={{ fontWeight: '700', color: '#666' }}>View Raw Extracted Text</summary>
              <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '13px', marginTop: '15px', color: '#444' }}>
                {analysisResult.raw_text}
              </pre>
            </details>
          </div>
        )}
      </main>

      <footer style={{ padding: '40px 20px', textAlign: 'center', color: 'rgba(255,255,255,0.8)', fontSize: '14px', fontWeight: '600' }}>
        AI-Powered Prescription Analyzer • System Upgrades Automatically via Feedback
      </footer>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes pulse { 0% { transform: translate(-50%, -50%) scale(1); opacity: 0.8; } 50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; } 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.8; } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
      `}</style>
    </div>
  );
};

const EditableField = ({ label, value, isEditing, onChange, small = false }) => (
  <div style={{ borderBottom: small ? 'none' : '1px solid #f0f0f0', paddingBottom: small ? 0 : '12px' }}>
    <div style={{ color: '#999', fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '0.5px' }}>{label}</div>
    {isEditing ? (
      <input
        type="text"
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%',
          padding: '8px 12px',
          borderRadius: '10px',
          border: '2px solid #eee',
          fontSize: '14px',
          fontFamily: 'inherit',
          outline: 'none',
          background: '#fff'
        }}
      />
    ) : (
      <div style={{ fontWeight: '700', color: '#333', fontSize: small ? '14px' : '16px' }}>{value || 'Not detected'}</div>
    )}
  </div>
);

export default App;