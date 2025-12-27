import React, { useState, useCallback } from 'react';
import { Upload, User, Pill, CheckCircle, AlertCircle, Star, Activity, Stethoscope, XCircle, Loader2, AlertTriangle, FileText, Camera } from 'lucide-react';

const PrescriptionAnalyzer = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);

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
  };

  const getConfidence = (result) => {
    return result?.confidence || result?.confidence_score || 0;
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #FFEFCA 0%, #F8D450 25%, #F89F1C 50%, #F8A80E 75%, #EA4235 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      {/* Header */}
      <div style={{
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        padding: '20px',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '20px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <div>
              <h1 style={{
                fontSize: 'clamp(24px, 4vw, 32px)',
                fontWeight: '800',
                background: 'linear-gradient(135deg, #F89F1C, #EA4235)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                margin: 0,
                lineHeight: 1.2,
              }}>
                PRESCRIPTION ANALYZER AI
              </h1>
              <p style={{
                fontSize: '14px',
                color: '#666',
                margin: '4px 0 0 0',
              }}>
                AI-Powered Prescription Analysis and Medicine Extraction
              </p>
            </div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
        {/* Error Alert */}
        {error && (
          <div style={{
            background: 'rgba(234, 66, 53, 0.1)',
            border: '2px solid rgba(234, 66, 53, 0.3)',
            borderRadius: '20px',
            padding: '16px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}>
            <AlertTriangle size={24} color="#EA4235" />
            <p style={{ margin: 0, color: '#EA4235', flex: 1 }}>{error}</p>
            <button
              onClick={() => setError(null)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '4px',
              }}
            >
              <XCircle size={20} color="#EA4235" />
            </button>
          </div>
        )}

        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          marginBottom: '30px',
        }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(20px)',
            borderRadius: '25px',
            padding: '8px',
            display: 'flex',
            gap: '8px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}>
            {['upload', 'analysis', 'results'].map((tab) => (
              <button
                key={tab}
                onClick={() => !isAnalyzing && (tab !== 'results' || analysisResult) && setActiveTab(tab)}
                disabled={tab === 'results' && !analysisResult}
                style={{
                  padding: '12px 28px',
                  borderRadius: '18px',
                  border: 'none',
                  fontSize: '15px',
                  fontWeight: '600',
                  cursor: activeTab === tab || (tab === 'results' && !analysisResult) ? 'default' : 'pointer',
                  background: activeTab === tab
                    ? 'linear-gradient(135deg, #F89F1C, #EA4235)'
                    : 'transparent',
                  color: activeTab === tab ? '#fff' : '#666',
                  transition: 'all 0.3s ease',
                  opacity: tab === 'results' && !analysisResult ? 0.5 : 1,
                  textTransform: 'capitalize',
                }}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        {/* Upload Section */}
        {activeTab === "upload" && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            borderRadius: '30px',
            padding: 'clamp(20px, 4vw, 40px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            maxWidth: '800px',
            margin: '0 auto',
          }}>
            <div
              onClick={() => document.getElementById("file-upload").click()}
              style={{
                border: `3px dashed ${selectedFile ? '#10b981' : '#F89F1C'}`,
                borderRadius: '24px',
                padding: 'clamp(30px, 6vw, 60px)',
                textAlign: 'center',
                cursor: 'pointer',
                background: selectedFile
                  ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(16, 185, 129, 0.1))'
                  : 'linear-gradient(135deg, rgba(248, 159, 28, 0.05), rgba(248, 168, 14, 0.1))',
                transition: 'all 0.3s ease',
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
                      maxHeight: '300px',
                      maxWidth: '100%',
                      borderRadius: '16px',
                      marginBottom: '20px',
                      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
                    }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#10b981', fontWeight: '600', marginBottom: '12px' }}>
                    <CheckCircle size={24} />
                    <span>{selectedFile.name}</span>
                  </div>
                  <p style={{ color: '#666', fontSize: '14px', margin: 0 }}>
                    Click to change file
                  </p>
                </div>
              ) : (
                <div>
                  <div style={{
                    width: '80px',
                    height: '80px',
                    background: 'linear-gradient(135deg, #F89F1C, #EA4235)',
                    borderRadius: '20px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 20px',
                    boxShadow: '0 4px 20px rgba(248, 159, 28, 0.3)',
                  }}>
                    <Upload size={40} color="#fff" />
                  </div>
                  <h3 style={{
                    fontSize: 'clamp(18px, 3vw, 24px)',
                    fontWeight: '700',
                    color: '#333',
                    marginBottom: '12px',
                  }}>
                    Upload Prescription
                  </h3>
                  <p style={{ color: '#666', marginBottom: '20px', fontSize: 'clamp(14px, 2vw, 16px)' }}>
                    Drag & drop or click to select
                  </p>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '20px',
                    flexWrap: 'wrap',
                    fontSize: '14px',
                    color: '#999',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Camera size={18} />
                      <span>Photos</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <FileText size={18} />
                      <span>Scans</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {selectedFile && (
              <div style={{ textAlign: 'center', marginTop: '30px' }}>
                <button
                  onClick={analyzePrescription}
                  disabled={isAnalyzing}
                  style={{
                    padding: '16px 40px',
                    borderRadius: '16px',
                    border: 'none',
                    fontSize: '16px',
                    fontWeight: '600',
                    cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                    background: isAnalyzing
                      ? '#ccc'
                      : 'linear-gradient(135deg, #F89F1C, #EA4235)',
                    color: '#fff',
                    boxShadow: isAnalyzing ? 'none' : '0 4px 20px rgba(248, 159, 28, 0.4)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    margin: '0 auto',
                    transition: 'all 0.3s ease',
                  }}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Activity size={20} />
                      Analyze Prescription
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Analysis Progress */}
        {activeTab === "analysis" && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            borderRadius: '30px',
            padding: '60px 40px',
            textAlign: 'center',
            maxWidth: '600px',
            margin: '0 auto',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          }}>
            {isAnalyzing ? (
              <>
                <Loader2 size={64} color="#F89F1C" style={{ animation: 'spin 1s linear infinite', marginBottom: '20px' }} />
                <h2 style={{ fontSize: '28px', fontWeight: '700', color: '#333', marginBottom: '12px' }}>
                  Analyzing Prescription
                </h2>
                <p style={{ color: '#666', fontSize: '16px' }}>
                  Our AI is processing your prescription...
                </p>
              </>
            ) : analysisResult ? (
              <>
                <CheckCircle size={64} color="#10b981" style={{ marginBottom: '20px' }} />
                <h2 style={{ fontSize: '28px', fontWeight: '700', color: '#333', marginBottom: '12px' }}>
                  Analysis Complete!
                </h2>
                <p style={{ color: '#666', fontSize: '16px' }}>
                  Your prescription has been analyzed successfully.
                </p>
              </>
            ) : (
              <>
                <AlertCircle size={64} color="#EA4235" style={{ marginBottom: '20px' }} />
                <h2 style={{ fontSize: '28px', fontWeight: '700', color: '#333', marginBottom: '12px' }}>
                  Analysis Failed
                </h2>
                <p style={{ color: '#EA4235', fontSize: '16px' }}>
                  {error || "Unable to analyze prescription"}
                </p>
              </>
            )}
          </div>
        )}

        {/* Results Section */}
        {activeTab === "results" && analysisResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Debug Info */}
            <div style={{
              background: 'rgba(14, 165, 233, 0.1)',
              border: '2px solid rgba(14, 165, 233, 0.3)',
              borderRadius: '20px',
              padding: '20px',
            }}>
              <h3 style={{ color: '#0ea5e9', marginBottom: '12px', fontSize: '18px', fontWeight: '600' }}>
                🔍 Debug Info
              </h3>
              <pre style={{
                fontSize: '12px',
                overflow: 'auto',
                maxHeight: '200px',
                background: 'rgba(255, 255, 255, 0.5)',
                padding: '12px',
                borderRadius: '12px',
                margin: 0,
              }}>
                {JSON.stringify(analysisResult, null, 2)}
              </pre>
            </div>

            {/* Confidence Score */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(10px)',
              borderRadius: '20px',
              padding: '24px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Star size={24} color="#F8A80E" />
                  <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#333', margin: 0 }}>
                    Analysis Confidence
                  </h3>
                </div>
                <span style={{
                  fontSize: '28px',
                  fontWeight: '800',
                  background: 'linear-gradient(135deg, #F89F1C, #EA4235)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}>
                  {Math.round(getConfidence(analysisResult) * 100)}%
                </span>
              </div>
              <div style={{
                height: '16px',
                background: '#f0f0f0',
                borderRadius: '8px',
                overflow: 'hidden',
              }}>
                <div
                  style={{
                    height: '100%',
                    width: `${getConfidence(analysisResult) * 100}%`,
                    background: 'linear-gradient(90deg, #F89F1C, #EA4235)',
                    transition: 'width 1s ease',
                  }}
                />
              </div>
            </div>

            {/* Patient and Doctor Info */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '20px',
            }}>
              {/* Patient */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(10px)',
                borderRadius: '20px',
                padding: '24px',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <User size={24} color="#fff" />
                  </div>
                  <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#333', margin: 0 }}>
                    Patient
                  </h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <InfoRow label="Name" value={analysisResult.patient?.name || "Not detected"} />
                  <InfoRow label="Age" value={analysisResult.patient?.age || "Not detected"} />
                  <InfoRow label="Gender" value={analysisResult.patient?.gender || "Not detected"} />
                </div>
              </div>

              {/* Doctor */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(10px)',
                borderRadius: '20px',
                padding: '24px',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    background: 'linear-gradient(135deg, #10b981, #059669)',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <Stethoscope size={24} color="#fff" />
                  </div>
                  <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#333', margin: 0 }}>
                    Doctor
                  </h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <InfoRow label="Name" value={analysisResult.doctor?.name || "Not detected"} />
                  <InfoRow label="Specialization" value={analysisResult.doctor?.specialization || "Not specified"} />
                  <InfoRow label="License" value={analysisResult.doctor?.registration || analysisResult.doctor?.registration_number || "Not detected"} />
                </div>
              </div>
            </div>

            {/* Medicines */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(10px)',
              borderRadius: '20px',
              padding: '24px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <Pill size={24} color="#fff" />
                  </div>
                  <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#333', margin: 0 }}>
                    Medicines
                  </h3>
                </div>
                <span style={{
                  padding: '8px 16px',
                  background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(124, 58, 237, 0.1))',
                  borderRadius: '12px',
                  color: '#8b5cf6',
                  fontWeight: '600',
                  fontSize: '14px',
                }}>
                  {analysisResult.medicines?.length || 0} items
                </span>
              </div>

              {analysisResult.medicines && analysisResult.medicines.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {analysisResult.medicines.map((medicine, index) => (
                    <div
                      key={index}
                      style={{
                        border: '2px solid #f0f0f0',
                        borderRadius: '16px',
                        padding: '20px',
                        transition: 'all 0.3s ease',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                        <h4 style={{ fontSize: '18px', fontWeight: '700', color: '#333', margin: 0 }}>
                          {medicine.name || `Medicine ${index + 1}`}
                        </h4>
                        <span style={{
                          padding: '6px 12px',
                          borderRadius: '8px',
                          fontSize: '12px',
                          fontWeight: '600',
                          background: medicine.available !== false
                            ? 'linear-gradient(135deg, #dcfce7, #bbf7d0)'
                            : 'linear-gradient(135deg, #fee2e2, #fecaca)',
                          color: medicine.available !== false ? '#166534' : '#991b1b',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                        }}>
                          {medicine.available !== false ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                          {medicine.available !== false ? 'Available' : 'Check'}
                        </span>
                      </div>
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                        gap: '12px',
                        fontSize: '14px',
                      }}>
                        <div>
                          <span style={{ color: '#999', display: 'block', marginBottom: '4px' }}>Dosage</span>
                          <span style={{ fontWeight: '600', color: '#333' }}>{medicine.dosage || "As prescribed"}</span>
                        </div>
                        <div>
                          <span style={{ color: '#999', display: 'block', marginBottom: '4px' }}>Frequency</span>
                          <span style={{ fontWeight: '600', color: '#333' }}>{medicine.frequency || "As directed"}</span>
                        </div>
                        <div>
                          <span style={{ color: '#999', display: 'block', marginBottom: '4px' }}>Duration</span>
                          <span style={{ fontWeight: '600', color: '#333' }}>{medicine.duration || "As prescribed"}</span>
                        </div>
                      </div>
                      {(medicine.instructions || medicine.timing) && (
                        <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #f0f0f0' }}>
                          <span style={{ color: '#999', fontSize: '14px' }}>Instructions: </span>
                          <span style={{ color: '#333', fontSize: '14px' }}>{medicine.instructions || medicine.timing}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ textAlign: 'center', color: '#999', padding: '40px 0' }}>
                  No medicines detected
                </p>
              )}
            </div>

            {/* New Analysis Button */}
            <div style={{ textAlign: 'center' }}>
              <button
                onClick={handleNewAnalysis}
                style={{
                  padding: '16px 40px',
                  borderRadius: '16px',
                  border: 'none',
                  fontSize: '16px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  background: 'linear-gradient(135deg, #F89F1C, #EA4235)',
                  color: '#fff',
                  boxShadow: '0 4px 20px rgba(248, 159, 28, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  margin: '0 auto',
                  transition: 'all 0.3s ease',
                }}
              >
                <Upload size={20} />
                Analyze New Prescription
              </button>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
          body {
            overflow-x: hidden;
          }
        }
      `}</style>
    </div>
  );
};

const InfoRow = ({ label, value }) => (
  <div style={{
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '12px',
    borderBottom: '1px solid #f0f0f0',
  }}>
    <span style={{ color: '#999', fontSize: '14px' }}>{label}</span>
    <span style={{ fontWeight: '600', color: '#333', fontSize: '14px' }}>{value}</span>
  </div>
);

export default PrescriptionAnalyzer;