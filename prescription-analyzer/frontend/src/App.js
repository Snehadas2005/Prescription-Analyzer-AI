import React, { useState, useCallback } from "react";
import {
  Upload,
  User,
  Pill,
  CheckCircle,
  AlertCircle,
  Star,
  Activity,
  Stethoscope,
  Heart,
  XCircle,
  Loader2,
  AlertTriangle
} from "lucide-react";
import "./App.css";

const EnhancedPrescriptionAnalyzer = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [error, setError] = useState(null);

  // API base URL - CRITICAL: Make sure this matches your backend
  const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8080/api/v1";

  const handleFileSelect = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith("image/")) {
        setSelectedFile(file);
        setAnalysisResult(null);
        setError(null);
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

      console.log("📤 Uploading to:", `${API_BASE_URL}/upload`);
      
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      console.log("📥 Response status:", response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("📊 Full response:", JSON.stringify(result, null, 2));
      
      // CRITICAL: Check both formats (backend might wrap in "data")
      const actualData = result.data || result;
      
      console.log("✅ Processed data:", {
        success: result.success,
        prescriptionId: actualData.prescription_id,
        patientName: actualData.patient?.name,
        doctorName: actualData.doctor?.name,
        medicinesCount: actualData.medicines?.length || 0,
        confidence: actualData.confidence || actualData.confidence_score
      });
      
      if (result.success) {
        // Store the ACTUAL data structure
        setAnalysisResult(actualData);
        setActiveTab("results");
        
        console.log("🎉 Analysis successful!");
        console.log("   Patient:", actualData.patient?.name);
        console.log("   Doctor:", actualData.doctor?.name);
        console.log("   Medicines:", actualData.medicines?.length);
        console.log("   Confidence:", actualData.confidence || actualData.confidence_score);
      } else {
        setError(result.error || "Failed to analyze prescription");
        console.error("❌ Analysis failed:", result.error);
      }
    } catch (error) {
      console.error("❌ Error during analysis:", error);
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
  };

  // Helper to safely get confidence score
  const getConfidence = (result) => {
    return result?.confidence || result?.confidence_score || 0;
  };

  const ConfidenceBar = ({ score }) => {
    const percentage = Math.round(score * 100);
    const color = score >= 0.8 ? "green" : score >= 0.6 ? "yellow" : "red";
    
    return (
      <div className="confidence-bar">
        <div
          className={`confidence-fill bg-gradient-to-r from-${color}-500 to-${color}-600`}
          style={{ width: `${percentage}%` }}
        />
        <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
          {percentage}%
        </div>
      </div>
    );
  };

  const TabButton = ({ tab, isActive, isDisabled, children }) => (
    <button
      onClick={() => !isDisabled && setActiveTab(tab)}
      disabled={isDisabled}
      className={`tab-button ${
        isActive ? "active" : isDisabled ? "disabled" : "inactive"
      }`}
    >
      {children}
    </button>
  );

  const ErrorAlert = ({ message, onClose }) => (
    <div className="error-alert">
      <div className="error-content">
        <AlertTriangle className="error-icon" size={20} />
        <p className="error-message">{message}</p>
        {onClose && (
          <button onClick={onClose} className="error-close">
            <XCircle size={20} />
          </button>
        )}
      </div>
    </div>
  );

  return (
    <div className="app-container">
      <div className="main-container">
        {/* Header */}
        <div className="app-header">
          <div className="app-title">
            <Stethoscope size={48} color="#2563eb" />
            <h1>AI Prescription Analyzer</h1>
          </div>
          <p className="app-subtitle">
            Upload your prescription for instant AI-powered analysis
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <ErrorAlert message={error} onClose={() => setError(null)} />
        )}

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <div className="tab-container">
            <TabButton 
              tab="upload" 
              isActive={activeTab === "upload"}
              isDisabled={false}
            >
              Upload
            </TabButton>
            <TabButton 
              tab="analysis" 
              isActive={activeTab === "analysis"}
              isDisabled={false}
            >
              Analysis
            </TabButton>
            <TabButton 
              tab="results" 
              isActive={activeTab === "results"}
              isDisabled={!analysisResult}
            >
              Results
            </TabButton>
          </div>
        </div>

        {/* Upload Section */}
        {activeTab === "upload" && (
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <div className="card">
              <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                <Upload size={64} color="#3b82f6" style={{ margin: '0 auto 1rem' }} />
                <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '0.5rem' }}>
                  Upload Prescription
                </h2>
                <p style={{ color: '#6b7280' }}>
                  Select a clear image of your prescription for analysis
                </p>
              </div>

              <div 
                className={`upload-area ${selectedFile ? 'has-file' : ''}`}
                onClick={() => document.getElementById("prescription-upload").click()}
              >
                <input
                  type="file"
                  id="prescription-upload"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
                
                {selectedFile ? (
                  <div>
                    <CheckCircle size={48} color="#10b981" style={{ margin: '0 auto 1rem' }} />
                    <p style={{ color: '#059669', fontWeight: '500', fontSize: '1.125rem' }}>
                      {selectedFile.name}
                    </p>
                    <p style={{ color: '#6b7280', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                      Click to change file
                    </p>
                  </div>
                ) : (
                  <div>
                    <Upload size={48} color="#9ca3af" style={{ margin: '0 auto 1rem' }} />
                    <p style={{ color: '#4b5563', fontSize: '1.125rem' }}>
                      Click to select prescription image
                    </p>
                    <p style={{ color: '#9ca3af', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                      Supports JPEG, PNG, TIFF formats (Max 10MB)
                    </p>
                  </div>
                )}
              </div>

              {selectedFile && (
                <div style={{ textAlign: 'center', marginTop: '2rem' }}>
                  <button
                    onClick={analyzePrescription}
                    disabled={isAnalyzing}
                    className={`btn ${isAnalyzing ? '' : 'btn-primary'}`}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 size={20} className="loading-spinner" style={{ marginRight: '0.5rem' }} />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Activity size={20} style={{ marginRight: '0.5rem' }} />
                        Analyze Prescription
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Analysis Progress */}
        {activeTab === "analysis" && (
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <div className="card">
              <div style={{ textAlign: 'center' }}>
                {isAnalyzing ? (
                  <div>
                    <div className="loading-spinner" style={{ margin: '0 auto 2rem' }} />
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '0.5rem' }}>
                      Analyzing Prescription
                    </h2>
                    <p style={{ color: '#6b7280', marginBottom: '1.5rem' }}>
                      Our AI is processing your prescription image...
                    </p>
                  </div>
                ) : analysisResult ? (
                  <div>
                    <CheckCircle size={64} color="#10b981" style={{ margin: '0 auto 1rem' }} />
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '0.5rem' }}>
                      Analysis Complete!
                    </h2>
                    <p style={{ color: '#6b7280' }}>
                      Your prescription has been successfully analyzed.
                    </p>
                  </div>
                ) : (
                  <div>
                    <AlertCircle size={64} color="#ef4444" style={{ margin: '0 auto 1rem' }} />
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '0.5rem' }}>
                      Analysis Failed
                    </h2>
                    <p style={{ color: '#ef4444' }}>
                      {error || "Unable to analyze prescription"}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {activeTab === "results" && analysisResult && (
          <div className="grid" style={{ gap: '1.5rem' }}>
            {/* Debug Info - Remove this after testing */}
            <div className="card" style={{ background: '#f0f9ff', border: '2px solid #0ea5e9' }}>
              <h3 style={{ color: '#0369a1', marginBottom: '1rem' }}>🔍 Debug Info (Remove after testing)</h3>
              <pre style={{ fontSize: '0.75rem', overflow: 'auto', maxHeight: '200px' }}>
                {JSON.stringify(analysisResult, null, 2)}
              </pre>
            </div>

            {/* Confidence Score */}
            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <Star size={24} color="#eab308" style={{ marginRight: '0.5rem' }} />
                  <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1f2937' }}>
                    Analysis Confidence
                  </h3>
                </div>
                <span style={{ fontSize: '1.5rem', fontWeight: '700', color: '#2563eb' }}>
                  {Math.round(getConfidence(analysisResult) * 100)}%
                </span>
              </div>
              <ConfidenceBar score={getConfidence(analysisResult)} />
            </div>

            {/* Patient and Doctor Info */}
            <div className="grid grid-cols-2" style={{ gap: '1.5rem' }}>
              <div className="card">
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
                  <User size={24} color="#3b82f6" style={{ marginRight: '0.5rem' }} />
                  <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1f2937' }}>
                    Patient Information
                  </h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>Name:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.patient?.name || "Not specified"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>Age:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.patient?.age || "Not specified"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>Gender:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.patient?.gender || "Not specified"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="card">
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
                  <Stethoscope size={24} color="#10b981" style={{ marginRight: '0.5rem' }} />
                  <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1f2937' }}>
                    Doctor Information
                  </h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>Name:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.doctor?.name || "Not specified"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>Specialization:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.doctor?.specialization || "Not specified"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>License:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.doctor?.registration || analysisResult.doctor?.registration_number || "Not specified"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Medicines */}
            <div className="card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <Pill size={24} color="#8b5cf6" style={{ marginRight: '0.5rem' }} />
                  <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1f2937' }}>
                    Prescribed Medicines
                  </h3>
                </div>
                <span style={{ color: '#6b7280' }}>
                  {analysisResult.medicines?.length || 0} items
                </span>
              </div>

              {analysisResult.medicines && analysisResult.medicines.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {analysisResult.medicines.map((medicine, index) => (
                    <div key={index} className="medicine-card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                        <h4 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#1f2937' }}>
                          {medicine.name || `Medicine ${index + 1}`}
                        </h4>
                        <div className={`status-badge ${
                          medicine.available !== false ? 'status-available' : 'status-unavailable'
                        }`}>
                          {medicine.available !== false ? (
                            <>
                              <CheckCircle size={16} style={{ marginRight: '0.25rem' }} />
                              Available
                            </>
                          ) : (
                            <>
                              <AlertCircle size={16} style={{ marginRight: '0.25rem' }} />
                              Check Availability
                            </>
                          )}
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3" style={{ gap: '1rem', fontSize: '0.875rem' }}>
                        <div>
                          <span style={{ color: '#6b7280' }}>Dosage:</span>
                          <p style={{ fontWeight: '500' }}>{medicine.dosage || "As prescribed"}</p>
                        </div>
                        <div>
                          <span style={{ color: '#6b7280' }}>Frequency:</span>
                          <p style={{ fontWeight: '500' }}>{medicine.frequency || "As directed"}</p>
                        </div>
                        <div>
                          <span style={{ color: '#6b7280' }}>Duration:</span>
                          <p style={{ fontWeight: '500' }}>{medicine.duration || "As prescribed"}</p>
                        </div>
                      </div>
                      
                      {(medicine.instructions || medicine.timing) && (
                        <div style={{ paddingTop: '0.75rem', borderTop: '1px solid #f3f4f6', marginTop: '0.75rem' }}>
                          <span style={{ color: '#6b7280' }}>Instructions:</span>
                          <p style={{ color: '#1f2937', marginTop: '0.25rem' }}>
                            {medicine.instructions || medicine.timing}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: '#6b7280', textAlign: 'center', padding: '2rem 0' }}>
                  No medicines detected in prescription.
                </p>
              )}
            </div>

            {/* New Analysis Button */}
            <div className="card" style={{ textAlign: 'center' }}>
              <button
                onClick={handleNewAnalysis}
                className="btn btn-primary"
              >
                <Upload size={20} style={{ marginRight: '0.5rem' }} />
                Analyze New Prescription
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedPrescriptionAnalyzer;