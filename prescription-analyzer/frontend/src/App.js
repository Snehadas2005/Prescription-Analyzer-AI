import React, { useState, useCallback } from "react";
import {
  Upload,
  FileText,
  User,
  Pill,
  ShoppingCart,
  CheckCircle,
  AlertCircle,
  Clock,
  Star,
  Activity,
  MapPin,
  Phone,
  Calendar,
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
  const [orderStatus, setOrderStatus] = useState(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [deliveryAddress, setDeliveryAddress] = useState("");
  const [contactNumber, setContactNumber] = useState("");
  const [selectedMedicines, setSelectedMedicines] = useState({});
  const [error, setError] = useState(null);
  const [isCreatingOrder, setIsCreatingOrder] = useState(false);

  // API base URL - Update this to your backend URL
  const API_BASE_URL = "http://localhost:8000";

  const handleFileSelect = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith("image/")) {
        setSelectedFile(file);
        setAnalysisResult(null);
        setOrderStatus(null);
        setError(null);
      } else {
        setError("Please select an image file (JPEG, PNG, etc.)");
      }
    }
  }, []);

  const analyzePrescrição = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setActiveTab("analysis");
    setError(null);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append("file", selectedFile);

      // Call the actual backend API
      const response = await fetch(`${API_BASE_URL}/api/analyze-prescription`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setAnalysisResult(result);
        setActiveTab("results");
        
        // Initialize medicine selection
        const initialSelection = {};
        if (result.medicines && result.medicines.length > 0) {
          result.medicines.forEach(med => {
            initialSelection[med.name] = med.available ? 1 : 0;
          });
          setSelectedMedicines(initialSelection);
        }
      } else {
        setError(result.error || "Failed to analyze prescription");
      }
    } catch (error) {
      console.error("Error analyzing prescription:", error);
      setError(`Failed to connect to server: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const createOrder = async () => {
    if (!analysisResult || !analysisResult.success) return;

    if (!deliveryAddress.trim() || !contactNumber.trim()) {
      setError("Please provide both delivery address and contact number");
      return;
    }

    setIsCreatingOrder(true);
    setError(null);

    try {
      const selectedMeds = analysisResult.medicines.filter(med => 
        selectedMedicines[med.name] > 0
      ).map(med => ({
        ...med,
        quantity: selectedMedicines[med.name]
      }));

      if (selectedMeds.length === 0) {
        setError("Please select at least one medicine to order");
        setIsCreatingOrder(false);
        return;
      }

      const orderData = {
        prescription_id: analysisResult.prescription_id,
        patient_info: analysisResult.patient,
        medicines: selectedMeds,
        delivery_address: deliveryAddress,
        contact_number: contactNumber,
      };

      const response = await fetch(`${API_BASE_URL}/api/create-order`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(orderData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const orderResult = await response.json();
      
      if (orderResult.success) {
        setOrderStatus(orderResult);
        setActiveTab("order");
      } else {
        setError(orderResult.error || "Failed to create order");
      }
    } catch (error) {
      console.error("Error creating order:", error);
      setError(`Failed to create order: ${error.message}`);
    } finally {
      setIsCreatingOrder(false);
    }
  };

  const handleMedicineQuantityChange = (medicineName, quantity) => {
    setSelectedMedicines(prev => ({
      ...prev,
      [medicineName]: parseInt(quantity) || 0
    }));
  };

  const ConfidenceBar = ({ score }) => (
    <div className="confidence-bar">
      <div
        className={`confidence-fill ${
          score >= 0.8 ? "bg-gradient-to-r from-green-500 to-green-600" :
          score >= 0.6 ? "bg-gradient-to-r from-yellow-500 to-yellow-600" :
          "bg-gradient-to-r from-red-500 to-red-600"
        }`}
        style={{ width: `${score * 100}%` }}
      />
      <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
        {(score * 100).toFixed(0)}%
      </div>
    </div>
  );

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
            Upload your prescription for instant analysis and medicine ordering
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
              isDisabled={!analysisResult?.success}
            >
              Results
            </TabButton>
            <TabButton 
              tab="order" 
              isActive={activeTab === "order"}
              isDisabled={!orderStatus}
            >
              Order
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
                    onClick={analyzePrescrição}
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
                    <div className="confidence-bar">
                      <div className="confidence-fill bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500" 
                           style={{width: "60%", animation: "pulse 2s infinite"}} />
                    </div>
                  </div>
                ) : analysisResult?.success ? (
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
        {activeTab === "results" && analysisResult?.success && (
          <div className="grid" style={{ gap: '1.5rem' }}>
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
                  {Math.round((analysisResult.confidence_score || 0) * 100)}%
                </span>
              </div>
              <ConfidenceBar score={analysisResult.confidence_score || 0} />
              <p style={{ color: '#6b7280', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Higher confidence indicates better text extraction and medicine identification
              </p>
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
                      {analysisResult.patient?.name || analysisResult.patient_name || "Not specified"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>Age:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.patient?.age || analysisResult.patient_age || "Not specified"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>Gender:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.patient?.gender || analysisResult.patient_gender || "Not specified"}
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
                      {analysisResult.doctor?.name || analysisResult.doctor_name || "Not specified"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#6b7280' }}>License:</span>
                    <span style={{ fontWeight: '500' }}>
                      {analysisResult.doctor?.registration_number || analysisResult.doctor_license || "Not specified"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Diagnosis */}
            {analysisResult.diagnosis && analysisResult.diagnosis.length > 0 && (
              <div className="card">
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
                  <Heart size={24} color="#ef4444" style={{ marginRight: '0.5rem' }} />
                  <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1f2937' }}>
                    Diagnosed Conditions
                  </h3>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {analysisResult.diagnosis.map((condition, index) => (
                    <span
                      key={index}
                      className="status-badge"
                      style={{ 
                        background: 'linear-gradient(135deg, #fee2e2, #fecaca)',
                        color: '#991b1b'
                      }}
                    >
                      {condition}
                    </span>
                  ))}
                </div>
              </div>
            )}

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
                          {medicine.name}
                        </h4>
                        <div className={`status-badge ${
                          medicine.available ? 'status-available' : 'status-unavailable'
                        }`}>
                          {medicine.available ? (
                            <>
                              <CheckCircle size={16} style={{ marginRight: '0.25rem' }} />
                              Available
                            </>
                          ) : (
                            <>
                              <AlertCircle size={16} style={{ marginRight: '0.25rem' }} />
                              Out of Stock
                            </>
                          )}
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-4" style={{ gap: '1rem', fontSize: '0.875rem', marginBottom: '1rem' }}>
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
                        <div>
                          <span style={{ color: '#6b7280' }}>Quantity for Order:</span>
                          <select
                            value={selectedMedicines[medicine.name] || 0}
                            onChange={(e) => handleMedicineQuantityChange(medicine.name, e.target.value)}
                            disabled={!medicine.available}
                            className="form-select"
                            style={{ marginTop: '0.25rem', padding: '0.5rem', fontSize: '0.875rem' }}
                          >
                            {[0, 1, 2, 3, 4, 5].map(num => (
                              <option key={num} value={num}>{num}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                      
                      {medicine.instructions && (
                        <div style={{ paddingTop: '0.75rem', borderTop: '1px solid #f3f4f6' }}>
                          <span style={{ color: '#6b7280' }}>Instructions:</span>
                          <p style={{ color: '#1f2937', marginTop: '0.25rem' }}>{medicine.instructions}</p>
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

            {/* Order Section */}
            <div className="card">
              <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1f2937', marginBottom: '1.5rem' }}>
                Delivery Details
              </h3>
              <div className="grid grid-cols-2" style={{ gap: '1rem', marginBottom: '1.5rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '0.5rem' }}>
                    <MapPin size={16} style={{ display: 'inline', marginRight: '0.25rem' }} />
                    Delivery Address
                  </label>
                  <textarea
                    value={deliveryAddress}
                    onChange={(e) => setDeliveryAddress(e.target.value)}
                    placeholder="Enter your complete delivery address..."
                    className="form-textarea"
                    rows="3"
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '0.5rem' }}>
                    <Phone size={16} style={{ display: 'inline', marginRight: '0.25rem' }} />
                    Contact Number
                  </label>
                  <input
                    type="text"
                    value={contactNumber}
                    onChange={(e) => setContactNumber(e.target.value)}
                    placeholder="Enter your contact number..."
                    className="form-input"
                  />
                </div>
              </div>
              
              <div style={{ textAlign: 'center' }}>
                <button
                  onClick={createOrder}
                  disabled={!deliveryAddress || !contactNumber || isCreatingOrder}
                  className={`btn ${
                    !deliveryAddress || !contactNumber || isCreatingOrder ? '' : 'btn-success'
                  }`}
                >
                  {isCreatingOrder ? (
                    <>
                      <Loader2 size={20} className="loading-spinner" style={{ marginRight: '0.5rem' }} />
                      Creating Order...
                    </>
                  ) : (
                    <>
                      <ShoppingCart size={20} style={{ marginRight: '0.5rem' }} />
                      Place Order
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Order Status Section */}
        {activeTab === "order" && orderStatus && (
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <div className="card">
              {orderStatus.success ? (
                <div style={{ textAlign: 'center' }}>
                  <CheckCircle size={64} color="#10b981" style={{ margin: '0 auto 1rem' }} />
                  <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '1rem' }}>
                    Order Confirmed!
                  </h2>
                  
                  <div style={{ 
                    background: 'linear-gradient(135deg, #dcfce7, #bbf7d0)', 
                    borderRadius: '1rem', 
                    padding: '1.5rem', 
                    marginBottom: '1.5rem' 
                  }}>
                    <div className="grid grid-cols-2" style={{ gap: '1rem', textAlign: 'left' }}>
                      <div>
                        <span style={{ color: '#6b7280' }}>Order ID:</span>
                        <p style={{ fontWeight: '700', color: '#166534' }}>{orderStatus.order_id}</p>
                      </div>
                      <div>
                        <span style={{ color: '#6b7280' }}>Estimated Delivery:</span>
                        <p style={{ fontWeight: '700', color: '#166534' }}>{orderStatus.estimated_delivery}</p>
                      </div>
                      <div>
                        <span style={{ color: '#6b7280' }}>Total Amount:</span>
                        <p style={{ fontWeight: '700', color: '#166534' }}>₹{orderStatus.total_amount}</p>
                      </div>
                      <div>
                        <span style={{ color: '#6b7280' }}>Status:</span>
                        <p style={{ fontWeight: '700', color: '#166534' }}>Confirmed</p>
                      </div>
                    </div>
                  </div>

                  <div style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
                    <p style={{ color: '#374151', marginBottom: '0.5rem' }}>
                      <strong>Delivery Address:</strong> {deliveryAddress}
                    </p>
                    <p style={{ color: '#374151' }}>
                      <strong>Contact:</strong> {contactNumber}
                    </p>
                  </div>

                  <p style={{ color: '#6b7280', marginBottom: '1.5rem' }}>
                    You will receive SMS updates about your order status
                  </p>

                  <button
                    onClick={() => {
                      setActiveTab("upload");
                      setSelectedFile(null);
                      setAnalysisResult(null);
                      setOrderStatus(null);
                      setDeliveryAddress("");
                      setContactNumber("");
                      setSelectedMedicines({});
                      setError(null);
                    }}
                    className="btn btn-primary"
                  >
                    Analyze New Prescription
                  </button>
                </div>
              ) : (
                <div style={{ textAlign: 'center' }}>
                  <AlertCircle size={64} color="#ef4444" style={{ margin: '0 auto 1rem' }} />
                  <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '0.5rem' }}>
                    Order Failed
                  </h2>
                  <p style={{ color: '#ef4444' }}>
                    {orderStatus.error || "Unable to place order. Please try again."}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedPrescriptionAnalyzer;