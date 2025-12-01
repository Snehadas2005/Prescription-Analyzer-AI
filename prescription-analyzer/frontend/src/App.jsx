import React, { useState } from 'react';
import UploadSection from './components/UploadSection';
import ResultsDisplay from './components/ResultsDisplay';
import FeedbackModal from './components/FeedbackModal';
import PrescriptionHistory from './components/PrescriptionHistory';
import { uploadPrescription, submitFeedback } from './services/api';
import { FileText, Upload, History, Settings } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('upload'); // upload, results, history
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await uploadPrescription(file);
      setResult(data);
      setActiveTab('results');
      
      // Show feedback modal after a short delay
      setTimeout(() => {
        setShowFeedback(true);
      }, 1000);
      
    } catch (err) {
      setError(err.message || 'Upload failed. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (feedback) => {
    try {
      await submitFeedback(result.prescription_id, feedback);
      setShowFeedback(false);
      
      // Show success message
      alert('Thank you for your feedback! This helps improve our AI model.');
    } catch (err) {
      console.error('Feedback error:', err);
      alert('Failed to submit feedback. Please try again.');
    }
  };

  const handleNewAnalysis = () => {
    setResult(null);
    setActiveTab('upload');
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center">
                <FileText className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">
                  AI Prescription Analyzer
                </h1>
                <p className="text-sm text-gray-600">
                  Powered by Machine Learning • Continuous Learning Enabled
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="p-2 rounded-xl hover:bg-gray-100 transition-colors">
                <Settings className="text-gray-600" size={24} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-center space-x-4">
          <button
            onClick={() => setActiveTab('upload')}
            className={`
              flex items-center space-x-2 px-6 py-3 rounded-2xl font-semibold
              transition-all duration-300
              ${activeTab === 'upload'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg'
                : 'bg-white text-gray-600 hover:bg-gray-50'
              }
            `}
          >
            <Upload size={20} />
            <span>New Analysis</span>
          </button>
          
          <button
            onClick={() => setActiveTab('history')}
            className={`
              flex items-center space-x-2 px-6 py-3 rounded-2xl font-semibold
              transition-all duration-300
              ${activeTab === 'history'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg'
                : 'bg-white text-gray-600 hover:bg-gray-50'
              }
            `}
          >
            <History size={20} />
            <span>History</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Error Display */}
        {error && (
          <div className="max-w-4xl mx-auto mb-6">
            <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
                <button
                  onClick={() => setError(null)}
                  className="ml-auto flex-shrink-0 text-red-500 hover:text-red-700"
                >
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Content based on active tab */}
        {activeTab === 'upload' && (
          <>
            {!result ? (
              <UploadSection onUpload={handleUpload} loading={loading} />
            ) : (
              <div className="space-y-6">
                <ResultsDisplay result={result} />
                <div className="flex justify-center">
                  <button
                    onClick={handleNewAnalysis}
                    className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg hover:shadow-xl"
                  >
                    Analyze New Prescription
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === 'results' && result && (
          <div className="space-y-6">
            <ResultsDisplay result={result} />
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setShowFeedback(true)}
                className="px-8 py-4 bg-white text-blue-600 border-2 border-blue-600 rounded-2xl font-semibold hover:bg-blue-50 transition-all"
              >
                Provide Feedback
              </button>
              <button
                onClick={handleNewAnalysis}
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg"
              >
                New Analysis
              </button>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <PrescriptionHistory />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white mt-16 py-8">
        <div className="container mx-auto px-4">
          <div className="text-center text-gray-600">
            <p className="mb-2">
              <strong>Medical Disclaimer:</strong> This is an AI-powered tool for educational purposes.
            </p>
            <p className="text-sm">
              Always consult with qualified healthcare professionals for medical advice.
              Do not rely solely on AI analysis for medical decisions.
            </p>
            <div className="mt-4 text-xs text-gray-500">
              <p>Built with ❤️ using React, Go, Python, and MongoDB</p>
              <p className="mt-1">© 2024 AI Prescription Analyzer. All rights reserved.</p>
            </div>
          </div>
        </div>
      </footer>

      {/* Feedback Modal */}
      {showFeedback && result && (
        <FeedbackModal
          result={result}
          onSubmit={handleFeedback}
          onClose={() => setShowFeedback(false)}
        />
      )}
    </div>
  );
}

export default App;