import React, { useState, useCallback } from 'react';
import { Upload, Camera, FileText, Loader2, CheckCircle2 } from 'lucide-react';

const UploadSection = ({ onUpload, loading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    setSelectedFile(file);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-3xl shadow-2xl p-8 md:p-12">
        {/* Upload Area */}
        <div
          className={`
            relative border-3 border-dashed rounded-2xl p-12 text-center
            transition-all duration-300 cursor-pointer
            ${dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : selectedFile 
                ? 'border-green-500 bg-green-50' 
                : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50'
            }
          `}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-upload').click()}
        >
          <input
            id="file-upload"
            type="file"
            className="hidden"
            accept="image/*"
            onChange={handleChange}
            disabled={loading}
          />

          {preview ? (
            <div className="space-y-4">
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg shadow-lg"
              />
              <div className="flex items-center justify-center space-x-2 text-green-600">
                <CheckCircle2 size={24} />
                <span className="font-semibold">{selectedFile.name}</span>
              </div>
              <p className="text-sm text-gray-500">
                Click to change or drag another file
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="mx-auto w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center">
                <Upload className="text-white" size={32} />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                  Upload Prescription
                </h3>
                <p className="text-gray-600">
                  Drag and drop your prescription image here, or click to browse
                </p>
              </div>
              <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
                <div className="flex items-center space-x-1">
                  <Camera size={16} />
                  <span>Photos</span>
                </div>
                <div className="flex items-center space-x-1">
                  <FileText size={16} />
                  <span>Scans</span>
                </div>
              </div>
              <p className="text-xs text-gray-400">
                Supported: JPG, PNG, TIFF • Max size: 10MB
              </p>
            </div>
          )}
        </div>

        {/* Upload Button */}
        {selectedFile && (
          <div className="mt-8 flex justify-center">
            <button
              onClick={handleUpload}
              disabled={loading}
              className={`
                px-8 py-4 rounded-xl font-semibold text-white
                transition-all duration-300 transform
                ${loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:scale-105 hover:shadow-xl'
                }
              `}
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <Loader2 className="animate-spin" size={20} />
                  <span>Analyzing...</span>
                </div>
              ) : (
                <span>Analyze Prescription</span>
              )}
            </button>
          </div>
        )}

        {/* Tips Section */}
        <div className="mt-8 pt-8 border-t border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-4">
            📝 Tips for better results:
          </h4>
          <ul className="grid md:grid-cols-2 gap-3 text-sm text-gray-600">
            <li className="flex items-start space-x-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>Ensure good lighting and clear image</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>Place prescription on flat surface</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>Avoid shadows and glare</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>Include all pages if multiple</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default UploadSection;