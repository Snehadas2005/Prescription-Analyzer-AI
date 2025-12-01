import React from 'react';
import {
  User,
  Stethoscope,
  Pill,
  Clock,
  Calendar,
  TrendingUp,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';

const ResultsDisplay = ({ result }) => {
  const { patient, doctor, medicines, confidence, prescription_id } = result;

  // Confidence color coding
  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getConfidenceLabel = (score) => {
    if (score >= 0.8) return 'High Confidence';
    if (score >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header with Confidence */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-3xl shadow-2xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold mb-2">Analysis Complete</h2>
            <p className="text-blue-100">
              Prescription ID: <span className="font-mono">{prescription_id}</span>
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp size={24} />
              <span className="text-2xl font-bold">
                {Math.round(confidence * 100)}%
              </span>
            </div>
            <span className={`
              px-4 py-2 rounded-full text-sm font-semibold
              ${getConfidenceColor(confidence)}
            `}>
              {getConfidenceLabel(confidence)}
            </span>
          </div>
        </div>
      </div>

      {/* Patient and Doctor Info */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Patient Card */}
        <div className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
              <User className="text-blue-600" size={24} />
            </div>
            <h3 className="text-xl font-bold text-gray-800">Patient Information</h3>
          </div>
          
          <div className="space-y-3">
            <InfoRow label="Name" value={patient.name || 'Not detected'} />
            <InfoRow label="Age" value={patient.age ? `${patient.age} years` : 'Not detected'} />
            <InfoRow label="Gender" value={patient.gender || 'Not detected'} />
          </div>
        </div>

        {/* Doctor Card */}
        <div className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
              <Stethoscope className="text-green-600" size={24} />
            </div>
            <h3 className="text-xl font-bold text-gray-800">Doctor Information</h3>
          </div>
          
          <div className="space-y-3">
            <InfoRow label="Name" value={doctor.name || 'Not detected'} />
            <InfoRow label="Specialization" value={doctor.specialization || 'Not specified'} />
            <InfoRow label="Registration" value={doctor.registration || 'Not detected'} />
          </div>
        </div>
      </div>

      {/* Medicines Section */}
      <div className="bg-white rounded-2xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
              <Pill className="text-purple-600" size={24} />
            </div>
            <h3 className="text-xl font-bold text-gray-800">Prescribed Medicines</h3>
          </div>
          <span className="px-4 py-2 bg-purple-100 text-purple-700 rounded-full font-semibold">
            {medicines?.length || 0} items
          </span>
        </div>

        {medicines && medicines.length > 0 ? (
          <div className="space-y-4">
            {medicines.map((medicine, index) => (
              <MedicineCard key={index} medicine={medicine} index={index} />
            ))}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <AlertCircle className="mx-auto mb-4 text-gray-400" size={48} />
            <p>No medicines detected in the prescription</p>
          </div>
        )}
      </div>

      {/* Important Notice */}
      <div className="bg-yellow-50 border-l-4 border-yellow-400 rounded-lg p-6">
        <div className="flex items-start space-x-3">
          <Info className="text-yellow-600 flex-shrink-0 mt-1" size={20} />
          <div>
            <h4 className="font-semibold text-yellow-800 mb-2">
              Important Notice
            </h4>
            <p className="text-sm text-yellow-700">
              This is an AI-generated analysis. Please verify all information with a healthcare 
              professional before taking any medication. Do not rely solely on this analysis for 
              medical decisions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper Components
const InfoRow = ({ label, value }) => (
  <div className="flex justify-between items-center py-2 border-b border-gray-100">
    <span className="text-gray-600 font-medium">{label}</span>
    <span className="text-gray-900 font-semibold">{value}</span>
  </div>
);

const MedicineCard = ({ medicine, index }) => (
  <div className="border border-gray-200 rounded-xl p-5 hover:border-purple-300 hover:shadow-md transition-all">
    <div className="flex items-start justify-between mb-4">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center font-bold text-purple-600">
          {index + 1}
        </div>
        <div>
          <h4 className="text-lg font-bold text-gray-800">{medicine.name}</h4>
          {medicine.dosage && (
            <p className="text-sm text-gray-600">{medicine.dosage}</p>
          )}
        </div>
      </div>
      <CheckCircle className="text-green-500" size={20} />
    </div>

    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <DetailBadge
        icon={Clock}
        label="Frequency"
        value={medicine.frequency || 'As directed'}
      />
      <DetailBadge
        icon={Calendar}
        label="Duration"
        value={medicine.duration || 'As prescribed'}
      />
      <DetailBadge
        icon={Info}
        label="Timing"
        value={medicine.timing || 'Anytime'}
      />
      <DetailBadge
        icon={Pill}
        label="Quantity"
        value={`${medicine.quantity || 1} package(s)`}
      />
    </div>
  </div>
);

const DetailBadge = ({ icon: Icon, label, value }) => (
  <div className="bg-gray-50 rounded-lg p-3">
    <div className="flex items-center space-x-1 mb-1">
      <Icon size={14} className="text-gray-500" />
      <span className="text-xs text-gray-500 font-medium">{label}</span>
    </div>
    <p className="text-sm font-semibold text-gray-800">{value}</p>
  </div>
);

export default ResultsDisplay;