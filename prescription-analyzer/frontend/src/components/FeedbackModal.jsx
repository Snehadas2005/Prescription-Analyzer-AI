import React, { useState } from 'react';
import {
  X,
  ThumbsUp,
  ThumbsDown,
  Edit,
  Star,
  Send,
  Check
} from 'lucide-react';

const FeedbackModal = ({ result, onSubmit, onClose }) => {
  const [feedbackType, setFeedbackType] = useState(null); // 'correct', 'incorrect', 'rating'
  const [rating, setRating] = useState(0);
  const [corrections, setCorrections] = useState({
    patient: { ...result.patient },
    doctor: { ...result.doctor },
    medicines: [...(result.medicines || [])]
  });
  const [comments, setComments] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleCorrection = (field, subfield, value) => {
    setCorrections(prev => ({
      ...prev,
      [field]: {
        ...prev[field],
        [subfield]: value
      }
    }));
  };

  const handleMedicineCorrection = (index, field, value) => {
    setCorrections(prev => ({
      ...prev,
      medicines: prev.medicines.map((med, i) =>
        i === index ? { ...med, [field]: value } : med
      )
    }));
  };

  const addMedicine = () => {
    setCorrections(prev => ({
      ...prev,
      medicines: [
        ...prev.medicines,
        {
          name: '',
          dosage: '',
          frequency: '',
          timing: '',
          duration: '',
          quantity: 1
        }
      ]
    }));
  };

  const removeMedicine = (index) => {
    setCorrections(prev => ({
      ...prev,
      medicines: prev.medicines.filter((_, i) => i !== index)
    }));
  };

  const handleSubmit = () => {
    const feedback = {
      prescription_id: result.prescription_id,
      feedback_type: feedbackType === 'correct' ? 'confirmation' : 
                      feedbackType === 'incorrect' ? 'correction' : 'rating',
      rating: feedbackType === 'rating' ? rating : null,
      corrections: feedbackType === 'incorrect' ? corrections : null,
      comments: comments
    };

    onSubmit(feedback);
    setSubmitted(true);
    setTimeout(() => onClose(), 2000);
  };

  if (submitted) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-3xl shadow-2xl p-12 max-w-md text-center">
          <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Check className="text-green-600" size={40} />
          </div>
          <h3 className="text-2xl font-bold text-gray-800 mb-2">
            Thank You!
          </h3>
          <p className="text-gray-600">
            Your feedback helps improve our AI model
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
      <div className="bg-white rounded-3xl shadow-2xl max-w-4xl w-full my-8">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-t-3xl p-6 text-white flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-1">Provide Feedback</h2>
            <p className="text-blue-100">Help us improve our AI accuracy</p>
          </div>
          <button
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-white bg-opacity-20 hover:bg-opacity-30 flex items-center justify-center transition-all"
          >
            <X size={24} />
          </button>
        </div>

        <div className="p-8">
          {/* Feedback Type Selection */}
          {!feedbackType && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                How accurate was the analysis?
              </h3>
              
              <button
                onClick={() => setFeedbackType('correct')}
                className="w-full p-6 border-2 border-gray-200 rounded-2xl hover:border-green-500 hover:bg-green-50 transition-all text-left"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                    <ThumbsUp className="text-green-600" size={24} />
                  </div>
                  <div>
                    <h4 className="font-bold text-gray-800">Accurate</h4>
                    <p className="text-sm text-gray-600">
                      All information was extracted correctly
                    </p>
                  </div>
                </div>
              </button>

              <button
                onClick={() => setFeedbackType('incorrect')}
                className="w-full p-6 border-2 border-gray-200 rounded-2xl hover:border-red-500 hover:bg-red-50 transition-all text-left"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                    <Edit className="text-red-600" size={24} />
                  </div>
                  <div>
                    <h4 className="font-bold text-gray-800">Needs Correction</h4>
                    <p className="text-sm text-gray-600">
                      Some information was incorrect or missing
                    </p>
                  </div>
                </div>
              </button>

              <button
                onClick={() => setFeedbackType('rating')}
                className="w-full p-6 border-2 border-gray-200 rounded-2xl hover:border-yellow-500 hover:bg-yellow-50 transition-all text-left"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center">
                    <Star className="text-yellow-600" size={24} />
                  </div>
                  <div>
                    <h4 className="font-bold text-gray-800">Rate Experience</h4>
                    <p className="text-sm text-gray-600">
                      Provide a rating for the overall experience
                    </p>
                  </div>
                </div>
              </button>
            </div>
          )}

          {/* Confirmation Feedback */}
          {feedbackType === 'correct' && (
            <div className="space-y-6">
              <div className="bg-green-50 border border-green-200 rounded-2xl p-6 text-center">
                <ThumbsUp className="text-green-600 mx-auto mb-3" size={48} />
                <h3 className="text-xl font-bold text-gray-800 mb-2">
                  Great! All information correct
                </h3>
                <p className="text-gray-600">
                  Your confirmation helps us improve our AI model
                </p>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Additional Comments (Optional)
                </label>
                <textarea
                  value={comments}
                  onChange={(e) => setComments(e.target.value)}
                  placeholder="Any additional feedback..."
                  className="w-full border border-gray-300 rounded-xl p-4 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows="3"
                />
              </div>

              <div className="flex space-x-4">
                <button
                  onClick={() => setFeedbackType(null)}
                  className="flex-1 px-6 py-3 border-2 border-gray-300 rounded-xl font-semibold text-gray-700 hover:bg-gray-50 transition-all"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl font-semibold text-white hover:from-blue-700 hover:to-indigo-700 transition-all flex items-center justify-center space-x-2"
                >
                  <Send size={20} />
                  <span>Submit Feedback</span>
                </button>
              </div>
            </div>
          )}

          {/* Correction Feedback */}
          {feedbackType === 'incorrect' && (
            <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-2">
              <h3 className="text-lg font-semibold text-gray-800 sticky top-0 bg-white pb-2">
                Correct the Information
              </h3>

              {/* Patient Corrections */}
              <div className="bg-blue-50 rounded-2xl p-6">
                <h4 className="font-bold text-gray-800 mb-4">Patient Information</h4>
                <div className="space-y-3">
                  <input
                    type="text"
                    value={corrections.patient.name}
                    onChange={(e) => handleCorrection('patient', 'name', e.target.value)}
                    placeholder="Patient Name"
                    className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500"
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      type="text"
                      value={corrections.patient.age}
                      onChange={(e) => handleCorrection('patient', 'age', e.target.value)}
                      placeholder="Age"
                      className="border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500"
                    />
                    <select
                      value={corrections.patient.gender}
                      onChange={(e) => handleCorrection('patient', 'gender', e.target.value)}
                      className="border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Gender</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Doctor Corrections */}
              <div className="bg-green-50 rounded-2xl p-6">
                <h4 className="font-bold text-gray-800 mb-4">Doctor Information</h4>
                <div className="space-y-3">
                  <input
                    type="text"
                    value={corrections.doctor.name}
                    onChange={(e) => handleCorrection('doctor', 'name', e.target.value)}
                    placeholder="Doctor Name"
                    className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-green-500"
                  />
                  <input
                    type="text"
                    value={corrections.doctor.specialization}
                    onChange={(e) => handleCorrection('doctor', 'specialization', e.target.value)}
                    placeholder="Specialization"
                    className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-green-500"
                  />
                </div>
              </div>

              {/* Medicine Corrections */}
              <div className="bg-purple-50 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-bold text-gray-800">Medicines</h4>
                  <button
                    onClick={addMedicine}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-all text-sm font-semibold"
                  >
                    + Add Medicine
                  </button>
                </div>
                
                <div className="space-y-4">
                  {corrections.medicines.map((medicine, index) => (
                    <div key={index} className="bg-white rounded-xl p-4 border border-purple-200">
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-semibold text-gray-700">Medicine {index + 1}</span>
                        <button
                          onClick={() => removeMedicine(index)}
                          className="text-red-600 hover:text-red-700 text-sm font-semibold"
                        >
                          Remove
                        </button>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={medicine.name}
                          onChange={(e) => handleMedicineCorrection(index, 'name', e.target.value)}
                          placeholder="Medicine Name"
                          className="border border-gray-300 rounded-lg p-2 text-sm"
                        />
                        <input
                          type="text"
                          value={medicine.dosage}
                          onChange={(e) => handleMedicineCorrection(index, 'dosage', e.target.value)}
                          placeholder="Dosage"
                          className="border border-gray-300 rounded-lg p-2 text-sm"
                        />
                        <input
                          type="text"
                          value={medicine.frequency}
                          onChange={(e) => handleMedicineCorrection(index, 'frequency', e.target.value)}
                          placeholder="Frequency"
                          className="border border-gray-300 rounded-lg p-2 text-sm"
                        />
                        <input
                          type="text"
                          value={medicine.duration}
                          onChange={(e) => handleMedicineCorrection(index, 'duration', e.target.value)}
                          placeholder="Duration"
                          className="border border-gray-300 rounded-lg p-2 text-sm"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Additional Notes
                </label>
                <textarea
                  value={comments}
                  onChange={(e) => setComments(e.target.value)}
                  placeholder="Describe what was incorrect..."
                  className="w-full border border-gray-300 rounded-xl p-4 focus:ring-2 focus:ring-blue-500"
                  rows="3"
                />
              </div>

              <div className="flex space-x-4 sticky bottom-0 bg-white pt-4">
                <button
                  onClick={() => setFeedbackType(null)}
                  className="flex-1 px-6 py-3 border-2 border-gray-300 rounded-xl font-semibold text-gray-700 hover:bg-gray-50"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl font-semibold text-white hover:from-blue-700 hover:to-indigo-700 flex items-center justify-center space-x-2"
                >
                  <Send size={20} />
                  <span>Submit Corrections</span>
                </button>
              </div>
            </div>
          )}

          {/* Rating Feedback */}
          {feedbackType === 'rating' && (
            <div className="space-y-6">
              <div className="text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-4">
                  Rate Your Experience
                </h3>
                <div className="flex justify-center space-x-2 mb-6">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      onClick={() => setRating(star)}
                      className="transition-transform hover:scale-110"
                    >
                      <Star
                        size={48}
                        className={star <= rating ? 'fill-yellow-400 text-yellow-400' : 'text-gray-300'}
                      />
                    </button>
                  ))}
                </div>
                <p className="text-gray-600">
                  {rating === 0 && 'Click to rate'}
                  {rating === 1 && 'Poor'}
                  {rating === 2 && 'Fair'}
                  {rating === 3 && 'Good'}
                  {rating === 4 && 'Very Good'}
                  {rating === 5 && 'Excellent'}
                </p>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Comments (Optional)
                </label>
                <textarea
                  value={comments}
                  onChange={(e) => setComments(e.target.value)}
                  placeholder="Tell us about your experience..."
                  className="w-full border border-gray-300 rounded-xl p-4 focus:ring-2 focus:ring-blue-500"
                  rows="4"
                />
              </div>

              <div className="flex space-x-4">
                <button
                  onClick={() => setFeedbackType(null)}
                  className="flex-1 px-6 py-3 border-2 border-gray-300 rounded-xl font-semibold text-gray-700 hover:bg-gray-50"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={rating === 0}
                  className={`
                    flex-1 px-6 py-3 rounded-xl font-semibold text-white
                    flex items-center justify-center space-x-2
                    ${rating === 0
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700'
                    }
                  `}
                >
                  <Send size={20} />
                  <span>Submit Rating</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FeedbackModal;