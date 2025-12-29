const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080/api/v1';

/**
 * Upload prescription image for analysis
 * @param {File} file - The prescription image file
 * @returns {Promise<Object>} Analysis result
 */
export const uploadPrescription = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Upload failed');
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Analysis failed');
    }

    return data.data;
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
};

/**
 * Submit user feedback for a prescription
 * @param {string} prescriptionId - The prescription ID
 * @param {Object} feedback - Feedback data
 * @returns {Promise<Object>} Submission result
 */
export const submitFeedback = async (prescriptionId, feedback) => {
  try {
    const response = await fetch(`${API_BASE_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prescription_id: prescriptionId,
        ...feedback,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Feedback submission failed');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Feedback error:', error);
    throw error;
  }
};

/**
 * Get prescription by ID
 * @param {string} prescriptionId - The prescription ID
 * @returns {Promise<Object>} Prescription data
 */
export const getPrescription = async (prescriptionId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/prescription/${prescriptionId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to fetch prescription');
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Prescription not found');
    }

    return data.data;
  } catch (error) {
    console.error('Get prescription error:', error);
    throw error;
  }
};

/**
 * Get prescription history
 * @param {number} page - Page number
 * @param {number} limit - Items per page
 * @returns {Promise<Object>} History data
 */
export const getPrescriptionHistory = async (page = 1, limit = 10) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/history?page=${page}&limit=${limit}`
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to fetch history');
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Failed to load history');
    }

    return {
      prescriptions: data.data,
      total: data.total,
      page: data.page,
      limit: data.limit,
    };
  } catch (error) {
    console.error('Get history error:', error);
    throw error;
  }
};

/**
 * Get feedback statistics
 * @returns {Promise<Object>} Feedback stats
 */
export const getFeedbackStats = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/feedback/stats`);

    if (!response.ok) {
      throw new Error('Failed to fetch statistics');
    }

    const data = await response.json();
    return data.data;
  } catch (error) {
    console.error('Get stats error:', error);
    throw error;
  }
};

/**
 * Trigger manual model training (admin only)
 * @returns {Promise<Object>} Training result
 */
export const triggerTraining = async () => {
  try {
    const ML_SERVICE_URL = process.env.REACT_APP_ML_SERVICE_URL || 'http://localhost:8000';
    
    const response = await fetch(`${ML_SERVICE_URL}/train`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error('Training trigger failed');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Training trigger error:', error);
    throw error;
  }
};

/**
 * Get ML service health status
 * @returns {Promise<Object>} Health status
 */
export const getMLServiceHealth = async () => {
  try {
    const ML_SERVICE_URL = process.env.REACT_APP_ML_SERVICE_URL || 'http://localhost:8000';
    
    const response = await fetch(`${ML_SERVICE_URL}/health`);

    if (!response.ok) {
      throw new Error('Health check failed');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
};

export default {
  uploadPrescription,
  submitFeedback,
  getPrescription,
  getPrescriptionHistory,
  getFeedbackStats,
  triggerTraining,
  getMLServiceHealth,
};