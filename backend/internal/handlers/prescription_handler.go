package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

// MLExtractionResult represents the response from ML service
type MLExtractionResult struct {
	Success         bool                     `json:"success"`
	PrescriptionID  string                   `json:"prescription_id"`
	Patient         map[string]interface{}   `json:"patient"`
	Doctor          map[string]interface{}   `json:"doctor"`
	Medicines       []map[string]interface{} `json:"medicines"`
	Diagnosis       []string                 `json:"diagnosis"`
	ConfidenceScore float64                  `json:"confidence_score"`
	RawText         string                   `json:"raw_text"`
	Error           string                   `json:"error,omitempty"`
	Message         string                   `json:"message,omitempty"`
}

// callMLService calls the ML service to analyze prescription
func (h *PrescriptionHandler) callMLService(imageBytes []byte) (*MLExtractionResult, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", "prescription.jpg")
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err = io.Copy(part, bytes.NewReader(imageBytes)); err != nil {
		return nil, fmt.Errorf("failed to copy image data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	req, err := http.NewRequest("POST", h.mlServiceURL+"/analyze-prescription", body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call ML service: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Log the response for debugging
	log.Printf("ML Service Response Status: %d", resp.StatusCode)
	log.Printf("ML Service Response Body: %s", string(bodyBytes))

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result MLExtractionResult
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to parse ML response: %w. Response: %s", err, string(bodyBytes))
	}

	// Validate the result
	if !result.Success {
		log.Printf("ML service analysis failed: %s", result.Error)
		return &result, nil
	}

	// Ensure confidence score is valid
	if result.ConfidenceScore < 0 || result.ConfidenceScore > 1 {
		log.Printf("Invalid confidence score: %f, setting to 0", result.ConfidenceScore)
		result.ConfidenceScore = 0
	}

	log.Printf("✅ ML Analysis successful - Confidence: %.2f%%", result.ConfidenceScore*100)
	log.Printf("   Patient: %v", result.Patient["name"])
	log.Printf("   Doctor: %v", result.Doctor["name"])
	log.Printf("   Medicines: %d", len(result.Medicines))

	return &result, nil
}

// Upload handles prescription upload
func (h *PrescriptionHandler) Upload(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}

	if !isValidImageType(file.Filename) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type"})
		return
	}

	fileContent, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read file"})
		return
	}
	defer fileContent.Close()

	fileBytes, err := io.ReadAll(fileContent)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read file content"})
		return
	}

	log.Printf("📄 Uploading file: %s, Size: %d bytes", file.Filename, len(fileBytes))

	imageURL, err := h.storageService.UploadImage(fileBytes, file.Filename)
	if err != nil {
		log.Printf("❌ Failed to upload image: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to upload image"})
		return
	}

	log.Printf("📤 Calling ML service...")
	extractionResult, err := h.callMLService(fileBytes)
	if err != nil {
		log.Printf("❌ ML service error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Failed to extract information: %v", err),
		})
		return
	}

	if !extractionResult.Success {
		log.Printf("⚠️ ML service returned unsuccessful result: %s", extractionResult.Error)
		// Return error but with 200 status so frontend can handle it
		c.JSON(http.StatusOK, gin.H{
			"success": false,
			"error":   extractionResult.Error,
			"message": extractionResult.Message,
		})
		return
	}

	prescription := Prescription{
		PrescriptionID: extractionResult.PrescriptionID,
		Patient:        convertPatientInfo(extractionResult.Patient),
		Doctor:         convertDoctorInfo(extractionResult.Doctor),
		Medicines:      convertMedicines(extractionResult.Medicines),
		Diagnosis:      extractionResult.Diagnosis,
		Confidence:     extractionResult.ConfidenceScore,
		ImageURL:       imageURL,
		RawText:        extractionResult.RawText,
		Status:         "pending_verification",
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}

	log.Printf("💾 Saving prescription to database...")
	collection := h.db.Collection("prescriptions")
	result, err := collection.InsertOne(c.Request.Context(), prescription)
	if err != nil {
		log.Printf("❌ Failed to save prescription: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save prescription"})
		return
	}

	prescription.ID = result.InsertedID.(primitive.ObjectID)

	log.Printf("✅ Prescription saved successfully: %s", prescription.PrescriptionID)
	log.Printf("   Confidence: %.2f%%", prescription.Confidence*100)
	log.Printf("   Patient: %s", prescription.Patient.Name)
	log.Printf("   Doctor: %s", prescription.Doctor.Name)
	log.Printf("   Medicines: %d", len(prescription.Medicines))

	go h.triggerLearningPipeline(prescription)

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescription,
	})
}

// Helper functions
func convertPatientInfo(data map[string]interface{}) PatientInfo {
	return PatientInfo{
		Name:   getStringFromMap(data, "name"),
		Age:    getStringFromMap(data, "age"),
		Gender: getStringFromMap(data, "gender"),
	}
}

func convertDoctorInfo(data map[string]interface{}) DoctorInfo {
	return DoctorInfo{
		Name:           getStringFromMap(data, "name"),
		Specialization: getStringFromMap(data, "specialization"),
		Registration:   getStringFromMap(data, "registration_number"),
	}
}

func convertMedicines(data []map[string]interface{}) []Medicine {
	medicines := make([]Medicine, 0, len(data))
	for _, m := range data {
		medicine := Medicine{
			Name:      getStringFromMap(m, "name"),
			Dosage:    getStringFromMap(m, "dosage"),
			Frequency: getStringFromMap(m, "frequency"),
			Timing:    getStringFromMap(m, "timing"),
			Duration:  getStringFromMap(m, "duration"),
			Quantity:  getIntFromMap(m, "quantity"),
		}
		medicines = append(medicines, medicine)
	}
	return medicines
}

func getStringFromMap(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok && v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getIntFromMap(m map[string]interface{}, key string) int {
	if v, ok := m[key]; ok && v != nil {
		switch val := v.(type) {
		case int:
			return val
		case float64:
			return int(val)
		case string:
			// Try to parse string to int
			var num int
			fmt.Sscanf(val, "%d", &num)
			return num
		}
	}
	return 1 // Default quantity
}

func isValidImageType(filename string) bool {
	validTypes := []string{".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
	for _, ext := range validTypes {
		if len(filename) >= len(ext) &&
			filename[len(filename)-len(ext):] == ext {
			return true
		}
	}
	return false
}

// Get retrieves a prescription by ID
func (h *PrescriptionHandler) Get(c *gin.Context) {
	id := c.Param("id")

	var prescription Prescription
	collection := h.db.Collection("prescriptions")

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid prescription ID"})
		return
	}

	err = collection.FindOne(c.Request.Context(), map[string]interface{}{"_id": objID}).Decode(&prescription)
	if err != nil {
		if err == mongo.ErrNoDocuments {
			c.JSON(http.StatusNotFound, gin.H{"error": "Prescription not found"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve prescription"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescription,
	})
}

// GetHistory retrieves prescription history
func (h *PrescriptionHandler) GetHistory(c *gin.Context) {
	collection := h.db.Collection("prescriptions")

	cursor, err := collection.Find(c.Request.Context(), map[string]interface{}{})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve history"})
		return
	}
	defer cursor.Close(c.Request.Context())

	var prescriptions []Prescription
	if err = cursor.All(c.Request.Context(), &prescriptions); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse history"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescriptions,
		"total":   len(prescriptions),
	})
}
