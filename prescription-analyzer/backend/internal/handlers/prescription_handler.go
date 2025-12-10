package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
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

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// First, unmarshal into a generic map to handle flexible structure
	var rawResult map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &rawResult); err != nil {
		return nil, fmt.Errorf("failed to parse ML response: %w. Response: %s", err, string(bodyBytes))
	}

	// Create MLExtractionResult with safe type conversions
	result := &MLExtractionResult{
		Success:         true,
		PrescriptionID:  getStringValue(rawResult, "prescription_id"),
		Patient:         getMapValue(rawResult, "patient"),
		Doctor:          getMapValue(rawResult, "doctor"),
		Medicines:       getMedicinesArray(rawResult),
		ConfidenceScore: getFloatValue(rawResult, "confidence_score"),
		RawText:         getStringValue(rawResult, "raw_text"),
		Message:         getStringValue(rawResult, "message"),
	}

	return result, nil
}

// Helper functions for safe type conversions
func getStringValue(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok && v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getFloatValue(m map[string]interface{}, key string) float64 {
	if v, ok := m[key]; ok && v != nil {
		switch val := v.(type) {
		case float64:
			return val
		case float32:
			return float64(val)
		case int:
			return float64(val)
		}
	}
	return 0.0
}

func getMapValue(m map[string]interface{}, key string) map[string]interface{} {
	if v, ok := m[key]; ok && v != nil {
		if mapVal, ok := v.(map[string]interface{}); ok {
			return mapVal
		}
	}
	return make(map[string]interface{})
}

func getMedicinesArray(m map[string]interface{}) []map[string]interface{} {
	if v, ok := m["medicines"]; ok && v != nil {
		if arr, ok := v.([]interface{}); ok {
			result := make([]map[string]interface{}, 0, len(arr))
			for _, item := range arr {
				if medMap, ok := item.(map[string]interface{}); ok {
					result = append(result, medMap)
				}
			}
			return result
		}
	}
	return []map[string]interface{}{}
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

	imageURL, err := h.storageService.UploadImage(fileBytes, file.Filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to upload image"})
		return
	}

	extractionResult, err := h.callMLService(fileBytes)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Failed to extract information: %v", err),
		})
		return
	}

	prescription := Prescription{
		PrescriptionID: extractionResult.PrescriptionID,
		Patient:        convertPatientInfo(extractionResult.Patient),
		Doctor:         convertDoctorInfo(extractionResult.Doctor),
		Medicines:      convertMedicines(extractionResult.Medicines),
		Confidence:     extractionResult.ConfidenceScore,
		ImageURL:       imageURL,
		RawText:        extractionResult.RawText,
		Status:         "pending_verification",
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}

	collection := h.db.Collection("prescriptions")
	result, err := collection.InsertOne(c.Request.Context(), prescription)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save prescription"})
		return
	}

	prescription.ID = result.InsertedID.(primitive.ObjectID)

	go h.triggerLearningPipeline(prescription)

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescription,
	})
}

// Helper functions
func convertPatientInfo(data map[string]interface{}) PatientInfo {
	return PatientInfo{
		Name:   getString(data, "name"),
		Age:    getString(data, "age"),
		Gender: getString(data, "gender"),
	}
}

func convertDoctorInfo(data map[string]interface{}) DoctorInfo {
	return DoctorInfo{
		Name:           getString(data, "name"),
		Specialization: getString(data, "specialization"),
		Registration:   getString(data, "registration_number"),
	}
}

func convertMedicines(data []map[string]interface{}) []Medicine {
	medicines := make([]Medicine, 0, len(data))
	for _, m := range data {
		medicines = append(medicines, Medicine{
			Name:      getString(m, "name"),
			Dosage:    getString(m, "dosage"),
			Frequency: getString(m, "frequency"),
			Timing:    getString(m, "timing"),
			Duration:  getString(m, "duration"),
			Quantity:  getInt(m, "quantity"),
		})
	}
	return medicines
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getInt(m map[string]interface{}, key string) int {
	if v, ok := m[key]; ok {
		switch val := v.(type) {
		case int:
			return val
		case float64:
			return int(val)
		}
	}
	return 0
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
	// TODO: Add pagination support
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
