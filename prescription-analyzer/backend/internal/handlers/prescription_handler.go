package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

type PrescriptionHandler struct {
	db             *mongo.Database
	mlServiceURL   string
	storageService StorageService
}

type Prescription struct {
	ID             primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	PrescriptionID string             `json:"prescription_id" bson:"prescription_id"`
	Patient        PatientInfo        `json:"patient" bson:"patient"`
	Doctor         DoctorInfo         `json:"doctor" bson:"doctor"`
	Medicines      []Medicine         `json:"medicines" bson:"medicines"`
	Confidence     float64            `json:"confidence" bson:"confidence"`
	ImageURL       string             `json:"image_url" bson:"image_url"`
	RawText        string             `json:"raw_text" bson:"raw_text"`
	Status         string             `json:"status" bson:"status"`
	CreatedAt      time.Time          `json:"created_at" bson:"created_at"`
	UpdatedAt      time.Time          `json:"updated_at" bson:"updated_at"`
}

type PatientInfo struct {
	Name   string `json:"name" bson:"name"`
	Age    string `json:"age" bson:"age"`
	Gender string `json:"gender" bson:"gender"`
}

type DoctorInfo struct {
	Name           string `json:"name" bson:"name"`
	Specialization string `json:"specialization" bson:"specialization"`
	Registration   string `json:"registration" bson:"registration"`
}

type Medicine struct {
	Name      string `json:"name" bson:"name"`
	Dosage    string `json:"dosage" bson:"dosage"`
	Frequency string `json:"frequency" bson:"frequency"`
	Timing    string `json:"timing" bson:"timing"`
	Duration  string `json:"duration" bson:"duration"`
	Quantity  int    `json:"quantity" bson:"quantity"`
}

func NewPrescriptionHandler(db *mongo.Database) *PrescriptionHandler {
	return &PrescriptionHandler{
		db:             db,
		mlServiceURL:   getEnv("ML_SERVICE_URL", "http://localhost:8000"),
		storageService: NewSupabaseStorage(),
	}
}

// Upload handles prescription image upload and extraction
func (h *PrescriptionHandler) Upload(c *gin.Context) {
	// Get uploaded file
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}

	// Validate file type
	if !isValidImageType(file.Filename) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type"})
		return
	}

	// Read file
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

	// Upload to storage
	imageURL, err := h.storageService.UploadImage(fileBytes, file.Filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to upload image"})
		return
	}

	// Call ML service for extraction
	extractionResult, err := h.callMLService(fileBytes)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to extract information"})
		return
	}

	// Create prescription record
	prescription := Prescription{
		PrescriptionID: extractionResult.ID,
		Patient:        extractionResult.Patient,
		Doctor:         extractionResult.Doctor,
		Medicines:      extractionResult.Medicines,
		Confidence:     extractionResult.Confidence,
		ImageURL:       imageURL,
		RawText:        extractionResult.RawText,
		Status:         "pending_verification",
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}

	// Save to database
	collection := h.db.Collection("prescriptions")
	result, err := collection.InsertOne(c.Request.Context(), prescription)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save prescription"})
		return
	}

	prescription.ID = result.InsertedID.(primitive.ObjectID)

	// Trigger background job for continuous learning
	go h.triggerLearningPipeline(prescription)

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescription,
	})
}

// Get retrieves a prescription by ID
func (h *PrescriptionHandler) Get(c *gin.Context) {
	id := c.Param("id")

	var prescription Prescription
	collection := h.db.Collection("prescriptions")

	// Try to find by prescription_id first
	err := collection.FindOne(c.Request.Context(), bson.M{
		"prescription_id": id,
	}).Decode(&prescription)

	if err != nil {
		// Try to find by MongoDB _id
		objectID, err := primitive.ObjectIDFromHex(id)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid prescription ID"})
			return
		}

		err = collection.FindOne(c.Request.Context(), bson.M{
			"_id": objectID,
		}).Decode(&prescription)

		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Prescription not found"})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescription,
	})
}

// GetHistory retrieves prescription history
func (h *PrescriptionHandler) GetHistory(c *gin.Context) {
	// Get query parameters
	limit := c.DefaultQuery("limit", "10")
	page := c.DefaultQuery("page", "1")

	// Parse parameters
	var limitInt, pageInt int
	fmt.Sscanf(limit, "%d", &limitInt)
	fmt.Sscanf(page, "%d", &pageInt)

	skip := (pageInt - 1) * limitInt

	// Query database
	collection := h.db.Collection("prescriptions")
	cursor, err := collection.Find(
		c.Request.Context(),
		bson.M{},
		&mongo.FindOptions{
			Limit: int64Ptr(int64(limitInt)),
			Skip:  int64Ptr(int64(skip)),
			Sort:  bson.D{{Key: "created_at", Value: -1}},
		},
	)

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch history"})
		return
	}
	defer cursor.Close(c.Request.Context())

	var prescriptions []Prescription
	if err = cursor.All(c.Request.Context(), &prescriptions); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse history"})
		return
	}

	// Get total count
	total, err := collection.CountDocuments(c.Request.Context(), bson.M{})
	if err != nil {
		total = 0
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescriptions,
		"total":   total,
		"page":    pageInt,
		"limit":   limitInt,
	})
}

// callMLService calls the Python ML service for extraction
func (h *PrescriptionHandler) callMLService(imageBytes []byte) (*MLExtractionResult, error) {
	// Create multipart form
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", "prescription.jpg")
	if err != nil {
		return nil, err
	}

	_, err = io.Copy(part, bytes.NewReader(imageBytes))
	if err != nil {
		return nil, err
	}

	writer.Close()

	// Make HTTP request
	req, err := http.NewRequest("POST", h.mlServiceURL+"/extract", body)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	// Parse response
	var result MLExtractionResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// triggerLearningPipeline triggers the continuous learning pipeline
func (h *PrescriptionHandler) triggerLearningPipeline(prescription Prescription) {
	// This runs in background
	// Store the prescription for future training
	collection := h.db.Collection("training_data")

	trainingData := bson.M{
		"prescription_id": prescription.PrescriptionID,
		"image_url":       prescription.ImageURL,
		"extracted_data": bson.M{
			"patient":   prescription.Patient,
			"doctor":    prescription.Doctor,
			"medicines": prescription.Medicines,
		},
		"confidence": prescription.Confidence,
		"status":     "pending_feedback",
		"created_at": time.Now(),
	}

	collection.InsertOne(nil, trainingData)
}

type MLExtractionResult struct {
	ID         string      `json:"id"`
	Patient    PatientInfo `json:"patient"`
	Doctor     DoctorInfo  `json:"doctor"`
	Medicines  []Medicine  `json:"medicines"`
	Confidence float64     `json:"confidence"`
	RawText    string      `json:"raw_text"`
}

// Helper functions
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

func int64Ptr(i int64) *int64 {
	return &i
}

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
