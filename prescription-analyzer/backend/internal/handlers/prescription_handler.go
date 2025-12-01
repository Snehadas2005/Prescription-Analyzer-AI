package handlers

import (
	"bytes"
	"context"
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
	"go.mongodb.org/mongo-driver/mongo/options"
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

type MLExtractionResult struct {
	ID         string                 `json:"prescription_id"`
	Patient    map[string]interface{} `json:"patient"`
	Doctor     map[string]interface{} `json:"doctor"`
	Medicines  []map[string]interface{} `json:"medicines"`
	Confidence float64                `json:"confidence_score"`
	RawText    string                 `json:"raw_text"`
}

// StorageService interface
type StorageService interface {
	UploadImage(data []byte, filename string) (string, error)
}

func NewPrescriptionHandler(db *mongo.Database) *PrescriptionHandler {
	return &PrescriptionHandler{
		db:             db,
		mlServiceURL:   getEnv("ML_SERVICE_URL", "http://localhost:8000"),
		storageService: NewLocalStorage(),
	}
}

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
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to extract information: %v", err)})
		return
	}

	prescription := Prescription{
		PrescriptionID: extractionResult.ID,
		Patient:        convertPatientInfo(extractionResult.Patient),
		Doctor:         convertDoctorInfo(extractionResult.Doctor),
		Medicines:      convertMedicines(extractionResult.Medicines),
		Confidence:     extractionResult.Confidence,
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

func (h *PrescriptionHandler) Get(c *gin.Context) {
	id := c.Param("id")
	var prescription Prescription
	collection := h.db.Collection("prescriptions")

	err := collection.FindOne(c.Request.Context(), bson.M{
		"prescription_id": id,
	}).Decode(&prescription)

	if err != nil {
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

func (h *PrescriptionHandler) GetHistory(c *gin.Context) {
	limit := 10
	page := 1
	
	if l := c.Query("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
	}
	if p := c.Query("page"); p != "" {
		fmt.Sscanf(p, "%d", &page)
	}

	skip := int64((page - 1) * limit)

	collection := h.db.Collection("prescriptions")
	findOptions := options.Find()
	findOptions.SetLimit(int64(limit))
	findOptions.SetSkip(skip)
	findOptions.SetSort(bson.D{{Key: "created_at", Value: -1}})

	cursor, err := collection.Find(c.Request.Context(), bson.M{}, findOptions)
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

	total, _ := collection.CountDocuments(c.Request.Context(), bson.M{})

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    prescriptions,
		"total":   total,
		"page":    page,
		"limit":   limit,
	})
}

func (h *PrescriptionHandler) callMLService(imageBytes []byte) (*MLExtractionResult, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", "prescription.jpg")
	if err != nil {
		return nil, err
	}

	if _, err = io.Copy(part, bytes.NewReader(imageBytes)); err != nil {
		return nil, err
	}

	writer.Close()

	req, err := http.NewRequest("POST", h.mlServiceURL+"/extract", body)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result MLExtractionResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

func (h *PrescriptionHandler) triggerLearningPipeline(prescription Prescription) {
	collection := h.db.Collection("training_data")
	ctx := context.Background()

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

	collection.InsertOne(ctx, trainingData)
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
		Registration:   getString(data, "registration"),
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

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
