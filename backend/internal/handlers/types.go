package handlers

import (
	"os"
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

// StorageService interface for image storage
type StorageService interface {
	UploadImage(data []byte, filename string) (string, error)
}

// PrescriptionHandler handles prescription-related requests
type PrescriptionHandler struct {
	db             *mongo.Database
	storageService StorageService
	mlServiceURL   string
}

// NewPrescriptionHandler creates a new prescription handler
func NewPrescriptionHandler(db *mongo.Database) *PrescriptionHandler {
	return &PrescriptionHandler{
		db:             db,
		storageService: NewLocalStorage(),
		mlServiceURL:   getEnv("ML_SERVICE_URL", "http://localhost:8000"),
	}
}

// Prescription represents a prescription document
type Prescription struct {
	ID             primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	PrescriptionID string             `json:"prescription_id" bson:"prescription_id"`
	Patient        PatientInfo        `json:"patient" bson:"patient"`
	Doctor         DoctorInfo         `json:"doctor" bson:"doctor"`
	Medicines      []Medicine         `json:"medicines" bson:"medicines"`
	Diagnosis      []string           `json:"diagnosis" bson:"diagnosis"`
	Confidence     float64            `json:"confidence" bson:"confidence"`
	ImageURL       string             `json:"image_url" bson:"image_url"`
	RawText        string             `json:"raw_text" bson:"raw_text"`
	Status         string             `json:"status" bson:"status"`
	CreatedAt      time.Time          `json:"created_at" bson:"created_at"`
	UpdatedAt      time.Time          `json:"updated_at" bson:"updated_at"`
}

// PatientInfo represents patient information
type PatientInfo struct {
	Name   string `json:"name" bson:"name"`
	Age    string `json:"age" bson:"age"`
	Gender string `json:"gender" bson:"gender"`
}

// DoctorInfo represents doctor information
type DoctorInfo struct {
	Name           string `json:"name" bson:"name"`
	Specialization string `json:"specialization" bson:"specialization"`
	Registration   string `json:"registration" bson:"registration"`
}

// Medicine represents a medicine entry
type Medicine struct {
	Name      string `json:"name" bson:"name"`
	Dosage    string `json:"dosage" bson:"dosage"`
	Frequency string `json:"frequency" bson:"frequency"`
	Timing    string `json:"timing" bson:"timing"`
	Duration  string `json:"duration" bson:"duration"`
	Quantity  int    `json:"quantity" bson:"quantity"`
}

// triggerLearningPipeline triggers the ML learning pipeline (placeholder)
func (h *PrescriptionHandler) triggerLearningPipeline(prescription Prescription) {
	// TODO: Implement learning pipeline trigger
	// This would typically send data to a queue or trigger an async job
}

// Helper function to get environment variables
func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
