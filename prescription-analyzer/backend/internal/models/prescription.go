package models

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

type Prescription struct {
	ID             primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	PrescriptionID string             `json:"prescription_id" bson:"prescription_id"`
	UserID         string             `json:"user_id,omitempty" bson:"user_id,omitempty"`
	Patient        PatientInfo        `json:"patient" bson:"patient"`
	Doctor         DoctorInfo         `json:"doctor" bson:"doctor"`
	Medicines      []Medicine         `json:"medicines" bson:"medicines"`
	Confidence     float64            `json:"confidence" bson:"confidence"`
	ImageURL       string             `json:"image_url" bson:"image_url"`
	RawText        string             `json:"raw_text" bson:"raw_text"`
	Status         string             `json:"status" bson:"status"` // pending_verification, verified, corrected
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
