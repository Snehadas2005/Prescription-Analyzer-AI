package models

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

type Feedback struct {
	ID             primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	PrescriptionID string             `json:"prescription_id" bson:"prescription_id"`
	UserID         string             `json:"user_id,omitempty" bson:"user_id,omitempty"`
	FeedbackType   string             `json:"feedback_type" bson:"feedback_type"` // correction, confirmation, rating
	Corrections    *Corrections       `json:"corrections,omitempty" bson:"corrections,omitempty"`
	Rating         int                `json:"rating,omitempty" bson:"rating,omitempty"`
	Comments       string             `json:"comments,omitempty" bson:"comments,omitempty"`
	Timestamp      time.Time          `json:"timestamp" bson:"timestamp"`
}

type Corrections struct {
	Patient   *PatientInfo `json:"patient,omitempty" bson:"patient,omitempty"`
	Doctor    *DoctorInfo  `json:"doctor,omitempty" bson:"doctor,omitempty"`
	Medicines []Medicine   `json:"medicines,omitempty" bson:"medicines,omitempty"`
}
