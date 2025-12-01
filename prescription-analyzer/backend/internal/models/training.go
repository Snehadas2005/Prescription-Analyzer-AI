package models

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

type TrainingLog struct {
	ID        primitive.ObjectID     `json:"id" bson:"_id,omitempty"`
	Version   string                 `json:"version" bson:"version"`
	Status    string                 `json:"status" bson:"status"` // triggered, running, completed, failed
	Metrics   map[string]interface{} `json:"metrics" bson:"metrics"`
	Reason    string                 `json:"reason" bson:"reason"`
	Timestamp time.Time              `json:"timestamp" bson:"timestamp"`
	Duration  float64                `json:"duration,omitempty" bson:"duration,omitempty"`
	Error     string                 `json:"error,omitempty" bson:"error,omitempty"`
}
