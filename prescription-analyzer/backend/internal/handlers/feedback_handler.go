package handlers

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type FeedbackHandler struct {
	db *mongo.Database
}

type Feedback struct {
	ID             primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	PrescriptionID string             `json:"prescription_id" bson:"prescription_id"`
	FeedbackType   string             `json:"feedback_type" bson:"feedback_type"` // "correction", "confirmation", "rating"
	Corrections    *Corrections       `json:"corrections,omitempty" bson:"corrections,omitempty"`
	Rating         int                `json:"rating,omitempty" bson:"rating,omitempty"`
	Comments       string             `json:"comments,omitempty" bson:"comments,omitempty"`
	UserID         string             `json:"user_id,omitempty" bson:"user_id,omitempty"`
	Timestamp      time.Time          `json:"timestamp" bson:"timestamp"`
}

type Corrections struct {
	Patient   *PatientInfo `json:"patient,omitempty" bson:"patient,omitempty"`
	Doctor    *DoctorInfo  `json:"doctor,omitempty" bson:"doctor,omitempty"`
	Medicines []Medicine   `json:"medicines,omitempty" bson:"medicines,omitempty"`
}

func NewFeedbackHandler(db *mongo.Database) *FeedbackHandler {
	return &FeedbackHandler{db: db}
}

// Submit handles feedback submission
func (h *FeedbackHandler) Submit(c *gin.Context) {
	var feedback Feedback

	if err := c.ShouldBindJSON(&feedback); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid feedback data"})
		return
	}

	// Validate feedback type
	if feedback.FeedbackType != "correction" &&
		feedback.FeedbackType != "confirmation" &&
		feedback.FeedbackType != "rating" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid feedback type"})
		return
	}

	feedback.Timestamp = time.Now()

	// Save feedback to database
	collection := h.db.Collection("feedback")
	result, err := collection.InsertOne(c.Request.Context(), feedback)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save feedback"})
		return
	}

	feedback.ID = result.InsertedID.(primitive.ObjectID)

	// Update prescription status
	if err := h.updatePrescriptionStatus(c, feedback); err != nil {
		// Log error but don't fail the request
		c.JSON(http.StatusOK, gin.H{
			"success": true,
			"data":    feedback,
			"warning": "Failed to update prescription status",
		})
		return
	}

	// Export feedback to CSV for training
	go h.exportFeedbackToCSV(feedback)

	// Trigger training check
	go h.checkTrainingThreshold()

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    feedback,
		"message": "Feedback received successfully",
	})
}

// GetFeedbackStats returns feedback statistics
func (h *FeedbackHandler) GetFeedbackStats(c *gin.Context) {
	collection := h.db.Collection("feedback")

	// Count by feedback type
	pipeline := []bson.M{
		{
			"$group": bson.M{
				"_id":   "$feedback_type",
				"count": bson.M{"$sum": 1},
			},
		},
	}

	cursor, err := collection.Aggregate(c.Request.Context(), pipeline)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get statistics"})
		return
	}
	defer cursor.Close(c.Request.Context())

	var stats []bson.M
	if err = cursor.All(c.Request.Context(), &stats); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse statistics"})
		return
	}

	// Get average rating
	ratingPipeline := []bson.M{
		{
			"$match": bson.M{
				"feedback_type": "rating",
				"rating":        bson.M{"$exists": true},
			},
		},
		{
			"$group": bson.M{
				"_id":        nil,
				"avg_rating": bson.M{"$avg": "$rating"},
				"count":      bson.M{"$sum": 1},
			},
		},
	}

	ratingCursor, err := collection.Aggregate(c.Request.Context(), ratingPipeline)
	if err == nil {
		defer ratingCursor.Close(c.Request.Context())

		var ratingStats []bson.M
		if err = ratingCursor.All(c.Request.Context(), &ratingStats); err == nil && len(ratingStats) > 0 {
			stats = append(stats, bson.M{
				"type":  "average_rating",
				"value": ratingStats[0]["avg_rating"],
				"count": ratingStats[0]["count"],
			})
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    stats,
	})
}

// updatePrescriptionStatus updates the prescription based on feedback
func (h *FeedbackHandler) updatePrescriptionStatus(c *gin.Context, feedback Feedback) error {
	collection := h.db.Collection("prescriptions")

	update := bson.M{
		"$set": bson.M{
			"updated_at": time.Now(),
		},
	}

	if feedback.FeedbackType == "correction" {
		update["$set"].(bson.M)["status"] = "corrected"

		// Update with corrected data
		if feedback.Corrections != nil {
			if feedback.Corrections.Patient != nil {
				update["$set"].(bson.M)["patient"] = feedback.Corrections.Patient
			}
			if feedback.Corrections.Doctor != nil {
				update["$set"].(bson.M)["doctor"] = feedback.Corrections.Doctor
			}
			if feedback.Corrections.Medicines != nil {
				update["$set"].(bson.M)["medicines"] = feedback.Corrections.Medicines
			}
		}
	} else if feedback.FeedbackType == "confirmation" {
		update["$set"].(bson.M)["status"] = "verified"
	}

	_, err := collection.UpdateOne(
		c.Request.Context(),
		bson.M{"prescription_id": feedback.PrescriptionID},
		update,
	)

	return err
}

// exportFeedbackToCSV exports feedback to CSV for training
func (h *FeedbackHandler) exportFeedbackToCSV(feedback Feedback) {
	// This runs in background
	csvPath := "ml-service/data/feedback/feedback_log.csv"

	// Check if file exists
	fileExists := true
	if _, err := os.Stat(csvPath); os.IsNotExist(err) {
		fileExists = false
	}

	// Open file for appending
	file, err := os.OpenFile(csvPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header if new file
	if !fileExists {
		header := []string{
			"prescription_id",
			"feedback_type",
			"rating",
			"has_corrections",
			"timestamp",
		}
		_ = writer.Write(header)
	}

	// Write feedback data
	hasCorrections := "false"
	if feedback.Corrections != nil {
		hasCorrections = "true"
	}

	record := []string{
		feedback.PrescriptionID,
		feedback.FeedbackType,
		strconv.Itoa(feedback.Rating),
		hasCorrections,
		feedback.Timestamp.Format(time.RFC3339),
	}

	_ = writer.Write(record)

	// Also save detailed JSON for corrections
	if feedback.FeedbackType == "correction" && feedback.Corrections != nil {
		h.saveCorrectionDetails(feedback)
	}
}

// saveCorrectionDetails saves detailed correction data as JSON
func (h *FeedbackHandler) saveCorrectionDetails(feedback Feedback) {
	detailsDir := "ml-service/data/feedback/details"
	_ = os.MkdirAll(detailsDir, 0755)

	filename := filepath.Join(detailsDir, feedback.PrescriptionID+".json")

	file, err := os.Create(filename)
	if err != nil {
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	_ = encoder.Encode(feedback)
}

// checkTrainingThreshold checks if training should be triggered
func (h *FeedbackHandler) checkTrainingThreshold() {
	ctx := context.Background()

	collection := h.db.Collection("feedback")
	logsCollection := h.db.Collection("training_logs")

	// Get last training time from training_logs
	var lastLog bson.M
	opts := options.FindOne().SetSort(bson.D{{Key: "timestamp", Value: -1}})

	err := logsCollection.FindOne(
		ctx,
		bson.M{},
		opts,
	).Decode(&lastLog)

	var lastTraining time.Time
	if err == nil {
		if ts, ok := lastLog["timestamp"].(primitive.DateTime); ok {
			lastTraining = ts.Time()
		} else {
			lastTraining = time.Now().AddDate(0, 0, -30)
		}
	} else {
		// No previous training, use 30 days ago
		lastTraining = time.Now().AddDate(0, 0, -30)
	}

	// Count new feedback
	count, err := collection.CountDocuments(
		ctx,
		bson.M{
			"timestamp":     bson.M{"$gte": lastTraining},
			"feedback_type": "correction",
		},
	)

	if err != nil {
		return
	}

	// Trigger training if we have 100+ new corrections
	if count >= 100 {
		h.triggerTraining()
	}
}

// triggerTraining triggers the ML training pipeline
func (h *FeedbackHandler) triggerTraining() {
	mlServiceURL := getEnv("ML_SERVICE_URL", "http://localhost:8000")

	req, err := http.NewRequest("POST", mlServiceURL+"/train", nil)
	if err != nil {
		return
	}

	client := &http.Client{Timeout: 5 * time.Second}
	_, _ = client.Do(req)

	// Log training trigger
	ctx := context.Background()
	logsCollection := h.db.Collection("training_logs")
	_, _ = logsCollection.InsertOne(ctx, bson.M{
		"timestamp": time.Now(),
		"status":    "triggered",
		"reason":    "threshold_reached",
	})
}
