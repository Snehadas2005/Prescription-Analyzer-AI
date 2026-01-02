package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

type FeedbackHandler struct {
	db           *mongo.Database
	mlServiceURL string
}

type Feedback struct {
	ID             primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	PrescriptionID string             `json:"prescription_id" bson:"prescription_id"`
	FeedbackType   string             `json:"feedback_type" bson:"feedback_type"`
	Corrections    interface{}        `json:"corrections,omitempty" bson:"corrections,omitempty"`
	Rating         int                `json:"rating,omitempty" bson:"rating,omitempty"`
	Comments       string             `json:"comments,omitempty" bson:"comments,omitempty"`
	Timestamp      time.Time          `json:"timestamp" bson:"timestamp"`
}

func NewFeedbackHandler(db *mongo.Database) *FeedbackHandler {
	mlURL := os.Getenv("ML_SERVICE_URL")
	if mlURL == "" {
		mlURL = "http://localhost:8000"
	}
	return &FeedbackHandler{
		db:           db,
		mlServiceURL: mlURL,
	}
}

func (h *FeedbackHandler) Submit(c *gin.Context) {
	var feedback Feedback

	if err := c.ShouldBindJSON(&feedback); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid feedback data"})
		return
	}

	feedback.Timestamp = time.Now()

	collection := h.db.Collection("feedback")
	result, err := collection.InsertOne(c.Request.Context(), feedback)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save feedback"})
		return
	}

	feedback.ID = result.InsertedID.(primitive.ObjectID)

	// Forward feedback to ML service for self-learning
	go h.forwardToMLService(feedback)

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    feedback,
		"message": "Feedback received successfully. AI knowledge base update triggered.",
	})
}

func (h *FeedbackHandler) forwardToMLService(feedback Feedback) {
	jsonData, err := json.Marshal(feedback)
	if err != nil {
		log.Printf("❌ Failed to marshal feedback: %v", err)
		return
	}

	req, err := http.NewRequest("POST", h.mlServiceURL+"/feedback", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("❌ Failed to create feedback request: %v", err)
		return
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("❌ Failed to forward feedback to ML service: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("⚠️ ML service returned error status for feedback: %d", resp.StatusCode)
		return
	}

	log.Printf("✅ Feedback successfully forwarded to ML service for %s", feedback.PrescriptionID)
}

func (h *FeedbackHandler) GetFeedbackStats(c *gin.Context) {
	collection := h.db.Collection("feedback")
	ctx := context.Background()

	pipeline := []bson.M{
		{
			"$group": bson.M{
				"_id":   "$feedback_type",
				"count": bson.M{"$sum": 1},
			},
		},
	}

	cursor, err := collection.Aggregate(ctx, pipeline)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get statistics"})
		return
	}
	defer cursor.Close(ctx)

	var stats []bson.M
	if err = cursor.All(ctx, &stats); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse statistics"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    stats,
	})
}
