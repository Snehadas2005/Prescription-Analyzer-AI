package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
)

type FeedbackHandler struct {
	db *mongo.Database
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
	return &FeedbackHandler{db: db}
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

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    feedback,
		"message": "Feedback received successfully",
	})
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