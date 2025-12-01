package main

import (
	"log"
	"prescription-analyzer/backend/internal/database"
	"prescription-analyzer/internal/handlers"

	"github.com/gin-gonic/gin"
)

func main() {
	// Initialize database
	db, err := database.NewMongoDB()
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	// Initialize handlers
	prescriptionHandler := handlers.NewPrescriptionHandler(db)
	feedbackHandler := handlers.NewFeedbackHandler(db)

	// Setup router
	r := gin.Default()
	r.Use(corsMiddleware())

	// Routes
	api := r.Group("/api/v1")
	{
		api.POST("/upload", prescriptionHandler.Upload)
		api.GET("/prescription/:id", prescriptionHandler.Get)
		api.POST("/feedback", feedbackHandler.Submit)
		api.GET("/history", prescriptionHandler.GetHistory)
	}

	// Start server
	log.Println("Server starting on :8080")
	r.Run(":8080")
}

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}
