package main

import (
	"log"
	"net/http"
	"os"

	"prescription-analyzer-ai/internal/database"
	"prescription-analyzer-ai/internal/handlers"

	"github.com/gin-gonic/gin"
)

func main() {
	// Load environment variables
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Initialize database
	db, err := database.NewMongoDB()
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	// Initialize handlers
	prescriptionHandler := handlers.NewPrescriptionHandler(db)

	// Setup router
	r := gin.Default()
	r.Use(corsMiddleware())

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":  "healthy",
			"service": "prescription-analyzer-backend",
		})
	})

	// ML service connectivity check
	r.GET("/ml-health", func(c *gin.Context) {
		mlURL := os.Getenv("ML_SERVICE_URL")
		resp, err := http.Get(mlURL + "/health")
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error(), "ml_url": mlURL})
			return
		}
		defer resp.Body.Close()
		c.JSON(200, gin.H{"ml_status": resp.StatusCode, "ml_url": mlURL})
	})

	// Routes
	api := r.Group("/api/v1")
	{
		api.POST("/upload", prescriptionHandler.Upload)
		api.GET("/prescription/:id", prescriptionHandler.Get)
		api.GET("/history", prescriptionHandler.GetHistory)
	}

	// Start server
	log.Printf("🚀 Server starting on port %s\n", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		c.Writer.Header().Set("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}
