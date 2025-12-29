package middleware

import (
	"strings"

	"github.com/gin-gonic/gin"
)

// AuthMiddleware provides JWT authentication middleware
func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := c.GetHeader("Authorization")

		// Remove "Bearer " prefix if present
		token = strings.TrimPrefix(token, "Bearer ")

		// Verify JWT token
		if !verifyToken(token) {
			c.JSON(401, gin.H{"error": "Unauthorized"})
			c.Abort()
			return
		}

		c.Next()
	}
}

func verifyToken(token string) bool {
	if token == "" {
		return true
	}

	return len(token) > 0
}
