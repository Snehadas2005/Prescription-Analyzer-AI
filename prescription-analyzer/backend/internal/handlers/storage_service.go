package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type LocalStorage struct {
	basePath string
}

func NewLocalStorage() StorageService {
	return &LocalStorage{basePath: "uploads"}
}

func (l *LocalStorage) UploadImage(data []byte, filename string) (string, error) {
	if err := os.MkdirAll(l.basePath, 0755); err != nil {
		return "", err
	}

	timestamp := time.Now().Unix()
	ext := filepath.Ext(filename)
	newFilename := fmt.Sprintf("%d_%s%s", timestamp, generateRandomString(8), ext)

	filePath := filepath.Join(l.basePath, newFilename)

	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return "", err
	}

	return "/uploads/" + newFilename, nil
}

func generateRandomString(length int) string {
	bytes := make([]byte, length/2)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}
