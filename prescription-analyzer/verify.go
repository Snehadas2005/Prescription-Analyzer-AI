package main

import (
	"log"
	"time"

	"github.com/getsentry/sentry-go"
)

func main() {
	err := sentry.Init(sentry.ClientOptions{
		Dsn: "https://12ce480cb03492afe32602fd6267cdee@o4510459767488512.ingest.us.sentry.io/4510459778433024",
	})
	if err != nil {
		log.Fatalf("sentry.Init: %s", err)
	}

	defer sentry.Flush(2 * time.Second)

	sentry.CaptureMessage("It works!")
}
