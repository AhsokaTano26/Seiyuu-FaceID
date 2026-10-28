package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"seiyuu-faceid/native/internal/core"
	"seiyuu-faceid/native/internal/httpapi"
)

func main() {
	model := flag.String("model", "models/selected/native-model.json", "Path to exported native-model.json and its .sha256 sidecar")
	addr := flag.String("addr", "127.0.0.1:8000", "HTTP listen address")
	flag.Parse()
	if err := run(*model, *addr); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}
func run(model, addr string) error {
	// Set before initializing ONNX Runtime, including native process startup.
	if err := os.Setenv("ORT_DISABLE_TELEMETRY", "1"); err != nil {
		return err
	}
	backend, err := core.Open(model)
	if err != nil {
		return err
	}
	api := httpapi.New(backend, os.Getenv("SEIYUU_FACEID_API_KEY"), httpapi.DefaultLimits())
	defer api.Close()
	server := &http.Server{Addr: addr, Handler: api, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 60 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 * 1024}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	exited := make(chan error, 1)
	go func() {
		log.Printf("Native FaceID listening on %s (model %s)", addr, backend.Model().ModelID)
		exited <- server.ListenAndServe()
	}()
	select {
	case err := <-exited:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			_ = server.Close()
			return err
		}
		return nil
	}
}
