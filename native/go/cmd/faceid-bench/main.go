// faceid-bench compares either HTTP implementation without a Python runtime.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"
)

const maxImageBytes = 10 * 1024 * 1024
const maxResponseBytes = 4 * 1024 * 1024

type latencyStats struct {
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P95  float64 `json:"p95"`
}
type report struct {
	Samples              int          `json:"sample_count"`
	Warmups              int          `json:"warmups"`
	Wall                 latencyStats `json:"wall_ms"`
	ServerProcessingMean float64      `json:"server_processing_ms_mean"`
}
type response struct {
	FaceCount    *int              `json:"face_count"`
	Faces        []json.RawMessage `json:"faces"`
	ProcessingMS *float64          `json:"processing_ms"`
}

func main() {
	endpoint := flag.String("url", "http://127.0.0.1:8000", "Base URL of the Python or native FaceID HTTP service")
	imagePath := flag.String("image", "", "Explicit local image file (required; at most 10 MiB)")
	runs := flag.Int("runs", 20, "Measured sequential requests (1-10000)")
	warmups := flag.Int("warmups", 3, "Unmeasured warm-up requests (0-1000)")
	flag.Parse()
	if err := runCLI(*endpoint, *imagePath, *runs, *warmups); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func runCLI(endpoint, imagePath string, runs, warmups int) error {
	if err := validateArguments(endpoint, runs, warmups); err != nil {
		return err
	}
	if imagePath == "" {
		return errors.New("an explicit local -image file is required")
	}
	file, err := os.Open(imagePath)
	if err != nil {
		return errors.New("could not open the input image")
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return errors.New("input must be a regular local file")
	}
	if info.Size() < 1 || info.Size() > maxImageBytes {
		return errors.New("input image must be between 1 byte and 10 MiB")
	}
	data, err := io.ReadAll(io.LimitReader(file, maxImageBytes+1))
	if err != nil {
		return errors.New("could not read the input image")
	}
	if len(data) < 1 || len(data) > maxImageBytes {
		return errors.New("input image must be between 1 byte and 10 MiB")
	}
	client := &http.Client{Timeout: 60 * time.Second, CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }}
	result, err := benchmark(client, endpoint, data, os.Getenv("SEIYUU_FACEID_API_KEY"), runs, warmups)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}
func validateArguments(endpoint string, runs, warmups int) error {
	if runs < 1 || runs > 10000 || warmups < 0 || warmups > 1000 {
		return errors.New("runs must be 1-10000 and warmups must be 0-1000")
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return errors.New("url must be an HTTP(S) base URL without credentials, query or fragment")
	}
	return nil
}
func summarize(samples []float64) latencyStats {
	if len(samples) == 0 {
		return latencyStats{}
	}
	ordered := append([]float64(nil), samples...)
	sort.Float64s(ordered)
	sum := 0.0
	for _, sample := range samples {
		sum += sample
	}
	percentile := func(q float64) float64 { return ordered[int(math.Ceil(q*float64(len(ordered))))-1] }
	return latencyStats{sum / float64(len(samples)), percentile(.5), percentile(.95)}
}
func benchmark(client *http.Client, endpoint string, imageBytes []byte, key string, runs, warmups int) (report, error) {
	var result report
	if err := validateArguments(endpoint, runs, warmups); err != nil {
		return result, err
	}
	if len(imageBytes) < 1 || len(imageBytes) > maxImageBytes {
		return result, errors.New("input image must be between 1 byte and 10 MiB")
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "benchmark-image")
	if err != nil {
		return result, errors.New("could not construct upload")
	}
	if _, err = part.Write(imageBytes); err != nil {
		return result, errors.New("could not construct upload")
	}
	if err = writer.Close(); err != nil {
		return result, errors.New("could not construct upload")
	}
	contentType := writer.FormDataContentType()
	endpoint = strings.TrimRight(endpoint, "/") + "/v1/recognize"
	walls, processing := make([]float64, 0, runs), make([]float64, 0, runs)
	for i := 0; i < warmups+runs; i++ {
		request, err := http.NewRequest("POST", endpoint, bytes.NewReader(body.Bytes()))
		if err != nil {
			return result, errors.New("could not construct HTTP request")
		}
		request.Header.Set("Content-Type", contentType)
		if key != "" {
			request.Header.Set("Authorization", "Bearer "+key)
		}
		started := time.Now()
		serverMS, err := requestOnce(client, request)
		elapsed := float64(time.Since(started).Nanoseconds()) / 1e6
		if err != nil {
			return result, fmt.Errorf("benchmark request %d failed: %w", i+1, err)
		}
		if i >= warmups {
			walls = append(walls, elapsed)
			processing = append(processing, serverMS)
		}
	}
	return report{Samples: runs, Warmups: warmups, Wall: summarize(walls), ServerProcessingMean: summarize(processing).Mean}, nil
}
func requestOnce(client *http.Client, request *http.Request) (float64, error) {
	reply, err := client.Do(request)
	if err != nil {
		return 0, errors.New("HTTP transport failure")
	}
	defer reply.Body.Close()
	if reply.StatusCode < 200 || reply.StatusCode >= 300 {
		_, _ = io.Copy(io.Discard, io.LimitReader(reply.Body, maxResponseBytes))
		return 0, fmt.Errorf("HTTP status %d", reply.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(reply.Body, maxResponseBytes+1))
	if err != nil {
		return 0, errors.New("could not read HTTP response")
	}
	if len(data) > maxResponseBytes {
		return 0, errors.New("HTTP response exceeds benchmark limit")
	}
	var decoded response
	if err = json.Unmarshal(data, &decoded); err != nil || decoded.FaceCount == nil || *decoded.FaceCount < 0 || decoded.Faces == nil || len(decoded.Faces) != *decoded.FaceCount || decoded.ProcessingMS == nil {
		return 0, errors.New("invalid recognition response")
	}
	if math.IsNaN(*decoded.ProcessingMS) || math.IsInf(*decoded.ProcessingMS, 0) || *decoded.ProcessingMS < 0 {
		return 0, errors.New("invalid processing time in response")
	}
	return *decoded.ProcessingMS, nil
}
