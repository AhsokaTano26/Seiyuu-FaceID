package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSummarizeUsesNearestRankAndDoesNotMutateSamples(t *testing.T) {
	samples := []float64{5, 1, 3, 2, 4}
	stats := summarize(samples)
	if stats.Mean != 3 || stats.P50 != 3 || stats.P95 != 5 || samples[0] != 5 {
		t.Fatal(stats, samples)
	}
	if got := summarize([]float64{7}); got.Mean != 7 || got.P50 != 7 || got.P95 != 7 {
		t.Fatal(got)
	}
}

func TestBenchmarkReusesUploadAndExcludesWarmups(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/recognize" || r.Header.Get("Authorization") != "Bearer private-key" {
			t.Error("unexpected request")
		}
		if err := r.ParseMultipartForm(1024); err != nil {
			t.Fatal(err)
		}
		defer r.MultipartForm.RemoveAll()
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		var b bytes.Buffer
		_, _ = b.ReadFrom(file)
		if b.String() != "image bytes" || header.Filename != "benchmark-image" {
			t.Error("upload changed")
		}
		processing := 99
		if calls > 2 {
			processing = 4
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"face_count": 0, "faces": []any{}, "processing_ms": processing})
	}))
	defer server.Close()
	result, err := benchmark(server.Client(), server.URL, []byte("image bytes"), "private-key", 3, 2)
	if err != nil || calls != 5 || result.Samples != 3 || result.Warmups != 2 || result.ServerProcessingMean != 4 || result.Wall.Mean <= 0 || math.IsNaN(result.Wall.P95) {
		t.Fatal(result, err, calls)
	}
	encoded, _ := json.Marshal(result)
	if strings.Contains(string(encoded), "private") || strings.Contains(string(encoded), "image bytes") {
		t.Fatal("private details in report")
	}
}

func TestBenchmarkRejectsErrorsAndMalformedResponsesWithoutEcho(t *testing.T) {
	for _, tc := range []struct {
		status int
		body   string
	}{{401, `{"secret":"sensitive-response"}`}, {200, `{`}, {200, `{"face_count":0,"faces":[]}`}, {200, `{"face_count":2,"faces":[],"processing_ms":1}`}, {200, `{"face_count":0,"faces":null,"processing_ms":1}`}, {200, `{"face_count":0,"faces":[],"processing_ms":-1}`}, {200, `{"face_count":0,"faces":[],"processing_ms":1} trailing`}} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tc.status)
			_, _ = w.Write([]byte(tc.body))
		}))
		_, err := benchmark(&http.Client{Timeout: time.Second}, server.URL, []byte("image"), "", 1, 0)
		server.Close()
		if err == nil || strings.Contains(err.Error(), tc.body) || strings.Contains(err.Error(), server.URL) {
			t.Fatal("missing or leaking error", err)
		}
	}
}

func TestBenchmarkArgumentsAreBounded(t *testing.T) {
	for _, tc := range []struct {
		url           string
		runs, warmups int
	}{{"ftp://host", 1, 0}, {"http://user:secret@host", 1, 0}, {"http://host?private=yes", 1, 0}, {"http://host", 0, 0}, {"http://host", 10001, 0}, {"http://host", 1, -1}, {"http://host", 1, 1001}} {
		if err := validateArguments(tc.url, tc.runs, tc.warmups); err == nil {
			t.Fatal(tc)
		}
	}
	if err := validateArguments("http://127.0.0.1:8000", 1, 0); err != nil {
		t.Fatal(err)
	}
}
