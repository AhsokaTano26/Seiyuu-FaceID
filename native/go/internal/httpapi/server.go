// Package httpapi exposes the existing FaceID HTTP contract over a native backend.
package httpapi

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

type Limits struct {
	MaxUploadBytes  int64 `json:"max_upload_bytes"`
	MaxRequestBytes int64 `json:"max_request_bytes"`
	MaxImagePixels  int64 `json:"max_image_pixels"`
	MaxBatchSize    int   `json:"max_batch_size"`
	detection       detectionPolicy
}

func DefaultLimits() Limits {
	return Limits{MaxUploadBytes: 10 * 1024 * 1024, MaxRequestBytes: 81 * 1024 * 1024,
		MaxImagePixels: 20_000_000, MaxBatchSize: 8}
}

type Options struct {
	TopK                 int     `json:"top_k"`
	ProbabilityThreshold float64 `json:"probability_threshold"`
	DetectionOnly        bool    `json:"-"`
	CompactDetection     bool    `json:"-"`
	ScoreThreshold       float64 `json:"-"`
}
type ModelInfo struct {
	ModelID          string   `json:"model_id"`
	RecognitionID    string   `json:"recognition_id"`
	ClassCount       int      `json:"class_count"`
	ClassNames       []string `json:"class_names"`
	Defaults         Options  `json:"defaults"`
	Limits           Limits   `json:"limits"`
	SupportedFormats []string `json:"supported_formats"`
}
type CoreResult struct {
	Width  int             `json:"width"`
	Height int             `json:"height"`
	Faces  json.RawMessage `json:"faces"`
}
type Backend interface {
	Model() ModelInfo
	Process([]byte, Options) (CoreResult, error)
	Close() error
}
type APIError struct {
	Status  int
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *APIError) Error() string                         { return e.Message }
func apiError(status int, code, message string) *APIError { return &APIError{status, code, message} }

type Server struct {
	timedDetect http.Handler
	backend     Backend
	model       ModelInfo
	key         string
	limits      Limits
	gate        chan struct{}
	slots       chan struct{}
	closed      atomic.Bool
	closeOnce   sync.Once
	closeError  error
}

func New(backend Backend, key string, limits Limits) *Server {
	if limits.MaxUploadBytes < 1 || limits.MaxRequestBytes < 1 || limits.MaxImagePixels < 1 || limits.MaxBatchSize < 1 {
		panic("invalid HTTP upload limits")
	}
	if limits.detection.timeout == 0 {
		limits.detection = detectionPolicy{timeout: 5 * time.Second, maxPixels: limits.MaxImagePixels}
	}
	s := &Server{backend: backend, key: key, limits: limits, gate: make(chan struct{}, 1), slots: make(chan struct{}, 8)}
	s.timedDetect = http.TimeoutHandler(http.HandlerFunc(s.serve), limits.detection.timeout, timeoutBody)
	s.gate <- struct{}{}
	if backend != nil {
		s.model = backend.Model()
		s.model.Limits = limits
		s.model.SupportedFormats = []string{"JPEG", "PNG", "WEBP"}
	}
	return s
}
func (s *Server) Close() error {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		<-s.gate
		defer func() { s.gate <- struct{}{} }()
		if s.backend != nil {
			s.closeError = s.backend.Close()
		}
	})
	return s.closeError
}
func (s *Server) ready() bool { return s.backend != nil && !s.closed.Load() }
func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
func writeError(w http.ResponseWriter, err error) {
	e := &APIError{Status: 500, Code: "internal_error", Message: "An internal inference error occurred."}
	var known *APIError
	if errors.As(err, &known) {
		e = known
	}
	if e.Status == 401 {
		w.Header().Set("WWW-Authenticate", "Bearer")
	}
	writeJSON(w, e.Status, map[string]any{"error": map[string]string{"code": e.Code, "message": e.Message}})
}
func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	defer func() {
		if recover() != nil {
			writeError(w, errors.New("internal request failure"))
		}
	}()
	p := r.URL.Path
	if (strings.HasPrefix(p, "/v1/") || p == detectPath) && s.key != "" {
		fields := strings.Fields(r.Header.Get("Authorization"))
		if len(fields) != 2 || !strings.EqualFold(fields[0], "Bearer") || subtle.ConstantTimeCompare([]byte(fields[1]), []byte(s.key)) != 1 {
			writeError(w, apiError(401, "unauthorized", "A valid Bearer API key is required."))
			return
		}
	}
	methods := map[string]string{"/health": "GET", "/ready": "GET", "/v1/model": "GET", "/v1/detect": "POST", detectPath: "POST", "/v1/recognize": "POST", "/v1/recognize/batch": "POST"}
	method, ok := methods[p]
	if !ok {
		writeError(w, apiError(404, "not_found", "Not Found"))
		return
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		writeError(w, apiError(405, "method_not_allowed", "Method Not Allowed"))
		return
	}
	if p == "/health" {
		writeJSON(w, 200, map[string]string{"status": "ok"})
		return
	}
	if !s.ready() {
		writeError(w, apiError(503, "model_unavailable", "The model is not ready."))
		return
	}
	if p == "/ready" {
		writeJSON(w, 200, map[string]string{"status": "ready"})
		return
	}
	if p == "/v1/model" {
		writeJSON(w, 200, s.model)
		return
	}
	opts, err := s.options(r, p == "/v1/detect" || p == detectPath)
	if err != nil {
		writeError(w, err)
		return
	}
	// Limit request bodies in memory/on disk while the single engine is occupied.
	select {
	case s.slots <- struct{}{}:
		defer func() { <-s.slots }()
	default:
		writeError(w, apiError(503, "server_busy", "The inference queue is full."))
		return
	}
	if r.Context().Err() != nil {
		writeError(w, cancelled())
		return
	}
	if r.ContentLength > s.limits.MaxRequestBytes {
		writeError(w, apiError(413, "request_too_large", "The request exceeds the byte limit."))
		return
	}
	guard := &bodyGuard{ReadCloser: http.MaxBytesReader(w, r.Body, s.limits.MaxRequestBytes)}
	r.Body = guard
	defer r.Body.Close()
	if err = r.ParseMultipartForm(32 * 1024); err != nil {
		if r.MultipartForm != nil {
			defer r.MultipartForm.RemoveAll()
		}
		var limit *http.MaxBytesError
		var readErr net.Error
		if p == detectPath && errors.As(err, &readErr) && readErr.Timeout() {
			writeError(w, apiError(503, "detection_timeout", "The detection request timed out."))
		} else if guard.exceeded || errors.As(err, &limit) {
			writeError(w, apiError(413, "request_too_large", "The request exceeds the byte limit."))
		} else {
			writeError(w, apiError(400, "bad_request", "Invalid multipart request."))
		}
		return
	}
	if r.MultipartForm == nil {
		writeError(w, apiError(400, "bad_request", "Expected multipart/form-data."))
		return
	}
	defer r.MultipartForm.RemoveAll()
	// Multipart ends at its closing boundary, but an epilogue is still part of
	// the HTTP body and must count toward the request limit before inference.
	_, err = io.Copy(io.Discard, r.Body)
	if guard.exceeded {
		writeError(w, apiError(413, "request_too_large", "The request exceeds the byte limit."))
		return
	}
	if err != nil {
		var readErr net.Error
		if p == detectPath && errors.As(err, &readErr) && readErr.Timeout() {
			writeError(w, apiError(503, "detection_timeout", "The detection request timed out."))
		} else {
			writeError(w, apiError(400, "bad_request", "Could not read the complete request body."))
		}
		return
	}
	field := "file"
	batch := p == "/v1/recognize/batch"
	if batch {
		field = "files"
	}
	uploads := r.MultipartForm.File[field]
	if len(uploads) == 0 || (!batch && len(uploads) != 1) || (batch && len(uploads) > s.limits.MaxBatchSize) {
		writeError(w, apiError(422, "validation_error", "Invalid number of uploaded images."))
		return
	}
	if err = s.acquire(r.Context()); err != nil {
		if p == detectPath && errors.Is(r.Context().Err(), context.DeadlineExceeded) {
			err = apiError(503, "detection_timeout", "The detection request timed out.")
		}
		writeError(w, err)
		return
	}
	defer func() { s.gate <- struct{}{} }()
	if batch {
		results := make([]map[string]any, 0, len(uploads))
		for i, upload := range uploads {
			if r.Context().Err() != nil {
				writeError(w, cancelled())
				return
			}
			result, err := s.process(upload, opts)
			if err != nil {
				var known *APIError
				if !errors.As(err, &known) || known.Status >= 500 {
					writeError(w, err)
					return
				}
				result = map[string]any{"filename": safeFilename(upload.Filename), "error": map[string]string{"code": known.Code, "message": known.Message}}
			}
			result["index"] = i
			results = append(results, result)
		}
		writeJSON(w, 200, map[string]any{"results": results})
		return
	}
	result, err := s.process(uploads[0], opts)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, 200, result)
}
func cancelled() *APIError { return apiError(408, "request_cancelled", "The request was cancelled.") }
func (s *Server) acquire(ctx context.Context) error {
	if ctx.Err() != nil {
		return cancelled()
	}
	select {
	case <-ctx.Done():
		return cancelled()
	case <-s.gate:
	}
	if ctx.Err() != nil {
		s.gate <- struct{}{}
		return cancelled()
	}
	if !s.ready() {
		s.gate <- struct{}{}
		return apiError(503, "model_unavailable", "The model is not ready.")
	}
	return nil
}
func (s *Server) options(r *http.Request, detection bool) (Options, error) {
	options := s.model.Defaults
	options.DetectionOnly = detection
	options.CompactDetection = r.URL.Path == detectPath
	if detection && !options.CompactDetection {
		return options, nil
	}
	query, err := url.ParseQuery(r.URL.RawQuery)
	invalid := func() (Options, error) {
		return Options{}, apiError(422, "validation_error", "Invalid request parameters.")
	}
	if err != nil {
		return invalid()
	}
	if options.CompactDetection {
		if values, ok := query["score_threshold"]; ok {
			if len(values) != 1 {
				return invalid()
			}
			options.ScoreThreshold, err = strconv.ParseFloat(values[0], 64)
			if err != nil || !(options.ScoreThreshold > 0 && options.ScoreThreshold <= 1) {
				return invalid()
			}
		}
		return options, nil
	}
	if values, ok := query["top_k"]; ok {
		if len(values) != 1 {
			return invalid()
		}
		options.TopK, err = strconv.Atoi(values[0])
		if err != nil || options.TopK < 1 || options.TopK > s.model.ClassCount {
			return invalid()
		}
	}
	if values, ok := query["probability_threshold"]; ok {
		if len(values) != 1 {
			return invalid()
		}
		options.ProbabilityThreshold, err = strconv.ParseFloat(values[0], 64)
		if err != nil || math.IsNaN(options.ProbabilityThreshold) || math.IsInf(options.ProbabilityThreshold, 0) || options.ProbabilityThreshold < 0 || options.ProbabilityThreshold > 1 {
			return invalid()
		}
	}
	return options, nil
}

// Multipart parsing can turn a body-limit error into an unexpected EOF. Keep
// the original reader signal so chunked oversized requests still return 413.
type bodyGuard struct {
	io.ReadCloser
	exceeded bool
}

func (b *bodyGuard) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	var limit *http.MaxBytesError
	if errors.As(err, &limit) {
		b.exceeded = true
	}
	return n, err
}
func safeFilename(name string) string {
	name = path.Base(strings.ReplaceAll(name, "\\", "/"))
	runes := []rune(strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) {
			return r
		}
		return -1
	}, name))
	if len(runes) > 255 {
		runes = runes[:255]
	}
	name = string(runes)
	if name == "" || name == "." || name == "/" {
		return "image"
	}
	return name
}
func (s *Server) process(upload *multipart.FileHeader, options Options) (map[string]any, error) {
	started := time.Now()
	if upload.Size > s.limits.MaxUploadBytes {
		return nil, apiError(413, "file_too_large", "The uploaded file exceeds the byte limit.")
	}
	f, err := upload.Open()
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, s.limits.MaxUploadBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > s.limits.MaxUploadBytes {
		return nil, apiError(413, "file_too_large", "The uploaded file exceeds the byte limit.")
	}
	maxPixels := s.limits.MaxImagePixels
	if options.CompactDetection {
		maxPixels = s.limits.detection.maxPixels
	}
	if err = validateImage(data, maxPixels); err != nil {
		return nil, err
	}
	result, err := s.backend.Process(data, options)
	if err != nil {
		return nil, err
	}
	if result.Width < 1 || result.Height < 1 || int64(result.Width) > maxPixels/int64(result.Height) {
		return nil, errors.New("invalid core output dimensions")
	}
	var faces []json.RawMessage
	if err = json.Unmarshal(result.Faces, &faces); err != nil || faces == nil {
		return nil, errors.New("invalid core output faces")
	}
	if options.CompactDetection {
		return map[string]any{"has_face": len(faces) > 0, "face_count": len(faces), "faces": faces}, nil
	}
	return map[string]any{"model_id": s.model.ModelID, "image": map[string]any{"filename": safeFilename(upload.Filename), "width": result.Width, "height": result.Height}, "face_count": len(faces), "faces": faces, "processing_ms": math.Round(float64(time.Since(started).Nanoseconds())/1e3) / 1e3}, nil
}
