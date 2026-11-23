package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeBackend struct {
	calls       atomic.Int32
	closed      atomic.Int32
	fail        error
	started     chan struct{}
	release     chan struct{}
	once        sync.Once
	seenOptions []Options
	mu          sync.Mutex
}

func (b *fakeBackend) Model() ModelInfo {
	return ModelInfo{ModelID: "model", RecognitionID: "native-config", ClassCount: 2, ClassNames: []string{"first", "second"}, Defaults: Options{TopK: 2, ProbabilityThreshold: .95}}
}
func (b *fakeBackend) Process(data []byte, o Options) (CoreResult, error) {
	b.calls.Add(1)
	b.mu.Lock()
	b.seenOptions = append(b.seenOptions, o)
	b.mu.Unlock()
	if b.started != nil {
		b.once.Do(func() { close(b.started) })
		<-b.release
	}
	if b.fail != nil {
		return CoreResult{}, b.fail
	}
	if o.CompactDetection {
		faces := json.RawMessage(`[{"bbox":[0,0,1,2],"detector_score":0.75}]`)
		if o.ScoreThreshold > 0.75 {
			faces = json.RawMessage("[]")
		}
		return CoreResult{Width: 2, Height: 3, Faces: faces}, nil
	}
	return CoreResult{Width: 2, Height: 3, Faces: json.RawMessage(`[{"status":"review","identity":null,"candidates":[]}]`)}, nil
}
func (b *fakeBackend) Close() error { b.closed.Add(1); return nil }
func pngData() []byte {
	var b bytes.Buffer
	_ = png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 2, 3)))
	return b.Bytes()
}
func uploadRequest(t *testing.T, path, field string, files [][]byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	for _, data := range files {
		p, err := w.CreateFormFile(field, "../photo.png")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = p.Write(data)
	}
	_ = w.Close()
	r := httptest.NewRequest("POST", path, &body)
	r.Header.Set("Content-Type", w.FormDataContentType())
	return r
}
func serve(s *Server, r *http.Request) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	return w
}
func errorCode(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var body struct{ Error struct{ Code string } }
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	return body.Error.Code
}

func TestAuthProtectsAllV1RoutesAndHealthRemainsPublic(t *testing.T) {
	b := &fakeBackend{}
	s := New(b, "secret", DefaultLimits())
	for _, path := range []string{"/v1/model", "/v1/recognize", "/v1/unknown"} {
		w := serve(s, httptest.NewRequest("GET", path, nil))
		if w.Code != 401 || errorCode(t, w) != "unauthorized" {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body.String())
		}
	}
	for _, path := range []string{"/health", "/ready"} {
		if w := serve(s, httptest.NewRequest("GET", path, nil)); w.Code != 200 {
			t.Fatal(w.Code)
		}
	}
	r := httptest.NewRequest("GET", "/v1/model", nil)
	r.Header.Set("Authorization", "Bearer secret")
	w := serve(s, r)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"max_image_pixels":20000000`) {
		t.Fatal(w.Body.String())
	}
}
func TestRecognizePreservesResponseAndSanitizesFilename(t *testing.T) {
	b := &fakeBackend{}
	s := New(b, "", DefaultLimits())
	w := serve(s, uploadRequest(t, "/v1/recognize?top_k=1&probability_threshold=0.8", "file", [][]byte{pngData()}))
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["model_id"] != "model" || body["face_count"] != float64(1) || body["image"].(map[string]any)["filename"] != "photo.png" {
		t.Fatal(body)
	}
	if b.calls.Load() != 1 {
		t.Fatal(b.calls.Load())
	}
	if b.seenOptions[0].TopK != 1 || b.seenOptions[0].ProbabilityThreshold != .8 {
		t.Fatal(b.seenOptions)
	}
}
func TestInvalidParametersNeverReachCore(t *testing.T) {
	for _, q := range []string{"top_k=0", "top_k=3", "top_k=1.5", "probability_threshold=NaN", "probability_threshold=Inf", "probability_threshold=-0.1", "probability_threshold=1.1", "top_k=1&top_k=2"} {
		b := &fakeBackend{}
		s := New(b, "", DefaultLimits())
		w := serve(s, uploadRequest(t, "/v1/recognize?"+q, "file", [][]byte{pngData()}))
		if w.Code != 422 || b.calls.Load() != 0 {
			t.Fatalf("%s: %d %s", q, w.Code, w.Body.String())
		}
	}
}
func TestUploadsAreBoundedAndMalformedInputsRejected(t *testing.T) {
	cases := []struct {
		name   string
		data   []byte
		limits Limits
		status int
		code   string
	}{
		{"empty", nil, DefaultLimits(), 400, "empty_file"},
		{"broken", []byte("garbage"), DefaultLimits(), 400, "invalid_image"},
		{"gif", []byte("GIF89a"), DefaultLimits(), 415, "unsupported_image"},
		{"file", pngData(), Limits{MaxUploadBytes: 8, MaxRequestBytes: 4096, MaxImagePixels: 20000000, MaxBatchSize: 8}, 413, "file_too_large"},
		{"pixels", pngData(), Limits{MaxUploadBytes: 4096, MaxRequestBytes: 8192, MaxImagePixels: 5, MaxBatchSize: 8}, 413, "image_too_large"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := &fakeBackend{}
			s := New(b, "", tc.limits)
			w := serve(s, uploadRequest(t, "/v1/recognize", "file", [][]byte{tc.data}))
			if w.Code != tc.status || errorCode(t, w) != tc.code || b.calls.Load() != 0 {
				t.Fatalf("%d %s", w.Code, w.Body.String())
			}
		})
	}
}
func TestRequestBodyBoundedWithoutContentLength(t *testing.T) {
	b := &fakeBackend{}
	limits := DefaultLimits()
	limits.MaxRequestBytes = 80
	s := New(b, "", limits)
	r := uploadRequest(t, "/v1/recognize", "file", [][]byte{pngData()})
	r.ContentLength = -1
	w := serve(s, r)
	if w.Code != 413 || errorCode(t, w) != "request_too_large" || b.calls.Load() != 0 {
		t.Fatal(w.Code, w.Body.String())
	}
}
func TestBatchPreservesOrderAndPerFileErrors(t *testing.T) {
	b := &fakeBackend{}
	s := New(b, "", DefaultLimits())
	w := serve(s, uploadRequest(t, "/v1/recognize/batch", "files", [][]byte{pngData(), {}, pngData()}))
	var body struct{ Results []map[string]any }
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if w.Code != 200 || len(body.Results) != 3 || body.Results[1]["index"] != float64(1) || body.Results[1]["error"].(map[string]any)["code"] != "empty_file" || b.calls.Load() != 2 {
		t.Fatal(w.Code, w.Body.String())
	}
}
func TestBatchSizeValidatedBeforeInference(t *testing.T) {
	b := &fakeBackend{}
	s := New(b, "", DefaultLimits())
	files := make([][]byte, 9)
	for i := range files {
		files[i] = pngData()
	}
	w := serve(s, uploadRequest(t, "/v1/recognize/batch", "files", files))
	if w.Code != 422 || b.calls.Load() != 0 {
		t.Fatal(w.Code)
	}
}
func TestInferenceErrorDoesNotLeakPaths(t *testing.T) {
	b := &fakeBackend{fail: errors.New("/private/data/secret")}
	s := New(b, "", DefaultLimits())
	w := serve(s, uploadRequest(t, "/v1/recognize", "file", [][]byte{pngData()}))
	if w.Code != 500 || strings.Contains(w.Body.String(), "private") {
		t.Fatal(w.Code, w.Body.String())
	}
}
func TestCancelledQueueNeverCallsCoreAndCloseWaits(t *testing.T) {
	b := &fakeBackend{started: make(chan struct{}), release: make(chan struct{})}
	s := New(b, "", DefaultLimits())
	done := make(chan struct{})
	go func() { defer close(done); serve(s, uploadRequest(t, "/v1/recognize", "file", [][]byte{pngData()})) }()
	<-b.started
	ctx, cancel := context.WithCancel(context.Background())
	r := uploadRequest(t, "/v1/recognize", "file", [][]byte{pngData()}).WithContext(ctx)
	body, _ := io.ReadAll(r.Body)
	read := make(chan struct{})
	r.Body = &notifyingBody{Reader: bytes.NewReader(body), read: read}
	response := make(chan *httptest.ResponseRecorder, 1)
	go func() { response <- serve(s, r) }()
	<-read
	select {
	case <-response:
		t.Fatal("queued request completed before cancellation")
	case <-time.After(10 * time.Millisecond):
	}
	cancel()
	w := <-response
	if w.Code != 408 || b.calls.Load() != 1 {
		t.Fatal(w.Code, b.calls.Load())
	}
	closed := make(chan struct{})
	go func() { _ = s.Close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("closed active engine")
	case <-time.After(10 * time.Millisecond):
	}
	close(b.release)
	<-done
	<-closed
	_ = s.Close()
	if b.closed.Load() != 1 {
		t.Fatal(b.closed.Load())
	}
	if w := serve(s, httptest.NewRequest("GET", "/ready", nil)); w.Code != 503 {
		t.Fatal(w.Code)
	}
}

type notifyingBody struct {
	*bytes.Reader
	read chan struct{}
	once sync.Once
}

func (b *notifyingBody) Read(p []byte) (int, error) {
	n, e := b.Reader.Read(p)
	if b.Len() == 0 {
		b.once.Do(func() { close(b.read) })
	}
	return n, e
}
func (b *notifyingBody) Close() error { return nil }

func TestHealthDoesNotWaitForInference(t *testing.T) {
	b := &fakeBackend{started: make(chan struct{}), release: make(chan struct{})}
	s := New(b, "", DefaultLimits())
	done := make(chan struct{})
	r := uploadRequest(t, "/v1/detect", "file", [][]byte{pngData()})
	go func() { defer close(done); serve(s, r) }()
	<-b.started
	if w := serve(s, httptest.NewRequest("GET", "/health", nil)); w.Code != 200 {
		t.Fatal(w.Code)
	}
	close(b.release)
	<-done
	if !b.seenOptions[0].DetectionOnly {
		t.Fatal("detect used recognition")
	}
}

func TestQueueLimitReturnsBusyBeforeReadingUpload(t *testing.T) {
	s := New(&fakeBackend{}, "", DefaultLimits())
	for i := 0; i < cap(s.slots); i++ {
		s.slots <- struct{}{}
	}
	w := serve(s, uploadRequest(t, "/v1/recognize", "file", [][]byte{pngData()}))
	if w.Code != 503 || errorCode(t, w) != "server_busy" {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestChunkedMultipartEpilogueStillCountsTowardBodyLimit(t *testing.T) {
	b := &fakeBackend{}
	limits := DefaultLimits()
	limits.MaxRequestBytes = 2048
	s := New(b, "", limits)
	r := uploadRequest(t, "/v1/recognize", "file", [][]byte{pngData()})
	body, _ := io.ReadAll(r.Body)
	r.Body = io.NopCloser(io.MultiReader(bytes.NewReader(body), strings.NewReader(strings.Repeat("x", 4096))))
	r.ContentLength = -1
	w := serve(s, r)
	if w.Code != 413 || errorCode(t, w) != "request_too_large" || b.calls.Load() != 0 {
		t.Fatal(w.Code, w.Body.String(), b.calls.Load())
	}
}
func TestMissingFieldAndWrongMethod(t *testing.T) {
	s := New(&fakeBackend{}, "", DefaultLimits())
	w := serve(s, uploadRequest(t, "/v1/recognize", "wrong", [][]byte{pngData()}))
	if w.Code != 422 {
		t.Fatal(w.Code)
	}
	w = serve(s, httptest.NewRequest("GET", "/v1/recognize", nil))
	if w.Code != 405 {
		t.Fatal(w.Code)
	}
}
func TestMalformedMultipart(t *testing.T) {
	s := New(&fakeBackend{}, "", DefaultLimits())
	r := httptest.NewRequest("POST", "/v1/recognize", io.NopCloser(strings.NewReader("broken")))
	r.Header.Set("Content-Type", "multipart/form-data; boundary=bad")
	if w := serve(s, r); w.Code != 400 {
		t.Fatal(w.Code)
	}
}

func TestInternalDetection(t *testing.T) {
	b := &fakeBackend{}
	s := New(b, "secret", DefaultLimits())
	for _, tc := range []struct {
		addr, key, query string
		status           int
	}{
		{"127.0.0.1:4321", "", "", 401},
		{"8.8.8.8:4321", "secret", "", 403},
		{"127.0.0.1:4321", "secret", "?score_threshold=nan", 422},
		{"127.0.0.1:4321", "secret", "?score_threshold=0.75", 200},
	} {
		r := uploadRequest(t, "/api/v1/detect"+tc.query, "file", [][]byte{pngData()})
		r.RemoteAddr = tc.addr
		r.Header.Set("Authorization", "Bearer "+tc.key)
		r.Header.Set("X-Forwarded-For", "127.0.0.1")
		w := serve(s, r)
		if w.Code != tc.status {
			t.Fatalf("%s: %d %s", tc.addr, w.Code, w.Body.String())
		}
		if w.Code == 200 {
			var body map[string]any
			if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body["has_face"] != true || body["face_count"] != float64(1) || len(body) != 3 || !b.seenOptions[0].DetectionOnly || b.seenOptions[0].ScoreThreshold != 0.75 {
				t.Fatal(body, b.seenOptions)
			}
		}
	}
}

func TestInternalDetectionDeadline(t *testing.T) {
	t.Setenv("SEIYUU_FACEID_DETECT_TIMEOUT_MS", "75")
	b := &fakeBackend{started: make(chan struct{}), release: make(chan struct{})}
	limits, err := LimitsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	s := New(b, "", limits)
	defer close(b.release)
	req := func() *http.Request {
		r := uploadRequest(t, "/api/v1/detect", "file", [][]byte{pngData()})
		r.RemoteAddr = "127.0.0.1:1234"
		return r
	}
	first := make(chan *httptest.ResponseRecorder, 1)
	go func() { first <- serve(s, req()) }()
	select {
	case <-b.started:
	case <-time.After(time.Second):
		t.Fatal("detection did not start")
	}
	select {
	case w := <-first:
		if w.Code != 503 || errorCode(t, w) != "detection_timeout" {
			t.Fatal(w.Code, w.Body.String())
		}
	case <-time.After(time.Second):
		t.Fatal("inference exceeded deadline")
	}
	w := serve(s, req())
	if w.Code != 503 || errorCode(t, w) != "detection_timeout" || b.calls.Load() != 1 {
		t.Fatal(w.Code, w.Body.String(), b.calls.Load())
	}
}

func TestInternalSubnetAndImageLimits(t *testing.T) {
	t.Setenv("SEIYUU_FACEID_DETECT_CLIENT_CIDR", "172.29.0.0/24")
	t.Setenv("SEIYUU_FACEID_DETECT_MAX_PIXELS", "5")
	limits, err := LimitsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	b := &fakeBackend{}
	s := New(b, "", limits)
	for _, tc := range []struct {
		route, peer string
		data        []byte
		status      int
		code        string
	}{
		{detectPath, "172.29.0.2:7777", pngData(), 413, "image_too_large"},
		{detectPath, "172.30.0.2:7777", pngData(), 403, "internal_only"},
		{detectPath, "172.29.0.2:7777", []byte("bad"), 400, "invalid_image"},
		{"/v1/detect", "172.29.0.2:7777", pngData(), 200, ""},
	} {
		r := uploadRequest(t, tc.route, "file", [][]byte{tc.data})
		r.RemoteAddr = tc.peer
		w := serve(s, r)
		if w.Code != tc.status || (tc.code != "" && errorCode(t, w) != tc.code) {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	if b.calls.Load() != 1 {
		t.Fatal(b.calls.Load())
	}
}

func TestInternalDeadlineIncludesUpload(t *testing.T) {
	t.Setenv("SEIYUU_FACEID_DETECT_TIMEOUT_MS", "100")
	limits, err := LimitsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	b := &fakeBackend{}
	server := httptest.NewServer(New(b, "", limits))
	defer server.Close()
	reader, writer := io.Pipe()
	defer writer.Close()
	req, err := http.NewRequest("POST", server.URL+detectPath, reader)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "multipart/form-data; boundary=slow")
	go func() {
		_, _ = io.WriteString(writer, "--slow\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.png\"\r\n\r\n")
	}()
	client := server.Client()
	client.Timeout = time.Second * 2
	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var payload struct{ Error struct{ Code string } }
	if err = json.NewDecoder(res.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 503 || payload.Error.Code != "detection_timeout" || b.calls.Load() != 0 {
		t.Fatal(res.StatusCode, payload, b.calls.Load())
	}
}
