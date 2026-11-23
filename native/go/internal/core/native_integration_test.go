//go:build native && cgo

package core

import (
	"bytes"
	"encoding/json"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"seiyuu-faceid/native/internal/httpapi"
)

// Private model assets are opt-in; ordinary CI checks the HTTP adapter without
// downloading models or publishing evaluation images.
func TestNativeHTTPIntegration(t *testing.T) {
	model := os.Getenv("FACEID_NATIVE_TEST_MODEL")
	if model == "" {
		t.Skip("set FACEID_NATIVE_TEST_MODEL to a prepared local native model")
	}
	backend, err := Open(model)
	if err != nil {
		t.Fatal(err)
	}
	api := httpapi.New(backend, "integration-key", httpapi.DefaultLimits())
	defer api.Close()
	server := httptest.NewServer(api)
	defer server.Close()
	client := server.Client()
	request := func(method, path string, body io.Reader, contentType, auth string) (int, map[string]any) {
		t.Helper()
		req, err := http.NewRequest(method, server.URL+path, body)
		if err != nil {
			t.Fatal(err)
		}
		if contentType != "" {
			req.Header.Set("Content-Type", contentType)
		}
		if auth != "" {
			req.Header.Set("Authorization", "Bearer "+auth)
		}
		response, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		var result map[string]any
		if err = json.NewDecoder(response.Body).Decode(&result); err != nil {
			t.Fatal(err)
		}
		return response.StatusCode, result
	}
	for _, path := range []string{"/health", "/ready"} {
		if status, _ := request("GET", path, nil, "", ""); status != 200 {
			t.Fatal(path, status)
		}
	}
	if status, _ := request("GET", "/v1/model", nil, "", ""); status != 401 {
		t.Fatal(status)
	}
	if status, result := request("GET", "/v1/model", nil, "", "integration-key"); status != 200 || result["model_id"] != backend.Model().ModelID || result["class_count"] != float64(backend.Model().ClassCount) {
		t.Fatal(status, result)
	}
	var pngImage, jpegImage bytes.Buffer
	pixels := image.NewRGBA(image.Rect(0, 0, 128, 128))
	_ = png.Encode(&pngImage, pixels)
	_ = jpeg.Encode(&jpegImage, pixels, nil)
	post := func(path, field string, images ...[]byte) (int, map[string]any) {
		t.Helper()
		var body bytes.Buffer
		w := multipart.NewWriter(&body)
		for _, data := range images {
			p, err := w.CreateFormFile(field, "test.png")
			if err != nil {
				t.Fatal(err)
			}
			_, _ = p.Write(data)
		}
		_ = w.Close()
		return request("POST", path, &body, w.FormDataContentType(), "integration-key")
	}
	for _, data := range [][]byte{pngImage.Bytes(), jpegImage.Bytes()} {
		status, result := post("/api/v1/detect?score_threshold=0.8", "file", data)
		if status != 200 || result["has_face"] != false || result["face_count"] != float64(0) {
			t.Fatal("internal detection", status, result)
		}
		for _, path := range []string{"/v1/detect", "/v1/recognize"} {
			status, result := post(path, "file", data)
			if status != 200 || result["image"].(map[string]any)["width"] != float64(128) || result["model_id"] != backend.Model().ModelID {
				t.Fatal(path, status, result)
			}
		}
	}
	if status, _ := post("/v1/recognize?probability_threshold=NaN", "file", pngImage.Bytes()); status != 422 {
		t.Fatal(status)
	}
	status, result := post("/v1/recognize/batch", "files", pngImage.Bytes(), []byte("broken"), jpegImage.Bytes())
	if status != 200 {
		t.Fatal(status, result)
	}
	items := result["results"].([]any)
	if len(items) != 3 || items[1].(map[string]any)["error"].(map[string]any)["code"] != "invalid_image" {
		t.Fatal(result)
	}
	if _, err := backend.Process([]byte("invalid"), backend.Model().Defaults); err == nil {
		t.Fatal("native decoder accepted invalid image")
	}
	if err = api.Close(); err != nil {
		t.Fatal(err)
	}
	if status, _ := request("GET", "/ready", nil, "", ""); status != 503 {
		t.Fatal(status)
	}
}
