package core

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type modelManifest struct {
	SchemaVersion  int               `json:"schema_version"`
	ModelID        string            `json:"model_id"`
	SourceAssets   map[string]string `json:"source_assets"`
	DetectorSHA256 string            `json:"detector_sha256"`
	Models         struct {
		Detector string `json:"detector"`
		AdaFace  string `json:"adaface"`
		R50      string `json:"r50"`
	} `json:"models"`
}

func assetModelID(assets map[string]string) string {
	keys := make([]string, 0, len(assets))
	for k := range assets {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	b.WriteByte('{')
	for i, key := range keys {
		if i > 0 {
			b.WriteString(", ")
		}
		k, _ := json.Marshal(key)
		v, _ := json.Marshal(assets[key])
		b.Write(k)
		b.WriteString(": ")
		b.Write(v)
	}
	b.WriteByte('}')
	digest := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(digest[:])
}
func readBounded(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	body, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, errors.New("model metadata is too large")
	}
	return body, nil
}
func validHash(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && strings.ToLower(value) == value
}
func fileHashMatches(path, expected string) error {
	if !validHash(expected) {
		return errors.New("invalid model checksum")
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("model asset unavailable: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return err
	}
	if subtle.ConstantTimeCompare([]byte(hex.EncodeToString(h.Sum(nil))), []byte(expected)) != 1 {
		return errors.New("model asset checksum mismatch")
	}
	return nil
}
func verifyModel(path string) (modelManifest, []byte, error) {
	var manifest modelManifest
	body, err := readBounded(path, 8*1024*1024)
	if err != nil {
		return manifest, nil, err
	}
	sidecar, err := readBounded(path+".sha256", 1024)
	if err != nil {
		return manifest, nil, fmt.Errorf("native export checksum unavailable: %w", err)
	}
	fields := strings.Fields(string(sidecar))
	if len(fields) == 0 || !validHash(fields[0]) {
		return manifest, nil, errors.New("invalid native export checksum")
	}
	digest := sha256.Sum256(body)
	if subtle.ConstantTimeCompare([]byte(hex.EncodeToString(digest[:])), []byte(fields[0])) != 1 {
		return manifest, nil, errors.New("native export checksum mismatch")
	}
	if err = json.Unmarshal(body, &manifest); err != nil {
		return manifest, nil, fmt.Errorf("invalid native model JSON: %w", err)
	}
	if manifest.SchemaVersion != 1 || len(manifest.SourceAssets) != 3 || manifest.ModelID != assetModelID(manifest.SourceAssets) {
		return manifest, nil, errors.New("unsupported native model manifest")
	}
	if manifest.Models.AdaFace != "adaface_ir101.onnx" || manifest.Models.R50 != "w600k_r50.onnx" || manifest.Models.Detector == "" || filepath.IsAbs(manifest.Models.Detector) {
		return manifest, nil, errors.New("model paths do not match checksummed assets")
	}
	dir := filepath.Dir(path)
	for _, name := range []string{"adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz"} {
		if err = fileHashMatches(filepath.Join(dir, name), manifest.SourceAssets[name]); err != nil {
			return manifest, nil, err
		}
	}
	if err = fileHashMatches(filepath.Join(dir, manifest.Models.Detector), manifest.DetectorSHA256); err != nil {
		return manifest, nil, err
	}
	return manifest, body, nil
}
