package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func modelFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	assets := map[string]string{}
	for _, name := range []string{"adaface_ir101.onnx", "w600k_r50.onnx", "classifier.npz"} {
		data := []byte("test " + name)
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		assets[name] = hex.EncodeToString(sum[:])
	}
	data := []byte("detector")
	_ = os.WriteFile(filepath.Join(dir, "detector.onnx"), data, 0600)
	sum := sha256.Sum256(data)
	model := map[string]any{"schema_version": 1, "model_id": assetModelID(assets), "source_assets": assets, "detector_sha256": hex.EncodeToString(sum[:]), "models": map[string]string{"adaface": "adaface_ir101.onnx", "r50": "w600k_r50.onnx", "detector": "detector.onnx"}}
	body, _ := json.Marshal(model)
	path := filepath.Join(dir, "native-model.json")
	_ = os.WriteFile(path, body, 0600)
	sum = sha256.Sum256(body)
	_ = os.WriteFile(path+".sha256", []byte(hex.EncodeToString(sum[:])+"  native-model.json\n"), 0600)
	return path
}
func TestModelIntegrityVerifiesEveryArtifact(t *testing.T) {
	path := modelFixture(t)
	if _, _, err := verifyModel(path); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(filepath.Dir(path), "w600k_r50.onnx"), []byte("corrupt"), 0600)
	if _, _, err := verifyModel(path); err == nil {
		t.Fatal("accepted corrupt ONNX")
	}
}
func TestModelExportChecksumMandatory(t *testing.T) {
	path := modelFixture(t)
	_ = os.Remove(path + ".sha256")
	if _, _, err := verifyModel(path); err == nil {
		t.Fatal("accepted missing checksum")
	}
}
func TestModelExportChecksumDetectsHeadChanges(t *testing.T) {
	path := modelFixture(t)
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	_, _ = f.WriteString(" ")
	_ = f.Close()
	if _, _, err := verifyModel(path); err == nil {
		t.Fatal("accepted changed JSON")
	}
}
func TestModelPathsMustMatchChecksummedAssets(t *testing.T) {
	path := modelFixture(t)
	body, _ := os.ReadFile(path)
	body = []byte(strings.Replace(string(body), `"adaface":"adaface_ir101.onnx"`, `"adaface":"w600k_r50.onnx"`, 1))
	_ = os.WriteFile(path, body, 0600)
	sum := sha256.Sum256(body)
	_ = os.WriteFile(path+".sha256", []byte(hex.EncodeToString(sum[:])), 0600)
	if _, _, err := verifyModel(path); err == nil {
		t.Fatal("accepted unchecked model path")
	}
}
