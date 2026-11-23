//go:build native && cgo

package core

/*
#cgo CFLAGS: -I${SRCDIR}/../../../include
#cgo LDFLAGS: -lfaceid_core
#include "faceid.h"
#include <stdlib.h>
*/
import "C"

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"runtime"
	"sync"
	"unsafe"

	"seiyuu-faceid/native/internal/httpapi"
)

type nativeEngine struct {
	mu     sync.Mutex
	handle *C.faceid_engine
	model  httpapi.ModelInfo
}

func Open(path string) (httpapi.Backend, error) {
	manifest, body, err := verifyModel(path)
	if err != nil {
		return nil, err
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	engine := &nativeEngine{}
	var result *C.char
	status := C.faceid_create(cpath, &engine.handle, &result)
	if result != nil {
		defer C.faceid_free(result)
	}
	if status != 0 || engine.handle == nil {
		if engine.handle != nil {
			_ = engine.Close()
		}
		return nil, errors.New("native model initialization failed")
	}
	var metadata *C.char
	if C.faceid_model(engine.handle, &metadata) != 0 || metadata == nil {
		_ = engine.Close()
		if metadata != nil {
			C.faceid_free(metadata)
		}
		return nil, errors.New("native model metadata unavailable")
	}
	err = json.Unmarshal([]byte(C.GoString(metadata)), &engine.model)
	C.faceid_free(metadata)
	if err != nil || engine.model.ModelID != manifest.ModelID || engine.model.ClassCount != len(engine.model.ClassNames) || engine.model.ClassCount < 2 || engine.model.Defaults.TopK < 1 || engine.model.Defaults.TopK > engine.model.ClassCount {
		_ = engine.Close()
		return nil, errors.New("invalid native model metadata")
	}
	hash := sha256.New()
	_, _ = hash.Write([]byte("native-v1\n"))
	_, _ = hash.Write(body)
	engine.model.RecognitionID = hex.EncodeToString(hash.Sum(nil))
	return engine, nil
}
func (e *nativeEngine) Model() httpapi.ModelInfo { return e.model }
func (e *nativeEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != nil {
		C.faceid_destroy(e.handle)
		e.handle = nil
	}
	return nil
}
func (e *nativeEngine) Process(data []byte, options httpapi.Options) (httpapi.CoreResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	var decoded httpapi.CoreResult
	if e.handle == nil {
		return decoded, &httpapi.APIError{Status: 503, Code: "model_unavailable", Message: "The model is not ready."}
	}
	if len(data) == 0 {
		return decoded, &httpapi.APIError{Status: 400, Code: "empty_file", Message: "The uploaded file is empty."}
	}
	mode := C.int(0)
	threshold := options.ProbabilityThreshold
	if options.CompactDetection {
		mode = 3
		threshold = options.ScoreThreshold
	} else if options.DetectionOnly {
		mode = 1
	}
	var result *C.char
	status := C.faceid_process_encoded(e.handle, (*C.uchar)(unsafe.Pointer(&data[0])), C.size_t(len(data)), mode, C.int(options.TopK), C.double(threshold), &result)
	runtime.KeepAlive(data)
	if result == nil {
		return decoded, errors.New("native inference returned no output")
	}
	defer C.faceid_free(result)
	body := []byte(C.GoString(result))
	if status != 0 {
		var failure struct {
			Error struct {
				Code string `json:"code"`
			}
		}
		_ = json.Unmarshal(body, &failure)
		// Only stable public error codes leave the C boundary; never expose C++ paths.
		switch failure.Error.Code {
		case "invalid_image":
			return decoded, &httpapi.APIError{Status: 400, Code: "invalid_image", Message: "Could not decode image."}
		case "image_too_large":
			return decoded, &httpapi.APIError{Status: 413, Code: "image_too_large", Message: "The image exceeds the pixel limit."}
		case "unsupported_image":
			return decoded, &httpapi.APIError{Status: 415, Code: "unsupported_image", Message: "Use a non-animated JPEG, PNG or WebP."}
		default:
			return decoded, errors.New("native inference failed")
		}
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return decoded, errors.New("invalid native inference output")
	}
	return decoded, nil
}
