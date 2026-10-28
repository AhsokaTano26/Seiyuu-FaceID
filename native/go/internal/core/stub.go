//go:build !native || !cgo

package core

import (
	"errors"
	"seiyuu-faceid/native/internal/httpapi"
)

func Open(path string) (httpapi.Backend, error) {
	return nil, errors.New("native inference requires CGO_ENABLED=1 and go build -tags native; see native/README.md")
}
