package httpapi

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"
)

func riff(chunks ...[]byte) []byte {
	var body bytes.Buffer
	body.WriteString("WEBP")
	for _, c := range chunks {
		body.Write(c)
	}
	var data bytes.Buffer
	data.WriteString("RIFF")
	_ = binary.Write(&data, binary.LittleEndian, uint32(body.Len()))
	data.Write(body.Bytes())
	return data.Bytes()
}
func chunk(tag string, data []byte) []byte {
	var b bytes.Buffer
	b.WriteString(tag)
	_ = binary.Write(&b, binary.LittleEndian, uint32(len(data)))
	b.Write(data)
	if len(data)%2 != 0 {
		b.WriteByte(0)
	}
	return b.Bytes()
}
func TestWebPDimensionsAndPixelLimit(t *testing.T) {
	payload := make([]byte, 5)
	payload[0] = 0x2f
	binary.LittleEndian.PutUint32(payload[1:], uint32(1|(2<<14)))
	data := riff(chunk("VP8L", payload))
	w, h, animated, ok := webpDimensions(data)
	if w != 2 || h != 3 || animated || !ok {
		t.Fatal(w, h, animated, ok)
	}
	if err := validateImage(data, 6); err != nil {
		t.Fatal(err)
	}
	var e *APIError
	if err := validateImage(data, 5); !errors.As(err, &e) || e.Code != "image_too_large" {
		t.Fatal(err)
	}
}
func TestRejectsAnimatedWebPAndTruncatedChunks(t *testing.T) {
	x := make([]byte, 10)
	x[0] = 2
	data := riff(chunk("VP8X", x))
	var e *APIError
	if err := validateImage(data, 100); !errors.As(err, &e) || e.Code != "unsupported_image" {
		t.Fatal(err)
	}
	if err := validateImage(data[:len(data)-1], 100); !errors.As(err, &e) || e.Code != "invalid_image" {
		t.Fatal(err)
	}
}
func TestRejectsAnimatedPNG(t *testing.T) {
	data := pngData()
	data = append(append(append([]byte{}, data[:33]...), []byte{0, 0, 0, 8, 'a', 'c', 'T', 'L', 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}...), data[33:]...)
	var e *APIError
	if err := validateImage(data, 100); !errors.As(err, &e) || e.Code != "unsupported_image" {
		t.Fatal(err)
	}
}

func TestWebPRejectsConflictingAndDuplicateDimensionChunks(t *testing.T) {
	payload := make([]byte, 5)
	payload[0] = 0x2f
	binary.LittleEndian.PutUint32(payload[1:], uint32(100|(100<<14)))
	small := make([]byte, 10)
	for _, data := range [][]byte{riff(chunk("VP8L", payload), chunk("VP8X", small)), riff(chunk("VP8X", small), chunk("VP8X", small)), riff(chunk("VP8L", payload), chunk("VP8L", payload))} {
		if _, _, _, ok := webpDimensions(data); ok {
			t.Fatal("accepted conflicting/duplicate WebP header")
		}
	}
}
