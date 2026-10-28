package httpapi

import (
	"bytes"
	"encoding/binary"
	"image"
	_ "image/jpeg"
	_ "image/png"
)

func validateImage(data []byte, maxPixels int64) error {
	if len(data) == 0 {
		return apiError(400, "empty_file", "The uploaded file is empty.")
	}
	invalid := func() error { return apiError(400, "invalid_image", "Could not decode image.") }
	unsupported := func() error { return apiError(415, "unsupported_image", "Use a non-animated JPEG, PNG or WebP.") }
	width, height := 0, 0
	switch {
	case bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")):
		if len(data) < 33 {
			return invalid()
		}
		for offset := 8; offset < len(data); {
			if len(data)-offset < 12 {
				return invalid()
			}
			size := int64(binary.BigEndian.Uint32(data[offset : offset+4]))
			if size > int64(len(data)-offset-12) {
				return invalid()
			}
			tag := string(data[offset+4 : offset+8])
			if tag == "acTL" {
				return unsupported()
			}
			offset += int(size) + 12
			if tag == "IEND" {
				break
			}
		}
		config, _, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil {
			return invalid()
		}
		width, height = config.Width, config.Height
	case len(data) >= 2 && data[0] == 0xff && data[1] == 0xd8:
		config, _, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil {
			return invalid()
		}
		width, height = config.Width, config.Height
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		var animated bool
		var ok bool
		width, height, animated, ok = webpDimensions(data)
		if animated {
			return unsupported()
		}
		if !ok {
			return invalid()
		}
	case bytes.HasPrefix(data, []byte("GIF87a")) || bytes.HasPrefix(data, []byte("GIF89a")) || bytes.HasPrefix(data, []byte("BM")) || bytes.HasPrefix(data, []byte("II*\x00")) || bytes.HasPrefix(data, []byte("MM\x00*")):
		return unsupported()
	default:
		return invalid()
	}
	if width < 1 || height < 1 {
		return invalid()
	}
	if int64(width) > maxPixels/int64(height) {
		return apiError(413, "image_too_large", "The image exceeds the pixel limit.")
	}
	return nil
}

// Dimensions are inspected without allocating the image. The native decoder
// remains responsible for validating compressed payloads.
func webpDimensions(data []byte) (width, height int, animated, ok bool) {
	total := int64(binary.LittleEndian.Uint32(data[4:8])) + 8
	if total < 12 || total > int64(len(data)) {
		return
	}
	seenExtended, seenBitstream := false, false
	for offset := int64(12); offset < total; {
		if total-offset < 8 {
			return 0, 0, false, false
		}
		size := int64(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		if size > total-offset-8 {
			return 0, 0, false, false
		}
		chunk := data[offset+8 : offset+8+size]
		switch string(data[offset : offset+4]) {
		case "VP8X":
			if size != 10 || offset != 12 || seenExtended || seenBitstream {
				return 0, 0, false, false
			}
			seenExtended = true
			if chunk[0]&2 != 0 {
				return 0, 0, true, false
			}
			width = 1 + int(chunk[4]) + int(chunk[5])<<8 + int(chunk[6])<<16
			height = 1 + int(chunk[7]) + int(chunk[8])<<8 + int(chunk[9])<<16
		case "ANIM", "ANMF":
			return 0, 0, true, false
		case "VP8 ":
			if seenBitstream || size < 10 || chunk[3] != 0x9d || chunk[4] != 1 || chunk[5] != 0x2a {
				return 0, 0, false, false
			}
			seenBitstream = true
			w, h := int(binary.LittleEndian.Uint16(chunk[6:8])&0x3fff), int(binary.LittleEndian.Uint16(chunk[8:10])&0x3fff)
			if width != 0 && (width != w || height != h) {
				return 0, 0, false, false
			}
			width, height = w, h
			ok = true
		case "VP8L":
			if seenBitstream || size < 5 || chunk[0] != 0x2f {
				return 0, 0, false, false
			}
			seenBitstream = true
			bits := binary.LittleEndian.Uint32(chunk[1:5])
			w, h := 1+int(bits&0x3fff), 1+int((bits>>14)&0x3fff)
			if width != 0 && (width != w || height != h) {
				return 0, 0, false, false
			}
			width, height = w, h
			ok = true
		}
		offset += 8 + size + (size & 1)
	}
	return
}
