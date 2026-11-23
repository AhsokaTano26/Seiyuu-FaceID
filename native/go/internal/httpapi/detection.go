package httpapi

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/netip"
	"os"
	"strconv"
	"time"
)

const detectPath = "/api/v1/detect"
const timeoutBody = `{"error":{"code":"detection_timeout","message":"The detection request timed out."}}`

type detectionPolicy struct {
	timeout   time.Duration
	maxPixels int64
	clients   netip.Prefix
}

// LimitsFromEnv resolves overrides for the internal route only.
func LimitsFromEnv() (Limits, error) {
	limits := DefaultLimits()
	integer := func(name string, fallback, max int64) (int64, error) {
		raw := os.Getenv(name)
		if raw == "" {
			return fallback, nil
		}
		n, err := strconv.ParseUint(raw, 10, 63)
		if err != nil || n < 1 || n > uint64(max) {
			return 0, fmt.Errorf("%s must be an integer between 1 and %d", name, max)
		}
		return int64(n), nil
	}
	ms, err := integer("SEIYUU_FACEID_DETECT_TIMEOUT_MS", 5000, 1<<31-1)
	if err != nil {
		return limits, err
	}
	pixels, err := integer("SEIYUU_FACEID_DETECT_MAX_PIXELS", limits.MaxImagePixels, limits.MaxImagePixels)
	if err != nil {
		return limits, err
	}
	policy := detectionPolicy{timeout: time.Duration(ms) * time.Millisecond, maxPixels: pixels}
	if cidr := os.Getenv("SEIYUU_FACEID_DETECT_CLIENT_CIDR"); cidr != "" {
		network, err := netip.ParsePrefix(cidr)
		if err != nil || network != network.Masked() || !privateSubnet(network) {
			return limits, errors.New("SEIYUU_FACEID_DETECT_CLIENT_CIDR must be a private subnet")
		}
		policy.clients = network
	}
	limits.detection = policy
	return limits, nil
}

func privateSubnet(network netip.Prefix) bool {
	for _, cidr := range []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"} {
		prefix := netip.MustParsePrefix(cidr)
		if prefix.Contains(network.Addr()) && network.Bits() >= prefix.Bits() {
			return true
		}
	}
	return false
}

type detectionWriter struct {
	http.ResponseWriter
	status int
}

func (w *detectionWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != detectPath {
		s.serve(w, r)
		return
	}
	peer, err := netip.ParseAddrPort(r.RemoteAddr)
	addr := peer.Addr().Unmap()
	if err != nil || !(addr.IsLoopback() || s.limits.detection.clients.Contains(addr)) {
		writeError(w, apiError(403, "internal_only", "This endpoint requires a local connection."))
		return
	}
	started := time.Now()
	// net/http may drain a partially uploaded body before it flushes a response.
	// Interrupt socket reads at the same deadline as queueing and inference.
	_ = http.NewResponseController(w).SetReadDeadline(started.Add(s.limits.detection.timeout))
	out := &detectionWriter{ResponseWriter: w}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	s.timedDetect.ServeHTTP(out, r)
	log.Printf("Face detection status=%d elapsed_ms=%.1f", out.status, float64(time.Since(started).Microseconds())/1000)
}
