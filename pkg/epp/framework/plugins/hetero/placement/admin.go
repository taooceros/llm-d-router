/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package placement

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

// Admin paths served on the plugin's adminAddress.
const (
	AdminStatePath = "/hetero/state"
	AdminResetPath = "/hetero/reset"
)

// State is the placement state an experiment run accumulates.
type State struct {
	FreshDispatched int `json:"fresh_dispatched"`
	FreshToLarge    int `json:"fresh_to_large"`
	// Reservations counts live reservations, Leases the unconfirmed ones.
	Reservations int                                 `json:"reservations"`
	Leases       int                                 `json:"leases"`
	Projection   map[string]capacity.ProjectionState `json:"projection"`
}

// State returns the quota counters and the reservation projection.
func (p *Plugin) State() State {
	perEndpoint, leases := p.tracker.State()
	dispatched, toLarge := p.selector.Counters()
	s := State{FreshDispatched: dispatched, FreshToLarge: toLarge, Leases: leases, Projection: perEndpoint}
	for _, row := range perEndpoint {
		s.Reservations += row.Requests
	}
	return s
}

// ErrReservationsLive is returned by Reset when requests still hold
// reservations and force is not set.
var ErrReservationsLive = errors.New("requests still hold reservations")

// Reset zeroes the quota counters and drops the projection, returning the
// state before the reset. Without force it refuses while any reservation is
// live, since dropping one makes the router overcommit its endpoint.
func (p *Plugin) Reset(force bool) (State, error) {
	p.bindMu.Lock()
	defer p.bindMu.Unlock()
	before := p.State()
	if before.Reservations > 0 && !force {
		return before, ErrReservationsLive
	}
	p.tracker.Reset()
	p.selector.Reset()
	return before, nil
}

// DumpState exposes State on the EPP plugin state debug endpoint.
func (p *Plugin) DumpState() (json.RawMessage, error) {
	return json.Marshal(p.State())
}

type resetResponse struct {
	Reset  bool   `json:"reset"`
	Error  string `json:"error,omitempty"`
	Before State  `json:"before"`
}

func (p *Plugin) adminHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET "+AdminStatePath, func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, p.State())
	})
	mux.HandleFunc("POST "+AdminResetPath, func(w http.ResponseWriter, r *http.Request) {
		before, err := p.Reset(r.URL.Query().Get("force") == "true")
		if err != nil {
			writeJSON(w, http.StatusConflict, resetResponse{Error: err.Error(), Before: before})
			return
		}
		writeJSON(w, http.StatusOK, resetResponse{Reset: true, Before: before})
	})
	return mux
}

func (p *Plugin) serveAdmin(ctx context.Context, address string) error {
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	srv := &http.Server{Handler: p.adminHandler(), ReadHeaderTimeout: 5 * time.Second}
	go func() { _ = srv.Serve(listener) }()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	return nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
