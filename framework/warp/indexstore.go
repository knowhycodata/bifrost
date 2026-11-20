package warp

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/vectorstore"
	"gorm.io/gorm"
)

// When warp.vector_store is added, changed or removed, Warp's log index starts
// over in the new store and the old one keeps its vectors: nothing in Bifrost
// deletes them on the way out, and once Bifrost stops connecting to a store
// the retention sweep cannot reach it either. So at each startup Warp compares
// where its index is now with where it was at the last one, and keeps a list
// of indexes it has left behind for the settings page to show. An index left
// in the shared store, which Bifrost still connects to, can be deleted from
// there; one left in a store of Warp's own can only be described, for the
// admin to delete by hand, and dismissed.

// indexStoresConfigKey is the governance_config row holding the record. It is a
// key of its own, so a binary that predates it never reads or rewrites it.
const indexStoresConfigKey = "warp_index_stores"

// ErrLeftBehindIndexNotFound is a left-behind index id the record does not hold.
var ErrLeftBehindIndexNotFound = errors.New("no Warp index left behind with that id")

// ErrLeftBehindIndexUnreachable is a delete of an index in a store Bifrost no
// longer connects to.
var ErrLeftBehindIndexUnreachable = errors.New("that Warp index is in a store Bifrost no longer connects to; delete it there by hand")

// IndexStoreLedger is the config store's key-value rows, where the record of
// Warp's index locations lives.
type IndexStoreLedger interface {
	GetConfig(ctx context.Context, key string) (*tables.TableGovernanceConfig, error)
	UpdateConfig(ctx context.Context, config *tables.TableGovernanceConfig, tx ...*gorm.DB) error
}

// IndexLocation is a store and namespace Warp's log index lives in.
type IndexLocation struct {
	Dedicated   bool   `json:"dedicated"`             // a store of Warp's own, not the shared vector_store
	Type        string `json:"type,omitempty"`        // the dedicated store's type
	Fingerprint string `json:"fingerprint,omitempty"` // the dedicated store's fingerprint; empty for the shared store
	Namespace   string `json:"namespace"`
}

// sameStore reports whether two locations are the same store, whatever the
// namespace: a namespace change inside one store is SaveConfig's to clean up.
func (l IndexLocation) sameStore(other IndexLocation) bool {
	return l.Dedicated == other.Dedicated && l.Fingerprint == other.Fingerprint
}

// LeftBehindIndex is a Warp log index in a store Warp no longer uses.
type LeftBehindIndex struct {
	ID string `json:"id"`
	IndexLocation
	MovedAt time.Time `json:"moved_at"`
	// Deletable is whether Bifrost can delete it: true only for the shared store,
	// which Bifrost still connects to.
	Deletable bool `json:"deletable"`
}

type indexStoreRecord struct {
	Current    *IndexLocation    `json:"current,omitempty"`
	LeftBehind []LeftBehindIndex `json:"left_behind,omitempty"`
}

// WithIndexStoreLedger keeps the record of where Warp's index has lived, and
// gives Warp the shared vector store so an index left there can be deleted.
// Without it nothing is recorded and no left-behind index is reported.
func WithIndexStoreLedger(ledger IndexStoreLedger, shared vectorstore.VectorStore) Option {
	return func(s *Service) {
		s.indexLedger = ledger
		s.sharedVectorStore = shared
	}
}

// leftBehindIndexID names an entry by its store and namespace, so the same
// index left twice is one entry.
func leftBehindIndexID(location IndexLocation) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%t|%s|%s", location.Dedicated, location.Fingerprint, location.Namespace)))
	return fmt.Sprintf("%x", sum[:8])
}

// ReconcileIndexStore runs once at startup, after config.json's warp section
// is applied: it records where the index lives now and, when that is a
// different store from the last startup, adds the old one to the left-behind
// list. A dedicated store that failed to connect records nothing, since the
// index is not anywhere yet. A record that cannot be read or written is
// logged and leaves the list as it was: the banner is a reminder, and Warp
// itself does not depend on it.
func (s *Service) ReconcileIndexStore(ctx context.Context) {
	if s.indexLedger == nil || s.vectorSource.Failed || s.vectorStore == nil {
		return
	}
	view, err := s.ConfigView(ctx)
	if err != nil {
		return
	}
	current := IndexLocation{
		Dedicated:   s.vectorSource.Dedicated,
		Type:        s.vectorSource.Type,
		Fingerprint: s.vectorSource.Fingerprint,
		Namespace:   normalizedNamespace(view.LogVectorStoreNamespace),
	}
	record, err := s.readIndexStoreRecord(ctx)
	if err != nil {
		s.warnf("failed to read where Warp's log index lived before: %v", err)
		return
	}
	// Before the record existed, every Warp index was in the shared store.
	previous := IndexLocation{Namespace: current.Namespace}
	if record.Current != nil {
		previous = *record.Current
	}
	// Back in a store it left: that index is live again, never one to delete.
	record.LeftBehind = slices.DeleteFunc(record.LeftBehind, func(entry LeftBehindIndex) bool {
		return entry.sameStore(current)
	})
	if !previous.sameStore(current) && s.indexWasLeftIn(ctx, previous) {
		left := LeftBehindIndex{ID: leftBehindIndexID(previous), IndexLocation: previous, MovedAt: time.Now().UTC(), Deletable: !previous.Dedicated}
		record.LeftBehind = slices.DeleteFunc(record.LeftBehind, func(entry LeftBehindIndex) bool { return entry.ID == left.ID })
		record.LeftBehind = append(record.LeftBehind, left)
	}
	record.Current = &current
	if err := s.writeIndexStoreRecord(ctx, record); err != nil {
		s.warnf("failed to record where Warp's log index lives: %v", err)
		return
	}
	s.mu.Lock()
	s.leftBehind = record.LeftBehind
	s.mu.Unlock()
}

// indexWasLeftIn reports whether the previous location may hold vectors. Only
// the shared store can be checked, being the one still connected: there, an
// index that was never built (Warp never configured, or no namespace yet) is
// no index at all. A dedicated store cannot be asked, so it counts.
func (s *Service) indexWasLeftIn(ctx context.Context, previous IndexLocation) bool {
	if previous.Dedicated {
		return true
	}
	if s.sharedVectorStore == nil {
		return false
	}
	namespaces, err := s.sharedVectorStore.ListNamespaces(ctx, previous.Namespace)
	if err != nil {
		return true
	}
	return slices.Contains(namespaces, previous.Namespace)
}

// RefreshLeftBehindIndexes drops entries in the shared store whose namespace
// is gone, deleted by hand or from another replica, so their notice clears
// itself. It asks the shared store, so it runs for the settings page only, not
// for every read of the configuration (the tray polls ConfigView), and only
// when there is a shared entry. A store that cannot be asked keeps the entry.
func (s *Service) RefreshLeftBehindIndexes(ctx context.Context) {
	if s.sharedVectorStore == nil {
		return
	}
	for _, entry := range s.LeftBehindIndexes() {
		if entry.Dedicated {
			continue
		}
		namespaces, err := s.sharedVectorStore.ListNamespaces(ctx, entry.Namespace)
		if err != nil || slices.Contains(namespaces, entry.Namespace) {
			continue
		}
		if err := s.forgetLeftBehindIndex(ctx, entry.ID); err != nil {
			s.warnf("failed to clear Warp's old log index %q, which is no longer in the shared vector store: %v", entry.Namespace, err)
		}
	}
}

// LeftBehindIndexes is the list for the settings page.
func (s *Service) LeftBehindIndexes() []LeftBehindIndex {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.leftBehind)
}

// DeleteLeftBehindIndex deletes a left-behind index from the shared store and
// takes it off the list. An index in a store Bifrost no longer connects to is
// refused with ErrLeftBehindIndexUnreachable.
func (s *Service) DeleteLeftBehindIndex(ctx context.Context, id string) error {
	entry, err := s.leftBehindIndex(id)
	if err != nil {
		return err
	}
	if entry.Dedicated || s.sharedVectorStore == nil {
		return ErrLeftBehindIndexUnreachable
	}
	// Never the store Warp is writing to: with Warp back on the shared store,
	// ReconcileIndexStore drops such an entry, and this holds even if the record
	// was edited by hand.
	if !s.vectorSource.Dedicated {
		return s.forgetLeftBehindIndex(ctx, id)
	}
	if err := s.sharedVectorStore.DeleteNamespace(ctx, entry.Namespace); err != nil && !errors.Is(err, vectorstore.ErrNotFound) {
		return fmt.Errorf("delete Warp's old log index %q from the shared vector store: %w", entry.Namespace, err)
	}
	return s.forgetLeftBehindIndex(ctx, id)
}

// DismissLeftBehindIndex takes an index off the list without deleting it, for
// an admin who has deleted it by hand or wants to keep it.
func (s *Service) DismissLeftBehindIndex(ctx context.Context, id string) error {
	if _, err := s.leftBehindIndex(id); err != nil {
		return err
	}
	return s.forgetLeftBehindIndex(ctx, id)
}

func (s *Service) leftBehindIndex(id string) (LeftBehindIndex, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, entry := range s.leftBehind {
		if entry.ID == id {
			return entry, nil
		}
	}
	return LeftBehindIndex{}, ErrLeftBehindIndexNotFound
}

// forgetLeftBehindIndex removes an entry from the stored record and from the
// list, rereading the record so it never writes back a stale Current.
func (s *Service) forgetLeftBehindIndex(ctx context.Context, id string) error {
	if s.indexLedger == nil {
		return ErrLeftBehindIndexNotFound
	}
	record, err := s.readIndexStoreRecord(ctx)
	if err != nil {
		return err
	}
	record.LeftBehind = slices.DeleteFunc(record.LeftBehind, func(entry LeftBehindIndex) bool { return entry.ID == id })
	if err := s.writeIndexStoreRecord(ctx, record); err != nil {
		return err
	}
	s.mu.Lock()
	s.leftBehind = record.LeftBehind
	s.mu.Unlock()
	return nil
}

func (s *Service) readIndexStoreRecord(ctx context.Context) (indexStoreRecord, error) {
	var record indexStoreRecord
	row, err := s.indexLedger.GetConfig(ctx, indexStoresConfigKey)
	if errors.Is(err, configstore.ErrNotFound) || (err == nil && (row == nil || row.Value == "")) {
		return record, nil
	}
	if err != nil {
		return record, err
	}
	if err := sonic.Unmarshal([]byte(row.Value), &record); err != nil {
		return indexStoreRecord{}, fmt.Errorf("parse %s: %w", indexStoresConfigKey, err)
	}
	return record, nil
}

func (s *Service) writeIndexStoreRecord(ctx context.Context, record indexStoreRecord) error {
	value, err := sonic.Marshal(record)
	if err != nil {
		return err
	}
	return s.indexLedger.UpdateConfig(ctx, &tables.TableGovernanceConfig{Key: indexStoresConfigKey, Value: string(value)})
}
