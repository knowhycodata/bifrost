package warp

import (
	"context"
	"errors"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"gorm.io/gorm"
)

// fakeIndexLedger is the governance_config key-value rows, in memory.
type fakeIndexLedger struct{ rows map[string]string }

func newFakeIndexLedger() *fakeIndexLedger { return &fakeIndexLedger{rows: map[string]string{}} }

func (l *fakeIndexLedger) GetConfig(_ context.Context, key string) (*tables.TableGovernanceConfig, error) {
	value, ok := l.rows[key]
	if !ok {
		return nil, configstore.ErrNotFound
	}
	return &tables.TableGovernanceConfig{Key: key, Value: value}, nil
}

func (l *fakeIndexLedger) UpdateConfig(_ context.Context, config *tables.TableGovernanceConfig, _ ...*gorm.DB) error {
	l.rows[config.Key] = config.Value
	return nil
}

// startWarp is one Bifrost startup: a service on the given store, reconciled
// against what the ledger remembers from the last one.
func startWarp(t *testing.T, ledger *fakeIndexLedger, shared *fakeWarpVectorStore, source VectorStoreSource) *Service {
	t.Helper()
	vectors := shared
	if source.Dedicated {
		vectors = newFakeWarpVectorStore()
	}
	opts := []Option{WithConfigStore(&recordingStore{row: validWarpConfigRow()}), WithVectorStoreSource(source), WithIndexStoreLedger(ledger, shared)}
	if !source.Failed {
		opts = append(opts, WithVectorStore(vectors))
	}
	service := NewService(nil, opts...)
	service.ReconcileIndexStore(context.Background())
	return service
}

var (
	qdrantSource = VectorStoreSource{Dedicated: true, Type: "qdrant", Fingerprint: "aaaa"}
	redisSource  = VectorStoreSource{Dedicated: true, Type: "redis", Fingerprint: "bbbb"}
)

func TestWarpReportsTheIndexLeftInTheSharedStoreAndDeletesIt(t *testing.T) {
	ledger := newFakeIndexLedger()
	shared := newFakeWarpVectorStore()
	shared.existing = []string{schemas.WarpDefaultLogVectorStoreNamespace}

	if left := startWarp(t, ledger, shared, VectorStoreSource{}).LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("on the shared store nothing is left behind, got %+v", left)
	}
	service := startWarp(t, ledger, shared, qdrantSource)
	left := service.LeftBehindIndexes()
	if len(left) != 1 || left[0].Dedicated || !left[0].Deletable || left[0].Namespace != schemas.WarpDefaultLogVectorStoreNamespace {
		t.Fatalf("moving to Warp's own store should leave the shared index, deletable: %+v", left)
	}
	view, err := service.ConfigView(context.Background())
	if err != nil || len(view.LeftBehindIndexes) != 1 {
		t.Fatalf("the settings view should carry the left-behind index: %+v, %v", view.LeftBehindIndexes, err)
	}

	if err := service.DeleteLeftBehindIndex(context.Background(), left[0].ID); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if len(shared.deleted) != 1 || shared.deleted[0] != schemas.WarpDefaultLogVectorStoreNamespace {
		t.Fatalf("the shared store should lose Warp's namespace, deleted %v", shared.deleted)
	}
	if left := service.LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("a deleted index leaves the list, got %+v", left)
	}
	if left := startWarp(t, ledger, shared, qdrantSource).LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("the delete should be remembered across a restart, got %+v", left)
	}
}

func TestWarpReportsNothingWhenTheSharedStoreNeverHeldAnIndex(t *testing.T) {
	// Warp was never set up before the move, so the shared store has no
	// namespace to point at; a first startup with no record counts as shared.
	ledger := newFakeIndexLedger()
	if left := startWarp(t, ledger, newFakeWarpVectorStore(), qdrantSource).LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("no namespace in the shared store means nothing left behind, got %+v", left)
	}
}

func TestWarpUpgradeStraightToItsOwnStoreFindsTheSharedIndex(t *testing.T) {
	// The release that adds warp.vector_store has no record yet: a deployment
	// that sets it on its first start indexed into the shared store until now.
	shared := newFakeWarpVectorStore()
	shared.existing = []string{schemas.WarpDefaultLogVectorStoreNamespace}
	left := startWarp(t, newFakeIndexLedger(), shared, qdrantSource).LeftBehindIndexes()
	if len(left) != 1 || !left[0].Deletable {
		t.Fatalf("the shared index from before the upgrade should be reported, got %+v", left)
	}
}

func TestWarpCannotDeleteAnIndexInAStoreItNoLongerReaches(t *testing.T) {
	ledger := newFakeIndexLedger()
	shared := newFakeWarpVectorStore()
	startWarp(t, ledger, shared, qdrantSource)
	service := startWarp(t, ledger, shared, VectorStoreSource{})
	left := service.LeftBehindIndexes()
	if len(left) != 1 || !left[0].Dedicated || left[0].Deletable || left[0].Type != "qdrant" {
		t.Fatalf("moving off Warp's own store should leave it, not deletable: %+v", left)
	}
	if err := service.DeleteLeftBehindIndex(context.Background(), left[0].ID); !errors.Is(err, ErrLeftBehindIndexUnreachable) {
		t.Fatalf("deleting from a store Bifrost does not reach should be refused, got %v", err)
	}
	if len(shared.deleted) != 0 {
		t.Fatalf("nothing may be deleted from the shared store, deleted %v", shared.deleted)
	}
	if err := service.DismissLeftBehindIndex(context.Background(), left[0].ID); err != nil {
		t.Fatalf("dismiss: %v", err)
	}
	if left := startWarp(t, ledger, shared, VectorStoreSource{}).LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("a dismissed index stays off the list, got %+v", left)
	}
	if err := service.DismissLeftBehindIndex(context.Background(), "nope"); !errors.Is(err, ErrLeftBehindIndexNotFound) {
		t.Fatalf("an unknown id should be not found, got %v", err)
	}
}

func TestWarpReturningToAStoreTakesItOffTheList(t *testing.T) {
	ledger := newFakeIndexLedger()
	shared := newFakeWarpVectorStore()
	shared.existing = []string{schemas.WarpDefaultLogVectorStoreNamespace}
	startWarp(t, ledger, shared, qdrantSource)
	startWarp(t, ledger, shared, redisSource)
	left := startWarp(t, ledger, shared, VectorStoreSource{}).LeftBehindIndexes()
	if len(left) != 2 {
		t.Fatalf("back on the shared store, the qdrant and redis indexes are left, got %+v", left)
	}
	for _, entry := range left {
		if !entry.Dedicated {
			t.Fatalf("the shared index is live again and must not be listed: %+v", left)
		}
	}
}

func TestWarpRecordsNothingWhileItsOwnStoreIsDown(t *testing.T) {
	ledger := newFakeIndexLedger()
	shared := newFakeWarpVectorStore()
	startWarp(t, ledger, shared, qdrantSource)
	failed := qdrantSource
	failed.Fingerprint = "cccc"
	failed.Failed = true
	startWarp(t, ledger, shared, failed)
	// The index never moved: the next good startup on qdrant has nothing left.
	if left := startWarp(t, ledger, shared, qdrantSource).LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("a failed startup must not count as a move, got %+v", left)
	}
}

func TestWarpClearsASharedIndexDeletedOutsideBifrost(t *testing.T) {
	ledger := newFakeIndexLedger()
	shared := newFakeWarpVectorStore()
	shared.existing = []string{schemas.WarpDefaultLogVectorStoreNamespace}
	service := startWarp(t, ledger, shared, qdrantSource)
	if len(service.LeftBehindIndexes()) != 1 {
		t.Fatalf("precondition: the shared index is listed")
	}

	shared.listErr = errors.New("store down")
	service.RefreshLeftBehindIndexes(context.Background())
	if len(service.LeftBehindIndexes()) != 1 {
		t.Fatalf("a store that cannot be asked keeps the entry")
	}

	// Deleted by hand, or by another replica's delete.
	shared.listErr = nil
	shared.existing = nil
	service.RefreshLeftBehindIndexes(context.Background())
	if left := service.LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("a namespace that is gone clears its notice, got %+v", left)
	}
	if left := startWarp(t, ledger, shared, qdrantSource).LeftBehindIndexes(); len(left) != 0 {
		t.Fatalf("the clear should be stored, got %+v", left)
	}
}
