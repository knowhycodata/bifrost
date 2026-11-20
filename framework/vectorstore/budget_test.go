package vectorstore

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// budgetTestNow is on the hour, so the hourly cutoff lands on the vectors.
var budgetTestNow = time.Date(2026, 11, 20, 12, 0, 0, 0, time.UTC)

// budgetTestIndexes stores two indexes in one Chromem store: "warp", four wide,
// with a vector every hour from 1 to 10 hours ago, and "capture", eight wide
// and left for the budget to measure, every other hour from 2 to 10. The warp
// namespace also holds an old vector that is not Warp's.
func budgetTestIndexes(t *testing.T) (VectorStore, []BudgetIndex) {
	t.Helper()
	setup := NewChromemTestSetup(t)
	t.Cleanup(func() { setup.Cleanup(t) })
	store, ctx := setup.Store, context.Background()
	properties := map[string]VectorStoreProperties{
		"warp_log":  {DataType: VectorStorePropertyTypeBoolean},
		"kind":      {DataType: VectorStorePropertyTypeString},
		"timestamp": {DataType: VectorStorePropertyTypeInteger},
	}
	require.NoError(t, store.CreateNamespace(ctx, "warp", 4, properties))
	require.NoError(t, store.CreateNamespace(ctx, "capture", 8, properties))
	at := func(hours int) int64 { return budgetTestNow.Add(-time.Duration(hours) * time.Hour).Unix() }
	for hour := 1; hour <= 10; hour++ {
		require.NoError(t, store.Add(ctx, "warp", fmt.Sprintf("warp-%02d", hour), []float32{1, 0, 0, 0}, map[string]interface{}{"warp_log": true, "timestamp": at(hour)}))
		if hour%2 == 0 {
			require.NoError(t, store.Add(ctx, "capture", fmt.Sprintf("capture-%02d", hour), []float32{1, 0, 0, 0, 0, 0, 0, 0}, map[string]interface{}{"kind": "request", "timestamp": at(hour)}))
		}
	}
	require.NoError(t, store.Add(ctx, "warp", "not-warps", []float32{0, 1, 0, 0}, map[string]interface{}{"timestamp": at(100)}))
	return store, []BudgetIndex{
		{Name: "warp", Store: store, Namespace: "warp", Dimension: 4, TimestampField: "timestamp",
			Filter: []Query{{Field: "warp_log", Operator: QueryOperatorEqual, Value: true}}},
		{Name: "capture", Store: store, Namespace: "capture", TimestampField: "timestamp",
			Filter: []Query{{Field: "kind", Operator: QueryOperatorEqual, Value: "request"}}},
	}
}

func budgetTestIDs(t *testing.T, store VectorStore, namespace string) []string {
	t.Helper()
	results, _, err := store.GetAll(context.Background(), namespace, nil, nil, nil, 100)
	require.NoError(t, err)
	ids := make([]string, 0, len(results))
	for _, result := range results {
		ids = append(ids, result.ID)
	}
	return ids
}

// The estimate is 10 warp vectors of 4*4+1024 bytes and 5 captures of
// 8*4+1024, the capture width read off a stored vector: 15,680 bytes, which
// is exactly 80% of 19,600, so nothing is trimmed yet.
func TestEnforceBudgetLeavesIndexesUnderTheHighWatermark(t *testing.T) {
	store, indexes := budgetTestIndexes(t)
	result, err := EnforceBudget(context.Background(), 19600, indexes, budgetTestNow)
	require.NoError(t, err)
	assert.EqualValues(t, 15680, result.EstimatedBytes)
	assert.Zero(t, result.Deleted)
	assert.True(t, result.Cutoff.IsZero())
	assert.Len(t, budgetTestIDs(t, store, "warp"), 11)
}

// Past 80% of the budget, one cutoff applies to every index: 65% of 10,000
// bytes keeps the last four hours (4 warp vectors and 2 captures, 6,272
// bytes; five hours would be 7,312), and everything older goes from both,
// though 15,680 bytes was still under the budget itself. What is not an
// index's own stays.
func TestEnforceBudgetDeletesTheOldestAcrossIndexes(t *testing.T) {
	store, indexes := budgetTestIndexes(t)
	result, err := EnforceBudget(context.Background(), 10000, indexes, budgetTestNow)
	require.NoError(t, err)
	assert.Equal(t, budgetTestNow.Add(-4*time.Hour), result.Cutoff)
	assert.Equal(t, 9, result.Deleted)
	assert.ElementsMatch(t, []string{"warp-01", "warp-02", "warp-03", "warp-04", "not-warps"}, budgetTestIDs(t, store, "warp"))
	assert.ElementsMatch(t, []string{"capture-02", "capture-04"}, budgetTestIDs(t, store, "capture"))

	again, err := EnforceBudget(context.Background(), 10000, indexes, budgetTestNow)
	require.NoError(t, err)
	assert.Zero(t, again.Deleted, "once under the high watermark, the next pass leaves it")
}

// A store that cannot count leaves the size unknown: nothing is trimmed, from
// that index or any other, and the pass says why.
func TestEnforceBudgetTrimsNothingWhenAStoreCannotCount(t *testing.T) {
	store, indexes := budgetTestIndexes(t)
	indexes[1].Store = struct{ VectorStore }{store}
	_, err := EnforceBudget(context.Background(), 1000, indexes, budgetTestNow)
	require.ErrorIs(t, err, ErrCountUnsupported)
	assert.Len(t, budgetTestIDs(t, store, "warp"), 11)
	assert.Len(t, budgetTestIDs(t, store, "capture"), 5)
}

// No budget, or nothing under it, is no pass at all.
func TestEnforceBudgetWithoutABudget(t *testing.T) {
	store, indexes := budgetTestIndexes(t)
	result, err := EnforceBudget(context.Background(), 0, indexes, budgetTestNow)
	require.NoError(t, err)
	assert.Zero(t, result)
	assert.Len(t, budgetTestIDs(t, store, "warp"), 11)
}

// The budget on every backend: ten hourly vectors in one index, a budget whose
// 65% fits five, and the five newest are what is left. It leans on each
// store's filtered count agreeing with its filtered delete.
func TestEnforceBudgetOnEveryBackend(t *testing.T) {
	for _, backend := range vectorStoreTestBackends() {
		t.Run(backend.name, func(t *testing.T) {
			store, ctx, shared := backend.setup(t)
			namespace := shared + "Budget"
			require.NoError(t, store.CreateNamespace(ctx, namespace, backend.dimension, map[string]VectorStoreProperties{
				"warp_log":  {DataType: VectorStorePropertyTypeBoolean},
				"timestamp": {DataType: VectorStorePropertyTypeInteger},
			}))
			t.Cleanup(func() { _ = store.DeleteNamespace(ctx, namespace) })
			ids := map[int]string{}
			for hour := 1; hour <= 10; hour++ {
				ids[hour] = generateUUID()
				require.NoError(t, store.Add(ctx, namespace, ids[hour], generateTestEmbedding(backend.dimension), map[string]interface{}{
					"warp_log": true, "timestamp": budgetTestNow.Add(-time.Duration(hour) * time.Hour).Unix(),
				}))
			}
			index := BudgetIndex{Name: backend.name, Store: store, Namespace: namespace, Dimension: backend.dimension, TimestampField: "timestamp",
				Filter: []Query{{Field: "warp_log", Operator: QueryOperatorEqual, Value: true}}}
			require.Eventually(t, func() bool {
				n, err := Count(ctx, store, namespace, index.Filter)
				return err == nil && n == 10
			}, 10*time.Second, 200*time.Millisecond)

			perVector := float64(backend.dimension*4 + BudgetBytesPerVectorOverhead)
			result, err := EnforceBudget(ctx, int64(5.5*perVector/BudgetLowWatermark), []BudgetIndex{index}, budgetTestNow)
			require.NoError(t, err)
			assert.Equal(t, budgetTestNow.Add(-5*time.Hour), result.Cutoff)
			require.Eventually(t, func() bool {
				n, err := Count(ctx, store, namespace, index.Filter)
				return err == nil && n == 5
			}, 10*time.Second, 200*time.Millisecond, "the five newest are left")
			for hour := 1; hour <= 5; hour++ {
				_, err := store.GetChunk(ctx, namespace, ids[hour])
				assert.NoError(t, err, "hour %d is kept", hour)
			}
		})
	}
}
