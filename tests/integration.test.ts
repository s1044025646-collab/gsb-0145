import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDb, rating, T0 } from './helpers.ts';
import { importRatings } from '../src/db/repo.ts';
import { trainFromCurrent, evaluateAndStore } from '../src/model/service.ts';
import { loadModel } from '../src/model/serialize.ts';
import { topK } from '../src/model/recommend.ts';
import { DEFAULT_CONFIG } from '../src/model/types.ts';
import { ndcgAtK } from '../src/model/evaluate.ts';
import { loadSplit } from '../src/model/split.ts';
import { closeDatabase, openDatabase } from '../src/db/database.ts';

function seedDecomposable(db: ReturnType<typeof makeDb>['db']): void {
  // Two latent groups with a clean additive pattern.
  const events = [];
  for (let u = 0; u < 6; u++) {
    for (let i = 0; i < 8; i++) {
      const groupMatch = Math.floor(u / 3) === Math.floor(i / 4);
      const r = groupMatch ? 5 : 1;
      events.push(rating(`u${u}`, `i${i}`, r, T0 + u * 100 + i, `s-${u}-${i}`));
    }
  }
  importRatings(db, events);
}

test('train -> evaluate produces finite RMSE and NDCG', () => {
  const { db, close } = makeDb();
  seedDecomposable(db);
  const { modelId } = trainFromCurrent(db, {
    ...DEFAULT_CONFIG,
    factors: 2,
    epochs: 80,
    learningRate: 0.02,
    regBias: 0.01,
    regFactor: 0.01,
    valFraction: 0.25,
  });
  const report = evaluateAndStore(db, modelId, 5);
  assert.equal(typeof report.rmse, 'number');
  assert.ok(Number.isFinite(report.rmse!));
  assert.ok(report.ndcgAtK === null || (report.ndcgAtK >= 0 && report.ndcgAtK <= 1));
  close();
});

test('training twice with identical seed yields identical epoch losses', () => {
  const { db, close } = makeDb();
  seedDecomposable(db);
  const a = trainFromCurrent(db, { ...DEFAULT_CONFIG, factors: 2, epochs: 20, seed: 99 });
  const b = trainFromCurrent(db, { ...DEFAULT_CONFIG, factors: 2, epochs: 20, seed: 99 });
  assert.deepEqual(a.epochLosses, b.epochLosses);
});

test('old model predictions stay identical after training a newer model', () => {
  const { db, close } = makeDb();
  seedDecomposable(db);
  const old = trainFromCurrent(db, { ...DEFAULT_CONFIG, factors: 2, epochs: 10, seed: 3 });
  const before = loadModel(db, old.modelId);
  const pBefore = JSON.stringify({
    ub: [...before.data.userBias.entries()],
    uf: [...before.data.userFactors.entries()],
    ib: [...before.data.itemBias.entries()],
  });
  trainFromCurrent(db, { ...DEFAULT_CONFIG, factors: 4, epochs: 5, seed: 123 });
  const after = loadModel(db, old.modelId);
  const pAfter = JSON.stringify({
    ub: [...after.data.userBias.entries()],
    uf: [...after.data.userFactors.entries()],
    ib: [...after.data.itemBias.entries()],
  });
  assert.equal(pBefore, pAfter);
  close();
});

test('persistence survives close and reopen', () => {
  const { db, path } = makeDb();
  seedDecomposable(db);
  const { modelId } = trainFromCurrent(db, { ...DEFAULT_CONFIG, epochs: 10 });
  const reopened = openDatabase(path);
  const model = loadModel(reopened.db, modelId);
  assert.equal(model.config.epochs, 10);
  assert.equal(model.data.stats.epochLosses.length, 10);
  closeDatabase(reopened.db);
});

test('recommendations exclude seen items and use fixed tie-break by itemId', () => {
  const { db, close } = makeDb();
  seedDecomposable(db);
  const { modelId } = trainFromCurrent(db, { ...DEFAULT_CONFIG, epochs: 30 });
  const model = loadModel(db, modelId);
  const seen = new Set(model.train.filter((r) => r.user_id === 'u0').map((r) => r.item_id));
  const recs = topK(model, 'u0', { k: 10, currentlySeen: seen, exclude: 'train' });
  assert.ok(recs.every((r) => !seen.has(r.itemId)));
  // Sorted descending; ties ascending itemId.
  for (let i = 1; i < recs.length; i++) {
    assert.ok(
      recs[i - 1]!.score > recs[i]!.score ||
        (recs[i - 1]!.score === recs[i]!.score &&
          recs[i - 1]!.itemId < recs[i]!.itemId),
    );
  }
  close();
});

test('empty candidate set yields empty recommendations', () => {
  const { db, close } = makeDb();
  seedDecomposable(db);
  const { modelId } = trainFromCurrent(db, { ...DEFAULT_CONFIG, epochs: 5 });
  const model = loadModel(db, modelId);
  const recs = topK(model, 'u0', { k: 5, candidates: [] });
  assert.deepEqual(recs, []);
  close();
});

test('NDCG: empty validation set => null; notes explain handling', () => {
  const { db, close } = makeDb();
  seedDecomposable(db);
  const { modelId } = trainFromCurrent(db, { ...DEFAULT_CONFIG, valFraction: 0, epochs: 5 });
  const model = loadModel(db, modelId);
  const { val } = loadSplit(db, model.splitId);
  const result = ndcgAtK(model, val, 5);
  assert.equal(result.value, null);
  assert.equal(result.usersConsidered, 0);
  close();
});

test('NDCG users with no relevant items are excluded, not counted as zero', () => {
  const { db, close } = makeDb();
  // Everyone rates everything 1 => no relevant (>3) items.
  const events = [];
  for (let u = 0; u < 3; u++)
    for (let i = 0; i < 5; i++)
      events.push(rating(`u${u}`, `i${i}`, 1, T0 + u * 10 + i, `l-${u}-${i}`));
  importRatings(db, events);
  const { modelId } = trainFromCurrent(db, { ...DEFAULT_CONFIG, valFraction: 0.4, epochs: 5 });
  const report = evaluateAndStore(db, modelId, 5);
  assert.equal(report.ndcgAtK, null);
  assert.ok(report.notes.some((n) => n.includes('empty relevant sets')));
  close();
});

test('sparse user with one rating stays entirely in training', () => {
  const { db, close } = makeDb();
  const events = [];
  for (let i = 0; i < 5; i++) events.push(rating('dense', `i${i}`, 5, T0 + i, `d-${i}`));
  events.push(rating('sparse', 'i0', 4, T0 + 99, 'sp'));
  importRatings(db, events);
  const { splitId } = createSplitId(db);
  const { train, val } = loadSplit(db, splitId);
  assert.ok(train.some((r) => r.user_id === 'sparse'));
  assert.ok(!val.some((r) => r.user_id === 'sparse'));
  close();
});

function createSplitId(db: ReturnType<typeof makeDb>['db']): { splitId: number } {
  const result = trainFromCurrent(db, { ...DEFAULT_CONFIG, epochs: 1 });
  return { splitId: result.splitId };
}
