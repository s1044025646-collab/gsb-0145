import assert from 'node:assert/strict';
import test from 'node:test';
import { openDatabase } from '../src/db.ts';
import { createSplit, getSplitRatings, insertRating, listRatings } from '../src/repository.ts';
import { evaluateAndStoreReport, loadModel, predictWithModel, recommendWithModel, trainAndStoreModel } from '../src/modelStore.ts';
import { ApiError } from '../src/errors.ts';

function resetDb(path: string) {
  const db = openDatabase(path);
  db.exec('PRAGMA foreign_keys = OFF; DELETE FROM model_factors; DELETE FROM reports; DELETE FROM models; DELETE FROM split_ratings; DELETE FROM splits; DELETE FROM rating_events; DELETE FROM users; DELETE FROM items; DELETE FROM sqlite_sequence; PRAGMA foreign_keys = ON;');
  return db;
}

test('repeated submission with idempotency key returns same event', () => {
  const db = resetDb('data/test-idempotency.sqlite');
  const payload = { userId: 'u1', itemId: 'i1', rating: 4, eventTime: '2024-01-01T00:00Z', idempotencyKey: 'same' };
  const first = insertRating(db, payload);
  const second = insertRating(db, { ...payload, rating: 5 });
  assert.equal(first.id, second.id);
  assert.equal(second.rating, 4);
  assert.equal(listRatings(db, 10, 0).total, 1);
  db.close();
});

test('revision imported out of insertion order is selected by public event time', () => {
  const db = resetDb('data/test-out-of-order.sqlite');
  insertRating(db, { userId: 'u1', itemId: 'i1', rating: 1, eventTime: '2024-01-03T00:00Z' });
  insertRating(db, { userId: 'u1', itemId: 'i1', rating: 5, eventTime: '2024-01-01T00:00Z' });
  const active = listRatings(db, 10, 0).items;
  assert.equal(active.length, 1);
  assert.equal(active[0]!.rating, 1);
  db.close();
});

test('public revision rule keeps old value in training and latest revision in validation', () => {
  const db = resetDb('data/test-revision.sqlite');
  insertRating(db, { userId: 'u1', itemId: 'i1', rating: 5, eventTime: '2024-01-01T00:00Z' });
  insertRating(db, { userId: 'u1', itemId: 'i1', rating: 1, eventTime: '2024-01-03T00:00Z' });
  insertRating(db, { userId: 'u1', itemId: 'i2', rating: 4, eventTime: '2024-01-02T00:00Z' });
  const split = getSplitRatings(db, createSplit(db, 1).id);
  assert.deepEqual(split.train.map((row) => [row.itemId, row.rating]).sort(), [['i1', 5], ['i2', 4]]);
  assert.deepEqual(split.validation.map((row) => [row.itemId, row.rating]), [['i1', 1]]);
  db.close();
});

test('frozen split does not acquire a future revision and old predictions survive restart', () => {
  const path = 'data/test-frozen-restart.sqlite';
  const db = resetDb(path);
  insertRating(db, { userId: 'u1', itemId: 'i1', rating: 2, eventTime: '2024-01-01T00:00Z' });
  insertRating(db, { userId: 'u1', itemId: 'i2', rating: 5, eventTime: '2024-01-02T00:00Z' });
  insertRating(db, { userId: 'u1', itemId: 'i2', rating: 1, eventTime: '2024-01-04T00:00Z' });
  insertRating(db, { userId: 'u1', itemId: 'i2', rating: 5, eventTime: '2024-01-10T00:00Z' });
  const splitId = createSplit(db, 2).id;
  const modelId = trainAndStoreModel(db, splitId, { factors: 2, epochs: 2, seed: 9 }).modelId;
  const before = predictWithModel(db, modelId, 'u1', 'i2');
  db.close();

  const reopened = openDatabase(path);
  insertRating(reopened, { userId: 'u1', itemId: 'i2', rating: 1, eventTime: '2024-01-20T00:00Z' });
  const frozen = getSplitRatings(reopened, splitId);
  assert.equal(frozen.validation.find((row) => row.itemId === 'i2')!.rating, 5);
  assert.deepEqual(predictWithModel(reopened, modelId, 'u1', 'i2'), before);
  reopened.close();
});

test('all-tie ordering is stable, sparse user is supported and cold recommendations include known items', () => {
  const db = resetDb('data/test-end-to-end.sqlite');
  const rows = [
    ['u1', 'i1', 5, '2024-01-01T00:00Z'],
    ['u1', 'i2', 5, '2024-01-02T00:00Z'],
    ['u2', 'i1', 5, '2024-01-03T00:00Z'],
    ['u2', 'i2', 5, '2024-01-04T00:00Z'],
    ['u3', 'i3', 1, '2024-01-05T00:00Z'],
    ['u3', 'i2', 4, '2024-01-06T00:00Z'],
  ] as const;
  for (const [userId, itemId, rating, eventTime] of rows) insertRating(db, { userId, itemId, rating, eventTime });
  const modelId = trainAndStoreModel(db, createSplit(db, 1).id, { factors: 2, epochs: 3, seed: 3 }).modelId;
  const report = evaluateAndStoreReport(db, modelId, 3);
  assert.equal(typeof report.rmse, 'number');
  assert.ok(report.ndcg.macroNdcg === null || (report.ndcg.macroNdcg >= 0 && report.ndcg.macroNdcg <= 1));
  assert.deepEqual(recommendWithModel(db, modelId, 'ghost', 5).map((row) => row.itemId), ['i1', 'i2', 'i3']);
  assert.deepEqual(recommendWithModel(db, modelId, 'u3', 5).map((row) => row.itemId), ['i1', 'i2']);
  assert.deepEqual(recommendWithModel(db, modelId, 'u3', 5, true).map((row) => row.itemId), ['i1', 'i2']);
  db.close();
});

test('invalid insert rolls back without partial user or item', () => {
  const db = resetDb('data/test-rollback.sqlite');
  assert.throws(() => insertRating(db, { userId: 'u-bad', itemId: 'i-bad', rating: 9, eventTime: '2024-01-01T00:00Z' }), ApiError);
  assert.equal((db.prepare('SELECT count(*) c FROM users').get() as { c: number }).c, 0);
  assert.equal((db.prepare('SELECT count(*) c FROM items').get() as { c: number }).c, 0);
  db.close();
});

test('migration repeats and archived model version cannot be loaded', () => {
  const path = 'data/test-migration.sqlite';
  const db = resetDb(path);
  insertRating(db, { userId: 'u1', itemId: 'i1', rating: 4, eventTime: '2024-01-01T00:00Z' });
  insertRating(db, { userId: 'u1', itemId: 'i2', rating: 2, eventTime: '2024-01-02T00:00Z' });
  const modelId = trainAndStoreModel(db, createSplit(db, 1).id, { epochs: 1 }).modelId;
  db.close();
  const migrated = openDatabase(path);
  assert.ok(loadModel(migrated, modelId).globalMean > 0);
  migrated.prepare('UPDATE models SET archived_at = ? WHERE id = ?').run('2024-01-03T00:00Z', modelId);
  assert.throws(() => loadModel(migrated, modelId), ApiError);
  migrated.close();
});
