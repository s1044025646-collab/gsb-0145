import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDb, rating, T0 } from './helpers.ts';
import { importRatings } from '../src/db/repo.ts';
import { computeSplit, createSplit, loadSplit } from '../src/model/split.ts';
import { DEFAULT_CONFIG } from '../src/model/types.ts';
import { AppError } from '../src/model/errors.ts';

test('revision rule: later event_time wins, older is ignored', () => {
  const { db, close } = makeDb();
  importRatings(db, [rating('u', 'i', 2, T0 + 100, 'a')]);
  const later = importRatings(db, [
    rating('u', 'i', 5, T0 + 200, 'b'),
    rating('u', 'i', 1, T0 + 150, 'c'),
  ]);
  assert.equal(later.ignoredOlder, 1);
  const row = db.prepare('SELECT rating FROM current_ratings').get() as { rating: number };
  assert.equal(row.rating, 5);
  close();
});

test('equal event_time keeps first-seen value (stable re-import)', () => {
  const { db, close } = makeDb();
  importRatings(db, [rating('u', 'i', 3, T0, 'a')]);
  const res = importRatings(db, [rating('u', 'i', 4, T0, 'b')]);
  assert.equal(res.ignoredOlder, 1);
  const row = db.prepare('SELECT rating FROM current_ratings').get() as { rating: number };
  assert.equal(row.rating, 3);
  close();
});

test('idempotency key resubmission is a no-op; different payload conflicts', () => {
  const { db, close } = makeDb();
  const first = importRatings(db, [rating('u', 'i', 4, T0, 'dup')]);
  assert.equal(first.inserted, 1);
  const again = importRatings(db, [rating('u', 'i', 4, T0, 'dup')]);
  assert.equal(again.duplicateIdempotency, 1);
  assert.equal(again.inserted, 0);
  assert.throws(
    () => importRatings(db, [rating('u', 'i', 5, T0, 'dup')]),
    (err: unknown) => err instanceof AppError && err.code === 'CONFLICT',
  );
  close();
});

test('batch rolls back when one event is invalid', () => {
  const { db, close } = makeDb();
  assert.throws(() =>
    importRatings(db, [
      rating('u', 'i', 4, T0, 'ok'),
      rating('u', 'j', 99, T0 + 1, 'bad'),
    ]),
  );
  const events = db.prepare('SELECT COUNT(*) AS n FROM rating_events').get() as { n: number };
  assert.equal(events.n, 0);
  const current = db.prepare('SELECT COUNT(*) AS n FROM current_ratings').get() as { n: number };
  assert.equal(current.n, 0);
  close();
});

test('validation range enforced', () => {
  const { db, close } = makeDb();
  assert.throws(() => importRatings(db, [rating('u', 'i', 0, T0)]));
  assert.throws(() => importRatings(db, [rating('u', 'i', 5.5, T0)]));
  close();
});

test('chronological split never puts an earlier rating in validation', () => {
  const rows = [1, 2, 3, 4, 5].map((r, idx) => ({
    user_id: 'u',
    item_id: `i${idx}`,
    rating: r,
    event_time: T0 + idx * 10,
    event_seq: idx + 1,
  }));
  const { train, val } = computeSplit(rows, { valFraction: 0.4, minTrainPerUser: 1 });
  assert.equal(val.length, 2);
  assert.ok(val.every((v) => train.every((t) => v.event_time > t.event_time)));
  assert.deepEqual(val.map((r) => r.item_id), ['i3', 'i4']);
});

test('no future leakage: every validation event time >= every train event time per user', () => {
  const { db, close } = makeDb();
  const events = [];
  for (let u = 0; u < 3; u++) {
    for (let i = 0; i < 6; i++) {
      events.push(rating(`u${u}`, `i${i}`, 1 + (i % 5), T0 + i * 10 + u, `k${u}-${i}`));
    }
  }
  importRatings(db, events);
  const { splitId } = createSplit(db, { ...DEFAULT_CONFIG, valFraction: 0.4 });
  const { train, val } = loadSplit(db, splitId);
  const by = <T>(rows: T[], f: (r: T) => string) => {
    const m = new Map<string, T[]>();
    for (const r of rows) {
      const key = f(r);
      m.set(key, [...(m.get(key) ?? []), r]);
    }
    return m;
  };
  const trainByUser = by(train, (r) => r.user_id);
  for (const v of val) {
    const maxTrain = Math.max(...trainByUser.get(v.user_id)!.map((r) => r.event_time));
    assert.ok(v.event_time >= maxTrain);
  }
  close();
});

test('all-tie ratings split and evaluate without NaN', () => {
  const rows = Array.from({ length: 6 }, (_, i) => ({
    user_id: 'u',
    item_id: `i${i}`,
    rating: 4,
    event_time: T0 + i,
    event_seq: i + 1,
  }));
  const { train, val } = computeSplit(rows, { valFraction: 0.5, minTrainPerUser: 1 });
  assert.equal(train.length + val.length, 6);
  assert.ok(train.every((r) => r.rating === 4));
});
