import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { makeDb, rating, T0 } from './helpers.ts';
import { createRequestListener } from '../src/api/app.ts';
import { importRatings } from '../src/db/repo.ts';

async function withServer(fn: (base: string) => Promise<void>): Promise<void> {
  const { db, close } = makeDb();
  const server = createServer(createRequestListener(db));
  await new Promise<void>((resolveListen) => server.listen(0, '127.0.0.1', resolveListen));
  const port = (server.address() as AddressInfo).port;
  try {
    await fn(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise<void>((done) => server.close(() => done()));
    close();
  }
}

async function call(
  base: string,
  path: string,
  init?: { method?: string; body?: unknown },
): Promise<{ status: number; body: any }> {
  const res = await fetch(base + path, {
    method: init?.method ?? 'GET',
    headers: { 'content-type': 'application/json' },
    body: init?.body === undefined ? undefined : JSON.stringify(init.body),
  });
  const body = await res.json();
  return { status: res.status, body };
}

test('health and pagination metadata', async () => {
  await withServer(async (base) => {
    const health = await call(base, '/health');
    assert.equal(health.status, 200);
    const page = await call(base, '/ratings?limit=10&offset=0');
    assert.equal(page.status, 200);
    assert.equal(page.body.limit, 10);
    assert.equal(page.body.offset, 0);
  });
});

test('invalid pagination returns VALIDATION_ERROR code', async () => {
  await withServer(async (base) => {
    const res = await call(base, '/ratings?limit=0');
    assert.equal(res.status, 400);
    assert.equal(res.body.error.code, 'VALIDATION_ERROR');
  });
});

test('import idempotency over HTTP then train/predict/recommend/evaluate', async () => {
  await withServer(async (base) => {
    const events = [];
    for (let u = 0; u < 4; u++)
      for (let i = 0; i < 6; i++)
        events.push(rating(`u${u}`, `i${i}`, (u + i) % 2 === 0 ? 5 : 1, T0 + u * 10 + i, `h-${u}-${i}`));
    const first = await call(base, '/ratings/import', { method: 'POST', body: { events } });
    assert.equal(first.status, 200);
    const again = await call(base, '/ratings/import', { method: 'POST', body: { events } });
    assert.equal(again.body.duplicateIdempotency, events.length);

    const train = await call(base, '/models/train', {
      method: 'POST',
      body: { factors: 2, epochs: 40, learningRate: 0.02, seed: 7, valFraction: 0.25 },
    });
    assert.equal(train.status, 201);
    const modelId = train.body.modelId;

    const predict = await call(base, `/models/${modelId}/predict`, {
      method: 'POST',
      body: { userId: 'u0', itemId: 'i0' },
    });
    assert.equal(predict.status, 200);
    assert.ok(typeof predict.body.prediction === 'number');
    assert.ok('contribution' in predict.body);

    const cold = await call(base, `/models/${modelId}/predict`, {
      method: 'POST',
      body: { userId: 'nobody', itemId: 'nothing' },
    });
    assert.equal(cold.body.kind, 'cold_both');
    assert.equal(cold.body.contribution.source, 'fallback');

    const rec = await call(base, `/models/${modelId}/recommend`, {
      method: 'POST',
      body: { userId: 'u0', k: 3, exclude: 'all_current' },
    });
    assert.equal(rec.status, 200);
    assert.ok(rec.body.recommendations.length <= 3);

    const report = await call(base, `/models/${modelId}/evaluate`, {
      method: 'POST',
      body: { k: 5 },
    });
    assert.equal(report.status, 200);
    assert.ok('rmse' in report.body);
  });
});

test('unknown model returns NOT_FOUND; unknown route returns 404', async () => {
  await withServer(async (base) => {
    const missing = await call(base, '/models/999/losses');
    assert.equal(missing.status, 404);
    assert.equal(missing.body.error.code, 'NOT_FOUND');
    const route = await call(base, '/nope');
    assert.equal(route.status, 404);
  });
});

test('conflicting idempotency payload rolls back with 409', async () => {
  await withServer(async (base) => {
    const ok = await call(base, '/ratings/import', {
      method: 'POST',
      body: { events: [rating('u', 'i', 4, T0, 'k')] },
    });
    assert.equal(ok.status, 200);
    const bad = await call(base, '/ratings/import', {
      method: 'POST',
      body: { events: [rating('u', 'i', 5, T0, 'k')] },
    });
    assert.equal(bad.status, 409);
    assert.equal(bad.body.error.code, 'CONFLICT');
  });
});
