import assert from 'node:assert/strict';
import test from 'node:test';
import type { AddressInfo } from 'node:net';
import { openDatabase } from '../src/db.ts';
import { createApp } from '../src/server.ts';
import { insertRating } from '../src/repository.ts';

async function request(port: number, path: string, init?: { method?: string; body?: unknown }) {
  const options: RequestInit = { method: init?.method ?? 'GET' };
  if (init?.body) {
    options.headers = { 'content-type': 'application/json' };
    options.body = JSON.stringify(init.body);
  }
  const response = await fetch(`http://127.0.0.1:${port}${path}`, options);
  return { status: response.status, body: (await response.json()) as any };
}

test('HTTP API validates, paginates, trains and returns errors', async () => {
  const db = openDatabase('data/test-api.sqlite');
  db.exec('PRAGMA foreign_keys = OFF; DELETE FROM model_factors; DELETE FROM reports; DELETE FROM models; DELETE FROM split_ratings; DELETE FROM splits; DELETE FROM rating_events; DELETE FROM users; DELETE FROM items; DELETE FROM sqlite_sequence; PRAGMA foreign_keys = ON;');
  const server = createApp(db);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;
  try {
    assert.equal((await request(port, '/health')).status, 200);
    const invalid = await request(port, '/ratings', { method: 'POST', body: { userId: 'u1', itemId: 'i1', rating: 9, eventTime: 'bad' } });
    assert.equal(invalid.status, 400);
    assert.equal(invalid.body.error.code, 'INVALID_PARAMETER');
    for (let index = 1; index <= 3; index += 1) {
      const response = await request(port, '/ratings', {
        method: 'POST',
        body: { userId: 'u1', itemId: `i${index}`, rating: index + 2, eventTime: `2024-01-0${index}T00:00Z`, idempotencyKey: `api-${index}` },
      });
      assert.equal(response.status, 201);
    }
    const duplicate = await request(port, '/ratings', {
      method: 'POST',
      body: { userId: 'different', itemId: 'different', rating: 1, eventTime: '2024-01-01T00:00Z', idempotencyKey: 'api-1' },
    });
    assert.equal(duplicate.body.userId, 'u1');
    const page = await request(port, '/ratings?limit=2&offset=0');
    assert.equal(page.body.items.length, 2);
    assert.equal(page.body.total, 3);
    const split = await request(port, '/splits', { method: 'POST', body: { validationCount: 1 } });
    assert.equal(split.status, 201);
    const trained = await request(port, '/models/train', { method: 'POST', body: { splitId: split.body.id, epochs: 2, factors: 2, seed: 2 } });
    assert.equal(trained.status, 201);
    const prediction = await request(port, '/predict', { method: 'POST', body: { modelId: trained.body.modelId, userId: 'u1', itemId: 'i9' } });
    assert.equal(prediction.status, 200);
    assert.equal(prediction.body.fallback, true);
    const evaluated = await request(port, '/models/evaluate', { method: 'POST', body: { modelId: trained.body.modelId, k: 3 } });
    assert.equal(evaluated.status, 200);
    const missing = await request(port, '/predict', { method: 'POST', body: { modelId: 9999, userId: 'u1', itemId: 'i1' } });
    assert.equal(missing.status, 404);
    assert.equal(missing.body.error.code, 'MODEL_NOT_FOUND');
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
    db.close();
  }
});
