import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { writeFileSync } from 'node:fs';
import type { Db } from './db.js';
import { openDatabase } from './db.js';
import { ApiError } from './errors.js';
import { createSplit, insertRating, listRatings } from './repository.js';
import { compareModels, evaluateAndStoreReport, predictWithModel, recommendWithModel, trainAndStoreModel } from './modelStore.js';
import { normalizePagination, requireInteger, requireString } from './validation.js';

async function readJson(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > 2_000_000) throw new ApiError(413, 'PAYLOAD_TOO_LARGE', 'request body must be smaller than 2 MB');
    chunks.push(buffer);
  }
  if (chunks.length === 0) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw new ApiError(400, 'INVALID_JSON', 'request body must be valid JSON');
  }
}

function send(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
  });
  response.end(JSON.stringify(body));
}

function idsFromBody(body: Record<string, unknown>, field: string): number[] {
  const value = body[field];
  if (!Array.isArray(value) || value.length === 0) throw new ApiError(400, 'INVALID_PARAMETER', `${field} must be a non-empty array`);
  return value.map((item) => requireInteger(item, `${field}[]`, 1, Number.MAX_SAFE_INTEGER));
}

function configFromBody(body: Record<string, unknown>) {
  const allowed = new Set(['factors', 'epochs', 'learningRate', 'regularization', 'seed', 'minRating', 'maxRating']);
  const config: Record<string, number> = {};
  for (const key of allowed) if (body[key] !== undefined) config[key] = requireIntegerOrNumber(body[key], key);
  return config;
}

function requireIntegerOrNumber(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new ApiError(400, 'INVALID_PARAMETER', `${field} must be a number`);
  if (field === 'factors' || field === 'epochs' || field === 'seed' || field === 'minRating' || field === 'maxRating') {
    if (!Number.isInteger(value)) throw new ApiError(400, 'INVALID_PARAMETER', `${field} must be an integer`);
  }
  return value;
}

export function createApp(db: Db): Server {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1');
      const path = url.pathname.replace(/\/$/, '') || '/';
      const method = request.method ?? 'GET';

      if (method === 'GET' && path === '/health') return send(response, 200, { status: 'ok' });

      if (method === 'GET' && path === '/ratings') {
        const { limit, offset } = normalizePagination(url.searchParams);
        const includeInactive = url.searchParams.get('includeInactive') === 'true';
        const result = listRatings(db, limit, offset, includeInactive);
        return send(response, 200, { ...result, limit, offset });
      }

      if (method === 'GET' && path === '/models') {
        const { limit, offset } = normalizePagination(url.searchParams);
        const total = (db.prepare('SELECT count(*) AS count FROM models').get() as { count: number }).count;
        const items = db.prepare('SELECT id,split_id,config_json,global_mean,losses_json,created_at,archived_at FROM models ORDER BY id LIMIT ? OFFSET ?').all(limit, offset);
        return send(response, 200, { items, total, limit, offset });
      }

      if (method === 'GET' && path === '/splits') {
        const { limit, offset } = normalizePagination(url.searchParams);
        const total = (db.prepare('SELECT count(*) AS count FROM splits').get() as { count: number }).count;
        const items = db.prepare('SELECT * FROM splits ORDER BY id LIMIT ? OFFSET ?').all(limit, offset);
        return send(response, 200, { items, total, limit, offset });
      }

      if (method === 'GET' && path === '/reports') {
        const { limit, offset } = normalizePagination(url.searchParams);
        const total = (db.prepare('SELECT count(*) AS count FROM reports').get() as { count: number }).count;
        const items = db.prepare('SELECT id,model_id,rmse,ndcg_json,details_json,created_at FROM reports ORDER BY id LIMIT ? OFFSET ?').all(limit, offset);
        return send(response, 200, { items, total, limit, offset });
      }

      if (method === 'POST' && path === '/ratings') {
        const body = (await readJson(request)) as Record<string, unknown>;
        return send(response, 201, insertRating(db, body));
      }

      if (method === 'POST' && path === '/splits') {
        const body = (await readJson(request)) as Record<string, unknown>;
        const validationCount = requireInteger(body.validationCount ?? 1, 'validationCount', 1, Number.MAX_SAFE_INTEGER);
        const before = (db.prepare('SELECT count(*) AS count FROM splits').get() as { count: number }).count;
        const split = createSplit(db, validationCount);
        const after = (db.prepare('SELECT count(*) AS count FROM splits').get() as { count: number }).count;
        return send(response, 201, { before: { splits: before }, after: { splits: after }, ...split });
      }

      if (method === 'POST' && path === '/models/train') {
        const body = (await readJson(request)) as Record<string, unknown>;
        const splitId = requireInteger(body.splitId, 'splitId', 1, Number.MAX_SAFE_INTEGER);
        const before = (db.prepare('SELECT count(*) AS count FROM models').get() as { count: number }).count;
        const trained = trainAndStoreModel(db, splitId, configFromBody(body));
        const after = (db.prepare('SELECT count(*) AS count FROM models').get() as { count: number }).count;
        return send(response, 201, { before: { models: before }, after: { models: after }, ...trained });
      }

      if (method === 'POST' && path === '/predict') {
        const body = (await readJson(request)) as Record<string, unknown>;
        const modelId = requireInteger(body.modelId, 'modelId', 1, Number.MAX_SAFE_INTEGER);
        const userId = requireString(body.userId, 'userId');
        const itemId = requireString(body.itemId, 'itemId');
        return send(response, 200, predictWithModel(db, modelId, userId, itemId));
      }

      if (method === 'POST' && path === '/recommend') {
        const body = (await readJson(request)) as Record<string, unknown>;
        const modelId = requireInteger(body.modelId, 'modelId', 1, Number.MAX_SAFE_INTEGER);
        const userId = requireString(body.userId, 'userId');
        const topK = requireInteger(body.topK ?? 10, 'topK', 1, 200);
        const excludeSeen = body.excludeSeen === undefined ? true : body.excludeSeen === true;
        const items = recommendWithModel(db, modelId, userId, topK, excludeSeen);
        return send(response, 200, { userId, topK, seenExcluded: excludeSeen, items });
      }

      if (method === 'POST' && path === '/models/evaluate') {
        const body = (await readJson(request)) as Record<string, unknown>;
        const modelId = requireInteger(body.modelId, 'modelId', 1, Number.MAX_SAFE_INTEGER);
        const k = requireInteger(body.k ?? 5, 'k', 1, 200);
        return send(response, 200, evaluateAndStoreReport(db, modelId, k));
      }

      if (method === 'POST' && path === '/models/compare') {
        const body = (await readJson(request)) as Record<string, unknown>;
        const modelIds = idsFromBody(body, 'modelIds');
        const k = requireInteger(body.k ?? 5, 'k', 1, 200);
        return send(response, 200, { models: compareModels(db, modelIds, k) });
      }

      throw new ApiError(404, 'NOT_FOUND', `no route for ${method} ${path}`);
    } catch (error) {
      if (error instanceof ApiError) return send(response, error.status, { error: { code: error.code, message: error.message, details: error.details } });
      const message = error instanceof Error ? error.message : 'unknown error';
      return send(response, 500, { error: { code: 'INTERNAL_ERROR', message } });
    }
  });
}

export function startServer(db: Db = openDatabase(), requestedPort = Number(process.env.CF_PORT ?? 0)): Promise<Server> {
  const server = createApp(db);
  return new Promise((resolve) => {
    server.listen(requestedPort, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') throw new Error('expected TCP address');
      writeFileSync('data/server.json', JSON.stringify({ host: '127.0.0.1', port: address.port, pid: process.pid }, null, 2));
      console.log(`listening on http://127.0.0.1:${address.port} pid=${process.pid}`);
      resolve(server);
    });
  });
}

if (import.meta.url === `file://${process.argv[1]?.replaceAll('\\', '/')}`) {
  void startServer();
}
