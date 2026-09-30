import type { IncomingMessage, ServerResponse } from 'node:http';
import type { DatabaseSync } from 'node:sqlite';
import {
  countCurrent,
  countEvents,
  distinctItems,
  distinctUsers,
  importRatings,
  itemsForUser,
  paginateCurrent,
  paginateEvents,
} from '../db/repo.ts';
import { DEFAULT_CONFIG } from '../model/types.ts';
import type { RatingEvent, TrainConfig } from '../model/types.ts';
import {
  compareModels,
  evaluateAndStore,
  trainFromCurrent,
} from '../model/service.ts';
import { loadModel } from '../model/serialize.ts';
import { explain } from '../model/mf.ts';
import { topK } from '../model/recommend.ts';
import { handleError, pagination, readJson, sendJson } from './helpers.ts';
import { AppError, validation } from '../model/errors.ts';

type Handler = (
  req: IncomingMessage,
  res: ServerResponse,
  ctx: { url: URL; params: Record<string, string>; body: unknown },
) => void;

interface Route {
  method: string;
  regex: RegExp;
  keys: string[];
  handler: Handler;
}

function route(method: string, path: string, handler: Handler): Route {
  const keys: string[] = [];
  const regex = new RegExp(
    `^${path.replace(/:([a-z]+)/g, (_, key: string) => {
      keys.push(key);
      return '([^/]+)';
    })}$`,
  );
  return { method, regex, keys, handler };
}

function mergeConfig(body: Record<string, unknown>): TrainConfig {
  return { ...DEFAULT_CONFIG, ...body } as TrainConfig;
}

export function createRequestListener(db: DatabaseSync) {
  const routes: Route[] = [
    route('GET', '/health', (_req, res) =>
      sendJson(res, 200, { status: 'ok', events: countEvents(db), ratings: countCurrent(db) }),
    ),

    route('GET', '/ratings', (_req, res, ctx) => {
      const { limit, offset } = pagination(ctx.url);
      sendJson(res, 200, paginateCurrent(db, limit, offset));
    }),

    route('GET', '/events', (_req, res, ctx) => {
      const { limit, offset } = pagination(ctx.url);
      sendJson(res, 200, paginateEvents(db, limit, offset));
    }),

    route('GET', '/items', (_req, res, ctx) => {
      const { limit, offset } = pagination(ctx.url);
      const all = distinctItems(db);
      sendJson(res, 200, {
        items: all.slice(offset, offset + limit),
        total: all.length,
        limit,
        offset,
      });
    }),

    route('GET', '/users', (_req, res, ctx) => {
      const { limit, offset } = pagination(ctx.url);
      const all = distinctUsers(db);
      sendJson(res, 200, {
        users: all.slice(offset, offset + limit),
        total: all.length,
        limit,
        offset,
      });
    }),

    route('POST', '/ratings/import', (_req, res, ctx) => {
      const body = ctx.body as { events?: RatingEvent[] };
      if (!body.events) throw validation('body requires an "events" array');
      sendJson(res, 200, importRatings(db, body.events));
    }),

    route('POST', '/models/train', (_req, res, ctx) => {
      const config = mergeConfig((ctx.body ?? {}) as Record<string, unknown>);
      const result = trainFromCurrent(db, config);
      sendJson(res, 201, result);
    }),

    route('GET', '/models', (_req, res) => {
      sendJson(res, 200, db.prepare('SELECT * FROM models ORDER BY id').all());
    }),

    route('GET', '/models/:id', (_req, res, ctx) => {
      const id = Number(ctx.params.id);
      const model = loadModel(db, id);
      sendJson(res, 200, {
        id: model.id,
        splitId: model.splitId,
        config: model.config,
        globalMean: model.data.globalMean,
        epochLosses: model.data.stats.epochLosses,
        finalTrainLoss: model.data.stats.finalLoss,
      });
    }),

    route('GET', '/models/:id/losses', (_req, res, ctx) => {
      const model = loadModel(db, Number(ctx.params.id));
      sendJson(res, 200, {
        modelId: model.id,
        epochLosses: model.data.stats.epochLosses,
      });
    }),

    route('POST', '/models/:id/evaluate', (_req, res, ctx) => {
      const body = (ctx.body ?? {}) as { k?: number };
      const k = body.k ?? 10;
      sendJson(res, 200, evaluateAndStore(db, Number(ctx.params.id), k));
    }),

    route('POST', '/models/compare', (_req, res, ctx) => {
      const body = ctx.body as { modelIds?: number[]; k?: number };
      if (!Array.isArray(body.modelIds) || body.modelIds.length === 0) {
        throw validation('body requires a non-empty "modelIds" array');
      }
      sendJson(res, 200, compareModels(db, body.modelIds.map(Number), body.k ?? 10));
    }),

    route('POST', '/models/:id/predict', (_req, res, ctx) => {
      const body = ctx.body as { userId?: string; itemId?: string };
      if (!body.userId || !body.itemId) throw validation('userId and itemId are required');
      const model = loadModel(db, Number(ctx.params.id));
      const result = explain(model.data, model.fallback, body.userId, body.itemId);
      sendJson(res, 200, {
        userId: body.userId,
        itemId: body.itemId,
        ...result,
        disclaimer: 'factor similarity is correlational, not causal',
      });
    }),

    route('POST', '/models/:id/recommend', (_req, res, ctx) => {
      const body = ctx.body as {
        userId?: string;
        k?: number;
        candidates?: string[];
        exclude?: 'train' | 'all_current';
      };
      if (!body.userId) throw validation('userId is required');
      const model = loadModel(db, Number(ctx.params.id));
      const recommendations = topK(model, body.userId, {
        k: body.k ?? 10,
        candidates: body.candidates,
        exclude: body.exclude,
        currentlySeen: itemsForUser(db, body.userId),
      });
      sendJson(res, 200, {
        userId: body.userId,
        exclude: body.exclude ?? 'train',
        seenItemsExcluded: [...(itemsForUser(db, body.userId))].sort(),
        recommendations,
      });
    }),

    route('GET', '/reports', (_req, res) => {
      sendJson(res, 200, db.prepare('SELECT * FROM reports ORDER BY id').all());
    }),
  ];

  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    try {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1');
      for (const candidate of routes) {
        if (candidate.method !== req.method) continue;
        const match = candidate.regex.exec(url.pathname);
        if (!match) continue;
        const params: Record<string, string> = {};
        candidate.keys.forEach((key, index) => {
          params[key] = decodeURIComponent(match[index + 1]!);
        });
        const body =
          req.method === 'POST' || req.method === 'PUT' ? await readJson(req) : {};
        candidate.handler(req, res, { url, params, body });
        return;
      }
      sendJson(res, 404, { error: { code: 'NOT_FOUND', message: `no route for ${req.method} ${url.pathname}` } });
    } catch (err) {
      handleError(res, err instanceof AppError ? err : err);
    }
  };
}
