import type { IncomingMessage, ServerResponse } from 'node:http';
import { AppError } from '../model/errors.ts';

export function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
): void {
  const payload = JSON.stringify(body, null, 2);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

export async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString('utf-8');
  if (raw.length === 0) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new AppError('VALIDATION_ERROR', 'request body must be valid JSON');
  }
}

export function pagination(query: URL): { limit: number; offset: number } {
  const limitParam = Number(query.searchParams.get('limit') ?? '50');
  const offsetParam = Number(query.searchParams.get('offset') ?? '0');
  if (!Number.isInteger(limitParam) || limitParam < 1 || limitParam > 200) {
    throw new AppError('VALIDATION_ERROR', 'limit must be an integer in [1,200]');
  }
  if (!Number.isInteger(offsetParam) || offsetParam < 0) {
    throw new AppError('VALIDATION_ERROR', 'offset must be a non-negative integer');
  }
  return { limit: limitParam, offset: offsetParam };
}

export function handleError(res: ServerResponse, err: unknown): void {
  if (err instanceof AppError) {
    const status =
      err.code === 'NOT_FOUND'
        ? 404
        : err.code === 'CONFLICT'
          ? 409
          : err.code === 'UNPROCESSABLE'
            ? 422
            : err.code === 'RANGE_ERROR'
              ? 416
              : 400;
    sendJson(res, status, { error: { code: err.code, message: err.message, details: err.details } });
    return;
  }
  const message = err instanceof Error ? err.message : String(err);
  sendJson(res, 500, { error: { code: 'INTERNAL', message } });
}
