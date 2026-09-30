import { ApiError } from './errors.js';

export function requireString(value: unknown, field: string, maxLength = 200): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new ApiError(400, 'INVALID_PARAMETER', `${field} is required`, { field });
  }
  const text = value.trim();
  if (text.length > maxLength) throw new ApiError(400, 'INVALID_PARAMETER', `${field} too long`, { field });
  return text;
}

export function requireNumber(value: unknown, field: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ApiError(400, 'INVALID_PARAMETER', `${field} must be a finite number`, { field });
  }
  if (value < min || value > max) throw new ApiError(400, 'INVALID_PARAMETER', `${field} must be between ${min} and ${max}`, { field });
  return value;
}

export function requireInteger(value: unknown, field: string, min: number, max: number): number {
  const number = requireNumber(value, field, min, max);
  if (!Number.isInteger(number)) throw new ApiError(400, 'INVALID_PARAMETER', `${field} must be an integer`, { field });
  return number;
}

export function normalizeTime(value: unknown, field = 'eventTime'): string {
  const text = requireString(value, field, 100);
  const milliseconds = Date.parse(text);
  if (!Number.isFinite(milliseconds)) throw new ApiError(400, 'INVALID_PARAMETER', `${field} must be valid ISO date time`, { field });
  return new Date(milliseconds).toISOString();
}

export function normalizePagination(query: URLSearchParams): { limit: number; offset: number } {
  const rawLimit = query.get('limit') ?? '50';
  const rawOffset = query.get('offset') ?? '0';
  const limit = requireInteger(Number(rawLimit), 'limit', 1, 200);
  const offset = requireInteger(Number(rawOffset), 'offset', 0, Number.MAX_SAFE_INTEGER);
  return { limit, offset };
}
