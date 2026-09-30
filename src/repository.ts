import type { Db } from './db.js';
import { ApiError } from './errors.js';
import { normalizeTime, requireNumber, requireString } from './validation.js';
import type { RatingInput, RatingRecord } from './types.js';

interface EventRow {
  id: number;
  user_id: string;
  item_id: string;
  rating: number;
  event_time: string;
  idempotency_key: string | null;
  created_at: string;
  active: number;
}

function eventFromRow(row: EventRow): RatingRecord {
  return {
    id: row.id,
    userId: row.user_id,
    itemId: row.item_id,
    rating: row.rating,
    eventTime: row.event_time,
    ...(row.idempotency_key ? { idempotencyKey: row.idempotency_key } : {}),
    createdAt: row.created_at,
    active: row.active === 1,
  };
}

export function normalizeRatingInput(input: unknown, minRating = 1, maxRating = 5): RatingInput {
  if (typeof input !== 'object' || input === null) throw new ApiError(400, 'INVALID_PARAMETER', 'rating object is required');
  const record = input as Record<string, unknown>;
  return {
    userId: requireString(record.userId, 'userId'),
    itemId: requireString(record.itemId, 'itemId'),
    rating: requireNumber(record.rating, 'rating', minRating, maxRating),
    eventTime: normalizeTime(record.eventTime),
    ...(record.idempotencyKey === undefined ? {} : { idempotencyKey: requireString(record.idempotencyKey, 'idempotencyKey') }),
  };
}

export function insertRating(db: Db, raw: unknown, minRating = 1, maxRating = 5): RatingRecord {
  const input = normalizeRatingInput(raw, minRating, maxRating);
  db.exec('BEGIN IMMEDIATE;');
  try {
    if (input.idempotencyKey) {
      const existing = db
        .prepare('SELECT * FROM rating_events WHERE idempotency_key = ?')
        .get(input.idempotencyKey) as unknown as EventRow | undefined;
      if (existing) {
        db.exec('COMMIT;');
        return eventFromRow(existing);
      }
    }
    db.prepare('INSERT OR IGNORE INTO users(user_id) VALUES (?)').run(input.userId);
    db.prepare('INSERT OR IGNORE INTO items(item_id) VALUES (?)').run(input.itemId);
    const inserted = db
      .prepare('INSERT INTO rating_events(user_id,item_id,rating,event_time,idempotency_key) VALUES (?,?,?,?,?)')
      .run(input.userId, input.itemId, input.rating, input.eventTime, input.idempotencyKey ?? null);
    const id = Number(inserted.lastInsertRowid);
    const pairRows = db
      .prepare('SELECT id, event_time FROM rating_events WHERE user_id = ? AND item_id = ? ORDER BY event_time, id')
      .all(input.userId, input.itemId) as unknown as Array<{ id: number; event_time: string }>;
    const latest = pairRows.reduce((current, row) => (row.event_time >= current.event_time ? row : current), pairRows[0]!);
    db.prepare('UPDATE rating_events SET active = CASE WHEN id = ? THEN 1 ELSE 0 END, superseded_by = CASE WHEN id = ? THEN NULL ELSE ? END WHERE user_id = ? AND item_id = ?')
      .run(latest.id, latest.id, latest.id, input.userId, input.itemId);
    const row = db.prepare('SELECT * FROM rating_events WHERE id = ?').get(id) as unknown as EventRow;
    db.exec('COMMIT;');
    return eventFromRow(row);
  } catch (error) {
    db.exec('ROLLBACK;');
    if (error instanceof Error && error.message.includes('UNIQUE constraint failed')) {
      throw new ApiError(409, 'DUPLICATE_REQUEST', 'idempotency key is already used by a different request');
    }
    throw error;
  }
}

export function listRatings(db: Db, limit: number, offset: number, includeInactive = false): { items: RatingRecord[]; total: number } {
  const where = includeInactive ? '' : 'WHERE active = 1';
  const total = (db.prepare(`SELECT count(*) AS count FROM rating_events ${where}`).get() as { count: number }).count;
  const rows = db
    .prepare(`SELECT * FROM rating_events ${where} ORDER BY event_time, id LIMIT ? OFFSET ?`)
    .all(limit, offset) as unknown as EventRow[];
  return { items: rows.map(eventFromRow), total };
}

export function activeRatingsAt(db: Db, cutoffTime: string): { train: RatingRecord[]; validation: RatingRecord[] } {
  const rows = db.prepare('SELECT * FROM rating_events ORDER BY event_time, id').all() as unknown as EventRow[];
  const latestBeforeCutoff = new Map<string, EventRow>();
  const validation: EventRow[] = [];
  for (const row of rows) {
    if (row.event_time >= cutoffTime) {
      validation.push(row);
      continue;
    }
    if (row.event_time > cutoffTime) continue;
    const key = `${row.user_id}\u0000${row.item_id}`;
    const current = latestBeforeCutoff.get(key);
    if (!current || row.event_time > current.event_time || (row.event_time === current.event_time && row.id > current.id)) {
      latestBeforeCutoff.set(key, row);
    }
  }
  const latestValidation = new Map<string, EventRow>();
  for (const row of validation) {
    const key = `${row.user_id}\u0000${row.item_id}`;
    const current = latestValidation.get(key);
    if (!current || row.event_time > current.event_time || (row.event_time === current.event_time && row.id > current.id)) {
      latestValidation.set(key, row);
    }
  }
  return { train: [...latestBeforeCutoff.values()].map(eventFromRow), validation: [...latestValidation.values()].map(eventFromRow) };
}

export function createSplit(db: Db, validationCount: number): { id: number; cutoffTime: string } {
  const count = (db.prepare('SELECT count(*) AS count FROM rating_events').get() as { count: number }).count;
  if (!Number.isInteger(validationCount) || validationCount < 1 || validationCount >= count) {
    throw new ApiError(400, 'INVALID_SPLIT', 'validationCount must be an integer smaller than total events');
  }
  const row = db.prepare('SELECT event_time FROM rating_events ORDER BY event_time, id LIMIT 1 OFFSET ?').get(count - validationCount) as
    | { event_time: string }
    | undefined;
  if (!row) throw new ApiError(400, 'INVALID_SPLIT', 'not enough ratings');
  const cutoffTime = row.event_time;
  const { train, validation } = activeRatingsAt(db, cutoffTime);
  if (train.length === 0 || validation.length === 0) throw new ApiError(400, 'INVALID_SPLIT', 'split must contain training and validation ratings');
  db.exec('BEGIN IMMEDIATE;');
  try {
    const inserted = db.prepare('INSERT INTO splits(cutoff_time, validation_count) VALUES (?,?)').run(cutoffTime, validation.length);
    const splitId = Number(inserted.lastInsertRowid);
    const insert = db.prepare('INSERT INTO split_ratings(split_id,rating_event_id,part) VALUES (?,?,?)');
    for (const rating of train) insert.run(splitId, rating.id, 'train');
    for (const rating of validation) insert.run(splitId, rating.id, 'validation');
    db.exec('COMMIT;');
    return { id: splitId, cutoffTime };
  } catch (error) {
    db.exec('ROLLBACK;');
    throw error;
  }
}

export function getSplitRatings(db: Db, splitId: number): {
  train: RatingRecord[];
  validation: RatingRecord[];
  cutoffTime: string;
  validationCount: number;
} {
  const split = db.prepare('SELECT * FROM splits WHERE id = ?').get(splitId) as
    | { cutoff_time: string; validation_count: number }
    | undefined;
  if (!split) throw new ApiError(404, 'SPLIT_NOT_FOUND', `split ${splitId} not found`);
  const select = db.prepare(
    `SELECT e.* FROM split_ratings s JOIN rating_events e ON e.id = s.rating_event_id
      WHERE s.split_id = ? AND s.part = ? ORDER BY e.event_time, e.id`,
  );
  return {
    train: (select.all(splitId, 'train') as unknown as EventRow[]).map(eventFromRow),
    validation: (select.all(splitId, 'validation') as unknown as EventRow[]).map(eventFromRow),
    cutoffTime: split.cutoff_time,
    validationCount: split.validation_count,
  };
}
