import type { DatabaseSync } from 'node:sqlite';
import type { CurrentRating, RatingEvent } from '../model/types.ts';
import { conflict, validation } from '../model/errors.ts';
import { withTransaction } from './database.ts';

export const RATING_MIN = 1;
export const RATING_MAX = 5;

export interface ImportResult {
  inserted: number;
  insertedEvents: number;
  duplicateIdempotency: number;
  ignoredOlder: number;
  currentTotal: number;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

function assertValidRating(event: RatingEvent): void {
  if (typeof event.userId !== 'string' || event.userId.trim() === '') {
    throw validation('userId must be a non-empty string');
  }
  if (typeof event.itemId !== 'string' || event.itemId.trim() === '') {
    throw validation('itemId must be a non-empty string');
  }
  if (typeof event.rating !== 'number' || Number.isNaN(event.rating)) {
    throw validation('rating must be a number');
  }
  if (event.rating < RATING_MIN || event.rating > RATING_MAX) {
    throw validation(`rating must be within [${RATING_MIN}, ${RATING_MAX}]`, {
      rating: event.rating,
    });
  }
  if (!Number.isInteger(event.eventTime) || event.eventTime < 0) {
    throw validation('eventTime must be a non-negative integer epoch (ms)');
  }
}

const insertEvent = (db: DatabaseSync) =>
  db.prepare(
    'INSERT INTO rating_events(user_id,item_id,rating,event_time,idempotency_key) VALUES (?,?,?,?,?)',
  );

const selectCurrent = (db: DatabaseSync) =>
  db.prepare('SELECT * FROM current_ratings WHERE user_id=? AND item_id=?');

const insertCurrent = (db: DatabaseSync) =>
  db.prepare(
    'INSERT INTO current_ratings(user_id,item_id,rating,event_time,event_seq) VALUES (?,?,?,?,?)',
  );

const updateCurrent = (db: DatabaseSync) =>
  db.prepare(
    'UPDATE current_ratings SET rating=?, event_time=?, event_seq=? WHERE user_id=? AND item_id=?',
  );

/**
 * Revision rule (public, deterministic):
 * last-write-wins by (event_time, insertion seq); an older or equal-time
 * event never replaces the current rating. Equal-time keeps the first-seen
 * value so re-imports are stable.
 */
function applyEvent(
  db: DatabaseSync,
  event: RatingEvent,
  seq: number,
  counters: ImportResult,
): void {
  const existing = selectCurrent(db).get(event.userId, event.itemId) as
    | CurrentRating
    | undefined;
  if (existing) {
    const newer = event.eventTime > existing.event_time;
    if (!newer) {
      counters.ignoredOlder++;
      return;
    }
    updateCurrent(db).run(
      event.rating,
      event.eventTime,
      seq,
      event.userId,
      event.itemId,
    );
    return;
  }
  insertCurrent(db).run(
    event.userId,
    event.itemId,
    event.rating,
    event.eventTime,
    seq,
  );
}

/**
 * Import a batch atomically. Duplicate idempotency keys cause the whole batch
 * to reject (CONFLICT) and roll back, except exact re-submissions which are
 * treated as no-ops. Within this engine a batch is always fully applied or
 * fully rolled back.
 */
export function importRatings(
  db: DatabaseSync,
  rawEvents: RatingEvent[],
): ImportResult {
  if (!Array.isArray(rawEvents) || rawEvents.length === 0) {
    throw validation('events must be a non-empty array');
  }
  for (const event of rawEvents) assertValidRating(event);

  const seenKeys = new Set<string>();
  for (const event of rawEvents) {
    const key = event.idempotencyKey;
    if (key !== undefined) {
      if (seenKeys.has(key)) {
        throw conflict(`duplicate idempotencyKey inside batch: ${key}`);
      }
      seenKeys.add(key);
    }
  }

  const keySelect = db.prepare(
    'SELECT user_id,item_id,rating,event_time FROM rating_events WHERE idempotency_key=?',
  );

  return withTransaction(db, () => {
    const counters: ImportResult = {
      inserted: 0,
      insertedEvents: 0,
      duplicateIdempotency: 0,
      ignoredOlder: 0,
      currentTotal: 0,
    };
    for (const event of rawEvents) {
      if (event.idempotencyKey !== undefined) {
        const prior = keySelect.get(event.idempotencyKey) as
          | { user_id: string; item_id: string; rating: number; event_time: number }
          | undefined;
        if (prior) {
          const same =
            prior.user_id === event.userId &&
            prior.item_id === event.itemId &&
            prior.rating === event.rating &&
            prior.event_time === event.eventTime;
          if (!same) {
            throw conflict(
              `idempotencyKey ${event.idempotencyKey} reused with different payload`,
            );
          }
          counters.duplicateIdempotency++;
          continue;
        }
      }
      const info = insertEvent(db).run(
        event.userId,
        event.itemId,
        event.rating,
        event.eventTime,
        event.idempotencyKey ?? null,
      );
      const seq = Number(info.lastInsertRowid);
      applyEvent(db, event, seq, counters);
      counters.inserted++;
      counters.insertedEvents++;
    }
    counters.currentTotal = countCurrent(db);
    return counters;
  });
}

export function countCurrent(db: DatabaseSync): number {
  const row = db.prepare('SELECT COUNT(*) AS n FROM current_ratings').get() as {
    n: number;
  };
  return Number(row.n);
}

export function countEvents(db: DatabaseSync): number {
  const row = db.prepare('SELECT COUNT(*) AS n FROM rating_events').get() as {
    n: number;
  };
  return Number(row.n);
}

export function allCurrentRatings(db: DatabaseSync): CurrentRating[] {
  return db
    .prepare(
      'SELECT user_id,item_id,rating,event_time,event_seq FROM current_ratings',
    )
    .all() as CurrentRating[];
}

export function itemsForUser(db: DatabaseSync, userId: string): Set<string> {
  const rows = db
    .prepare('SELECT item_id FROM current_ratings WHERE user_id=?')
    .all(userId) as { item_id: string }[];
  return new Set(rows.map((r) => r.item_id));
}

export function distinctItems(db: DatabaseSync): string[] {
  const rows = db
    .prepare('SELECT DISTINCT item_id FROM current_ratings ORDER BY item_id')
    .all() as { item_id: string }[];
  return rows.map((r) => r.item_id);
}

export function distinctUsers(db: DatabaseSync): string[] {
  const rows = db
    .prepare('SELECT DISTINCT user_id FROM current_ratings ORDER BY user_id')
    .all() as { user_id: string }[];
  return rows.map((r) => r.user_id);
}

export function paginateEvents(
  db: DatabaseSync,
  limit: number,
  offset: number,
): Page<Record<string, unknown>> {
  const total = countEvents(db);
  const items = db
    .prepare(
      'SELECT seq,user_id,item_id,rating,event_time,idempotency_key FROM rating_events ORDER BY seq LIMIT ? OFFSET ?',
    )
    .all(limit, offset) as Record<string, unknown>[];
  return { items, total, limit, offset };
}

export function paginateCurrent(
  db: DatabaseSync,
  limit: number,
  offset: number,
): Page<CurrentRating> {
  const total = countCurrent(db);
  const items = db
    .prepare(
      'SELECT user_id,item_id,rating,event_time,event_seq FROM current_ratings ORDER BY user_id,item_id LIMIT ? OFFSET ?',
    )
    .all(limit, offset) as CurrentRating[];
  return { items, total, limit, offset };
}
