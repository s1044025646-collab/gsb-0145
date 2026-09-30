import { rmSync } from 'node:fs';
import { join } from 'node:path';
import type { DatabaseSync } from 'node:sqlite';
import { closeDatabase, openDatabase } from '../src/db/database.ts';
import type { RatingEvent } from '../src/model/types.ts';

let counter = 0;

export function tempDbPath(): string {
  counter++;
  return join('tmp', `test-${process.pid}-${counter}-${Math.random().toString(36).slice(2, 8)}.db`);
}

export function makeDb(): { db: DatabaseSync; path: string; close: () => void } {
  const path = tempDbPath();
  const { db } = openDatabase(path);
  return {
    db,
    path,
    close: () => {
      closeDatabase(db);
      for (const suffix of ['', '-wal', '-shm']) rmSync(path + suffix, { force: true });
    },
  };
}

export function rating(
  userId: string,
  itemId: string,
  rating: number,
  eventTime: number,
  idempotencyKey?: string,
): RatingEvent {
  return { userId, itemId, rating, eventTime, idempotencyKey };
}

export const T0 = Date.UTC(2024, 0, 1);
