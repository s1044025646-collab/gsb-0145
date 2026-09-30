import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { migrate } from './schema.ts';

export interface OpenDbResult {
  db: DatabaseSync;
  appliedMigrations: string[];
}

/** Open (creating if needed) the project database and ensure schema. */
export function openDatabase(filePath = 'data/app.db'): OpenDbResult {
  const absolute = resolve(filePath);
  mkdirSync(dirname(absolute), { recursive: true });
  const db = new DatabaseSync(absolute, { enableForeignKeyConstraints: true });
  db.exec('PRAGMA journal_mode=WAL;');
  db.exec('PRAGMA foreign_keys=ON;');
  const { applied } = migrate(db);
  return { db, appliedMigrations: applied };
}

/**
 * Checkpoint and close. The checkpoint is required so data written in WAL
 * mode is fully visible to a later connection opening the same file.
 */
export function closeDatabase(db: DatabaseSync): void {
  try {
    db.exec('PRAGMA wal_checkpoint(TRUNCATE);');
  } finally {
    db.close();
  }
}

/** Run a function inside a transaction with rollback on any throw. */
export function withTransaction<T>(db: DatabaseSync, fn: () => T): T {
  db.exec('BEGIN');
  try {
    const result = fn();
    db.exec('COMMIT');
    return result;
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
}
