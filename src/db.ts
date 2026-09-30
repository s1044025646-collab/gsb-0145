import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

export type Db = DatabaseSync;

export function openDatabase(path = process.env.CF_DB_PATH ?? 'data/app.sqlite'): Db {
  const absolute = resolve(path);
  mkdirSync(dirname(absolute), { recursive: true });
  const db = new DatabaseSync(absolute);
  db.exec('PRAGMA foreign_keys = ON;');
  migrate(db);
  return db;
}

export function migrate(db: Db): void {
  db.exec('BEGIN IMMEDIATE;');
  try {
    db.exec(`
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
      );

      CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
      );

      CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
      );

      CREATE TABLE IF NOT EXISTS rating_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        rating REAL NOT NULL,
        event_time TEXT NOT NULL,
        idempotency_key TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        active INTEGER NOT NULL DEFAULT 1,
        superseded_by INTEGER,
        UNIQUE(idempotency_key),
        FOREIGN KEY(user_id) REFERENCES users(user_id),
        FOREIGN KEY(item_id) REFERENCES items(item_id),
        FOREIGN KEY(superseded_by) REFERENCES rating_events(id)
      );

      CREATE INDEX IF NOT EXISTS idx_rating_pair_time
        ON rating_events(user_id, item_id, event_time, id);
      CREATE INDEX IF NOT EXISTS idx_rating_active_pair
        ON rating_events(active, user_id, item_id);

      CREATE TABLE IF NOT EXISTS splits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cutoff_time TEXT NOT NULL,
        validation_count INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
      );

      CREATE TABLE IF NOT EXISTS split_ratings (
        split_id INTEGER NOT NULL,
        rating_event_id INTEGER NOT NULL,
        part TEXT NOT NULL CHECK(part IN ('train', 'validation')),
        PRIMARY KEY(split_id, rating_event_id),
        FOREIGN KEY(split_id) REFERENCES splits(id),
        FOREIGN KEY(rating_event_id) REFERENCES rating_events(id)
      );

      CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        split_id INTEGER NOT NULL,
        config_json TEXT NOT NULL,
        global_mean REAL NOT NULL,
        losses_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        archived_at TEXT,
        FOREIGN KEY(split_id) REFERENCES splits(id)
      );

      CREATE TABLE IF NOT EXISTS model_factors (
        model_id INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('user', 'item')),
        key TEXT NOT NULL,
        bias REAL NOT NULL,
        vector_json TEXT NOT NULL,
        PRIMARY KEY(model_id, kind, key),
        FOREIGN KEY(model_id) REFERENCES models(id)
      );

      CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER NOT NULL UNIQUE,
        rmse REAL,
        ndcg_json TEXT,
        details_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY(model_id) REFERENCES models(id)
      );
    `);
    const applied = db.prepare('SELECT version FROM schema_migrations').all() as Array<{ version: number }>;
    if (!applied.some((row) => row.version === 1)) {
      db.prepare('INSERT INTO schema_migrations(version) VALUES (1)').run();
    }
    db.exec('COMMIT;');
  } catch (error) {
    db.exec('ROLLBACK;');
    throw error;
  }
}
