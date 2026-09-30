import type { DatabaseSync } from 'node:sqlite';

/**
 * All schema changes live here. Every migration is tracked by name in
 * `schema_migrations` and wrapped so re-running migrate() is a no-op
 * (idempotent / repeatable).
 */

interface Migration {
  name: string;
  up: string;
}

const MIGRATIONS: Migration[] = [
  {
    name: '001_init',
    up: `
    CREATE TABLE rating_events (
      seq        INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id    TEXT NOT NULL,
      item_id    TEXT NOT NULL,
      rating     REAL NOT NULL,
      event_time INTEGER NOT NULL,
      idempotency_key TEXT
    );
    CREATE UNIQUE INDEX ux_rating_events_idem
      ON rating_events(idempotency_key)
      WHERE idempotency_key IS NOT NULL;
    CREATE INDEX ix_rating_events_user_time
      ON rating_events(user_id, event_time, seq);
    CREATE INDEX ix_rating_events_item
      ON rating_events(item_id);

    CREATE TABLE current_ratings (
      user_id    TEXT NOT NULL,
      item_id    TEXT NOT NULL,
      rating     REAL NOT NULL,
      event_time INTEGER NOT NULL,
      event_seq  INTEGER NOT NULL,
      PRIMARY KEY (user_id, item_id)
    );
    CREATE INDEX ix_current_item ON current_ratings(item_id);

    CREATE TABLE splits (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at    INTEGER NOT NULL,
      val_fraction  REAL NOT NULL,
      min_train_per_user INTEGER NOT NULL,
      revision_rule TEXT NOT NULL,
      split_rule    TEXT NOT NULL,
      n_train       INTEGER NOT NULL,
      n_val         INTEGER NOT NULL
    );

    CREATE TABLE split_ratings (
      split_id   INTEGER NOT NULL REFERENCES splits(id),
      user_id    TEXT NOT NULL,
      item_id    TEXT NOT NULL,
      rating     REAL NOT NULL,
      event_time INTEGER NOT NULL,
      event_seq  INTEGER NOT NULL,
      role       TEXT NOT NULL CHECK (role IN ('train','val')),
      PRIMARY KEY (split_id, user_id, item_id)
    );
    CREATE INDEX ix_split_role ON split_ratings(split_id, role);

    CREATE TABLE models (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      split_id     INTEGER NOT NULL REFERENCES splits(id),
      created_at   INTEGER NOT NULL,
      factors      INTEGER NOT NULL,
      epochs       INTEGER NOT NULL,
      learning_rate REAL NOT NULL,
      reg_bias     REAL NOT NULL,
      reg_factor   REAL NOT NULL,
      seed         INTEGER NOT NULL,
      global_mean  REAL NOT NULL,
      update_order TEXT NOT NULL,
      final_train_loss REAL,
      note         TEXT
    );

    CREATE TABLE model_biases (
      model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      kind     TEXT NOT NULL CHECK (kind IN ('user','item')),
      key      TEXT NOT NULL,
      bias     REAL NOT NULL,
      PRIMARY KEY (model_id, kind, key)
    );

    CREATE TABLE model_factors (
      model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      kind     TEXT NOT NULL CHECK (kind IN ('user','item')),
      key      TEXT NOT NULL,
      factor_index INTEGER NOT NULL,
      value    REAL NOT NULL,
      PRIMARY KEY (model_id, kind, key, factor_index)
    );

    CREATE TABLE epoch_losses (
      model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      epoch    INTEGER NOT NULL,
      train_loss REAL NOT NULL,
      PRIMARY KEY (model_id, epoch)
    );

    CREATE TABLE reports (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      model_id    INTEGER NOT NULL REFERENCES models(id),
      split_id    INTEGER NOT NULL REFERENCES splits(id),
      evaluated_at INTEGER NOT NULL,
      k           INTEGER NOT NULL,
      rmse        REAL,
      ndcg_at_k   REAL,
      notes       TEXT NOT NULL
    );
    `,
  },
];

export function migrate(db: DatabaseSync): { applied: string[] } {
  db.exec(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      name TEXT PRIMARY KEY,
      applied_at INTEGER NOT NULL
    );
  `);
  const appliedRow = db.prepare('SELECT name FROM schema_migrations').all() as { name: string }[];
  const applied = new Set(appliedRow.map((r) => r.name));
  const ran: string[] = [];
  for (const migration of MIGRATIONS) {
    if (applied.has(migration.name)) continue;
    const tx = db.exec.bind(db);
    // Run DDL + bookkeeping atomically.
    db.exec('BEGIN');
    try {
      tx(migration.up);
      db.prepare('INSERT INTO schema_migrations(name, applied_at) VALUES (?, ?)').run(
        migration.name,
        Date.now(),
      );
      db.exec('COMMIT');
      ran.push(migration.name);
    } catch (err) {
      db.exec('ROLLBACK');
      throw err;
    }
  }
  return { applied: ran };
}
