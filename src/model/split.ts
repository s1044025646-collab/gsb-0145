import type { DatabaseSync } from 'node:sqlite';
import { allCurrentRatings } from '../db/repo.ts';
import { withTransaction } from '../db/database.ts';
import { unprocessable, validation } from './errors.ts';
import type { CurrentRating, TrainConfig } from './types.ts';

export interface SplitResult {
  splitId: number;
  train: CurrentRating[];
  val: CurrentRating[];
}

/**
 * Chronological split, computed per user. Each user's ratings are ordered by
 * (event_time, event_seq); the latest `round(n * valFraction)` ratings form
 * validation, the rest training. Users with <= minTrainPerUser ratings stay
 * fully in training. Validation statistics are never used for initialization.
 */
export function computeSplit(
  ratings: CurrentRating[],
  config: Pick<TrainConfig, 'valFraction' | 'minTrainPerUser'>,
): { train: CurrentRating[]; val: CurrentRating[] } {
  if (!(config.valFraction >= 0 && config.valFraction < 1)) {
    throw validation('valFraction must be in [0, 1)');
  }
  const byUser = new Map<string, CurrentRating[]>();
  for (const row of ratings) {
    const list = byUser.get(row.user_id) ?? [];
    list.push(row);
    byUser.set(row.user_id, list);
  }
  const train: CurrentRating[] = [];
  const val: CurrentRating[] = [];
  for (const list of byUser.values()) {
    list.sort(
      (a, b) =>
        a.event_time - b.event_time || a.event_seq - b.event_seq,
    );
    if (list.length <= config.minTrainPerUser) {
      train.push(...list);
      continue;
    }
    const nVal = Math.floor(list.length * config.valFraction);
    const cut = list.length - nVal;
    for (let i = 0; i < list.length; i++) {
      (i < cut ? train : val).push(list[i]!);
    }
  }
  return { train, val };
}

export function createSplit(db: DatabaseSync, config: TrainConfig): SplitResult {
  const current = allCurrentRatings(db);
  if (current.length === 0) {
    throw unprocessable('no ratings available to split');
  }
  const { train, val } = computeSplit(current, config);
  if (train.length === 0) {
    throw unprocessable('split produced an empty training set');
  }
  return withTransaction(db, () => {
    const info = db
      .prepare(
        `INSERT INTO splits(created_at,val_fraction,min_train_per_user,revision_rule,split_rule,n_train,n_val)
         VALUES (?,?,?,?,?,?,?)`,
      )
      .run(
        Date.now(),
        config.valFraction,
        config.minTrainPerUser,
        'last-write-wins by (event_time,event_seq)',
        'per-user chronological tail',
        train.length,
        val.length,
      );
    const splitId = Number(info.lastInsertRowid);
    const insert = db.prepare(
      'INSERT INTO split_ratings(split_id,user_id,item_id,rating,event_time,event_seq,role) VALUES (?,?,?,?,?,?,?)',
    );
    for (const r of train) {
      insert.run(splitId, r.user_id, r.item_id, r.rating, r.event_time, r.event_seq, 'train');
    }
    for (const r of val) {
      insert.run(splitId, r.user_id, r.item_id, r.rating, r.event_time, r.event_seq, 'val');
    }
    return { splitId, train, val };
  });
}

export function loadSplit(
  db: DatabaseSync,
  splitId: number,
): { train: CurrentRating[]; val: CurrentRating[] } {
  const split = db.prepare('SELECT id FROM splits WHERE id=?').get(splitId);
  if (!split) throw validation(`split ${splitId} not found`);
  const read = (role: string) =>
    db
      .prepare(
        'SELECT user_id,item_id,rating,event_time,event_seq FROM split_ratings WHERE split_id=? AND role=?',
      )
      .all(splitId, role) as CurrentRating[];
  return { train: read('train'), val: read('val') };
}
