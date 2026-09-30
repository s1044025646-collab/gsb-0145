import type { Db } from './db.js';

export function catalogItemIdsAtSplit(db: Db, splitId: number): string[] {
  const rows = db
    .prepare(
      `SELECT DISTINCT e.item_id
         FROM rating_events e
         JOIN split_ratings s ON s.rating_event_id = e.id
        WHERE s.split_id = ?
        ORDER BY e.item_id`,
    )
    .all(splitId) as unknown as Array<{ item_id: string }>;
  return rows.map((row) => row.item_id);
}
