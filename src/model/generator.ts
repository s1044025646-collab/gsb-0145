import { Rng } from './rng.ts';
import type { RatingEvent } from './types.ts';

export interface GeneratorOptions {
  users: number;
  items: number;
  /** Approx fraction of the full user-item grid that gets rated. */
  density: number;
  seed: number;
  /** Earliest event time (ms); defaults injectable for tests. */
  startTime: number;
  /** Span of event timestamps in milliseconds. */
  timeSpanMs: number;
  /** Add a sparse user who rates very few items. */
  includeSparseUser?: boolean;
}

/**
 * Fully synthetic ratings generated from a KNOWN low-rank latent structure
 * plus biases and noise. Timestamps are synthetic too; nothing here reflects
 * real people.
 */
export function generateRatings(options: GeneratorOptions): RatingEvent[] {
  if (
    options.users <= 0 ||
    options.items <= 0 ||
    options.density <= 0 ||
    options.density > 1
  ) {
    throw new Error('invalid generator options');
  }
  const rng = new Rng(options.seed);
  const latentD = 2;
  const userBias = Array.from({ length: options.users }, () => rng.uniform(-0.6, 0.6));
  const itemBias = Array.from({ length: options.items }, () => rng.uniform(-0.8, 0.8));
  const pu = Array.from({ length: options.users }, () =>
    Array.from({ length: latentD }, () => rng.uniform(-1, 1)),
  );
  const qi = Array.from({ length: options.items }, () =>
    Array.from({ length: latentD }, () => rng.uniform(-1, 1)),
  );
  const base = 3;

  const events: RatingEvent[] = [];
  for (let u = 0; u < options.users; u++) {
    const sparse = options.includeSparseUser && u === options.users - 1;
    for (let i = 0; i < options.items; i++) {
      const keep = sparse ? i < 1 : rng.next() < options.density;
      if (!keep) continue;
      let dot = 0;
      for (let f = 0; f < latentD; f++) dot += pu[u]![f]! * qi[i]![f]!;
      const noise = rng.uniform(-0.3, 0.3);
      const raw = base + userBias[u]! + itemBias[i]! + dot * 0.5 + noise;
      const rating = Math.min(5, Math.max(1, Math.round(raw * 2) / 2));
      events.push({
        userId: `u${u}`,
        itemId: `i${i}`,
        rating,
        eventTime:
          options.startTime +
          Math.floor(rng.next() * options.timeSpanMs),
        idempotencyKey: `gen-${options.seed}-${u}-${i}`,
      });
    }
  }
  return events;
}
