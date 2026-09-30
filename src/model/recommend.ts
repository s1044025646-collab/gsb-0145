import { explain } from './mf.ts';
import type { StoredModel } from './serialize.ts';
import type { ScoredItem } from './types.ts';

export interface RecommendOptions {
  k: number;
  /**
   * Candidate universe. Defaults to all items known in training. Already-seen
   * items (rated in the training split) are always excluded.
   */
  candidates?: string[];
  /**
   * Exclusion scope: 'train' excludes only training-seen items (default for
   * offline eval so validation targets remain rankable); 'all_current'
   * excludes every item currently rated by the user.
   */
  exclude?: 'train' | 'all_current';
  currentlySeen?: Set<string>;
}

/**
 * Top-K with deterministic tie-breaking: descending score, then ascending
 * itemId. Returns an empty list when no candidates remain.
 */
export function topK(model: StoredModel, userId: string, options: RecommendOptions): ScoredItem[] {
  if (!Number.isInteger(options.k) || options.k <= 0) {
    throw new Error('k must be a positive integer');
  }
  const trainedItems = new Set(
    model.train.filter((r) => r.user_id === userId).map((r) => r.item_id),
  );
  const excludeSet =
    options.exclude === 'all_current'
      ? (options.currentlySeen ?? trainedItems)
      : trainedItems;
  const universe =
    options.candidates ?? [...model.data.itemBias.keys()].sort();

  const scored: ScoredItem[] = [];
  for (const itemId of universe) {
    if (excludeSet.has(itemId)) continue;
    const { prediction } = explain(model.data, model.fallback, userId, itemId);
    scored.push({ itemId, score: prediction, tieBreak: itemId });
  }
  scored.sort((a, b) => b.score - a.score || a.tieBreak.localeCompare(b.tieBreak));
  return scored.slice(0, options.k);
}
