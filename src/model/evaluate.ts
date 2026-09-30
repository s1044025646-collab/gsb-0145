import { explain } from './mf.ts';
import type { StoredModel } from './serialize.ts';
import type { CurrentRating, EvalReport } from './types.ts';

/** RMSE on validation pairs; null when the validation set is empty. */
export function rmse(model: StoredModel, val: CurrentRating[]): number | null {
  if (val.length === 0) return null;
  let sumSq = 0;
  for (const r of val) {
    const { prediction } = explain(model.data, model.fallback, r.user_id, r.item_id);
    const diff = r.rating - prediction;
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq / val.length);
}

/** Relevance for NDCG: rating above the global midpoint of the rating scale. */
export function isRelevant(rating: number): boolean {
  return rating > 3;
}

function dcg(gains: number[]): number {
  let value = 0;
  gains.forEach((gain, index) => {
    value += gain / Math.log2(index + 2);
  });
  return value;
}

export interface NdcgResult {
  value: number | null;
  usersConsidered: number;
  usersEmptyRelevant: number;
  usersEmptyCandidates: number;
}

/**
 * NDCG@K over a FIXED candidate set per user: the union of that user's
 * validation items plus all training items (other users' items), so every
 * validation item is rankable. Already-seen training items are excluded from
 * ranking. Users with an empty relevant set are excluded from the mean and
 * reported separately; users with no candidate items contribute a note.
 */
export function ndcgAtK(
  model: StoredModel,
  val: CurrentRating[],
  k: number,
): NdcgResult {
  const valByUser = new Map<string, CurrentRating[]>();
  for (const r of val) {
    const list = valByUser.get(r.user_id) ?? [];
    list.push(r);
    valByUser.set(r.user_id, list);
  }
  // Fixed universe: every item present in the training split.
  const trainItemsByUser = new Map<string, Set<string>>();
  const allTrainItems = new Set<string>();
  for (const r of model.train) {
    allTrainItems.add(r.item_id);
    const set = trainItemsByUser.get(r.user_id) ?? new Set<string>();
    set.add(r.item_id);
    trainItemsByUser.set(r.user_id, set);
  }

  let sum = 0;
  let usersConsidered = 0;
  let usersEmptyRelevant = 0;
  let usersEmptyCandidates = 0;

  for (const [userId, rows] of valByUser) {
    const relevantItems = rows.filter((r) => isRelevant(r.rating));
    if (relevantItems.length === 0) {
      usersEmptyRelevant++;
      continue;
    }
    const seen = trainItemsByUser.get(userId) ?? new Set<string>();
    const valItems = new Set(rows.map((r) => r.item_id));
    const candidates = [...allTrainItems, ...valItems]
      .filter((item) => !seen.has(item))
      .sort();
    if (candidates.length === 0) {
      usersEmptyCandidates++;
      continue;
    }
    const relevance = new Map(rows.map((r) => [r.item_id, r.rating]));
    const ranked = candidates
      .map((itemId) => ({
        itemId,
        score: explain(model.data, model.fallback, userId, itemId).prediction,
      }))
      .sort((a, b) => b.score - a.score || a.itemId.localeCompare(b.itemId))
      .slice(0, k);

    const gains = ranked.map((s) =>
      isRelevant(relevance.get(s.itemId) ?? 0) ? 1 : 0,
    );
    const idealGains = relevantItems
      .slice()
      .sort((a, b) => b.rating - a.rating || a.item_id.localeCompare(b.item_id))
      .slice(0, k)
      .map(() => 1);
    const idcg = dcg(idealGains);
    sum += dcg(gains) / idcg;
    usersConsidered++;
  }

  return {
    value: usersConsidered === 0 ? null : sum / usersConsidered,
    usersConsidered,
    usersEmptyRelevant,
    usersEmptyCandidates,
  };
}

export function evaluateModel(
  model: StoredModel,
  val: CurrentRating[],
  k: number,
): EvalReport {
  const notes: string[] = [];
  const ndcg = ndcgAtK(model, val, k);
  if (val.length === 0) notes.push('validation set empty: RMSE and NDCG not defined (null)');
  if (ndcg.usersEmptyRelevant > 0)
    notes.push(`${ndcg.usersEmptyRelevant} user(s) had empty relevant sets and were excluded from NDCG`);
  if (ndcg.usersEmptyCandidates > 0)
    notes.push(`${ndcg.usersEmptyCandidates} user(s) had no candidate items after seen exclusion`);
  notes.push('NDCG uses a fixed candidate set: validation items union all training items');
  notes.push('low-rank similarity is correlational, not causal');
  return {
    modelId: model.id,
    splitId: model.splitId,
    rmse: rmse(model, val),
    ndcgAtK: ndcg.value,
    k,
    notes,
    evaluatedAt: Date.now(),
  };
}
