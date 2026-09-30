import { seededNormal } from './prng.js';
import type {
  EpochLoss,
  PredictionExplanation,
  RatingRecord,
  ScoredItem,
  TrainedModel,
  TrainConfig,
} from './types.js';

export const DEFAULT_CONFIG: TrainConfig = {
  factors: 4,
  epochs: 20,
  learningRate: 0.02,
  regularization: 0.05,
  seed: 1729,
  validationCount: 1,
  minRating: 1,
  maxRating: 5,
};

function stableTrainingOrder(ratings: RatingRecord[]): RatingRecord[] {
  return [...ratings].sort((a, b) =>
    a.userId < b.userId ? -1 : a.userId > b.userId ? 1 : a.itemId < b.itemId ? -1 : a.itemId > b.itemId ? 1 : a.id - b.id,
  );
}

export function trainModel(ratings: RatingRecord[], supplied: Partial<TrainConfig> = {}): { model: TrainedModel; losses: EpochLoss[] } {
  const config: TrainConfig = { ...DEFAULT_CONFIG, ...supplied };
  if (ratings.length === 0) throw new Error('training data is empty');
  const globalMean = ratings.reduce((sum, row) => sum + row.rating, 0) / ratings.length;
  const userIds = [...new Set(ratings.map((row) => row.userId))].sort();
  const itemIds = [...new Set(ratings.map((row) => row.itemId))].sort();
  const users = new Map(userIds.map((id) => [id, { bias: 0, vector: Array.from({ length: config.factors }, () => 0) }]));
  const items = new Map(itemIds.map((id) => [id, { bias: 0, vector: Array.from({ length: config.factors }, () => 0) }]));
  const ordered = stableTrainingOrder(ratings);
  const normal = seededNormal(config.seed);
  for (const factor of items.values()) for (let index = 0; index < config.factors; index += 1) factor.vector[index] = normal() * 0.01;

  const losses: EpochLoss[] = [];
  for (let epoch = 1; epoch <= config.epochs; epoch += 1) {
    let sse = 0;
    for (const row of ordered) {
      const user = users.get(row.userId)!;
      const item = items.get(row.itemId)!;
      const interaction = dot(user.vector, item.vector);
      const prediction = globalMean + user.bias + item.bias + interaction;
      const error = row.rating - prediction;
      sse += error * error;
      user.bias += config.learningRate * (error - config.regularization * user.bias);
      item.bias += config.learningRate * (error - config.regularization * item.bias);
      for (let index = 0; index < config.factors; index += 1) {
        const userFactor = user.vector[index]!;
        const itemFactor = item.vector[index]!;
        user.vector[index] = userFactor + config.learningRate * (error * itemFactor - config.regularization * userFactor);
        item.vector[index] = itemFactor + config.learningRate * (error * userFactor - config.regularization * itemFactor);
      }
    }
    let regularizationTerm = 0;
    for (const factor of [...users.values(), ...items.values()]) {
      regularizationTerm += factor.bias * factor.bias + dot(factor.vector, factor.vector);
    }
    losses.push({
      epoch,
      sse,
      rmse: Math.sqrt(sse / ordered.length),
      objective: sse + config.regularization * regularizationTerm,
    });
  }
  return { model: { globalMean, users, items, config }, losses };
}

export function dot(a: number[], b: number[]): number {
  let total = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) total += a[index]! * b[index]!;
  return total;
}

export function explainPrediction(model: TrainedModel, userId: string, itemId: string): PredictionExplanation {
  const user = model.users.get(userId);
  const item = model.items.get(itemId);
  const fallbackSources: string[] = [];
  let userBias = 0;
  let itemBias = 0;
  let interaction = 0;
  if (user) userBias = user.bias;
  else fallbackSources.push('unknown user: training global mean contributes no user bias');
  if (item) itemBias = item.bias;
  else fallbackSources.push('unknown item: training global mean contributes no item popularity');
  if (user && item) interaction = dot(user.vector, item.vector);
  else fallbackSources.push('cold-start: latent interaction unavailable');
  const unclamped = model.globalMean + userBias + itemBias + interaction;
  const score = clamp(unclamped, model.config.minRating, model.config.maxRating);
  return {
    userId,
    itemId,
    score,
    unclampedScore: unclamped,
    contributions: { globalMean: model.globalMean, userBias, itemBias, interaction },
    fallbackSources,
    fallback: fallbackSources.length > 0,
    note: 'Contributions are an additive score decomposition from observed training ratings; low-rank similarity is associative and not causal.',
  };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function recommend(
  model: TrainedModel,
  userId: string,
  candidateItemIds: Iterable<string>,
  seenItemIds: ReadonlySet<string>,
  limit: number,
  excludeSeen = true,
): ScoredItem[] {
  const candidates = [...candidateItemIds]
    .filter((itemId) => !excludeSeen || !seenItemIds.has(itemId))
    .map((itemId) => ({ itemId, prediction: explainPrediction(model, userId, itemId) }));
  candidates.sort((a, b) =>
    b.prediction.score - a.prediction.score ||
    b.prediction.unclampedScore - a.prediction.unclampedScore ||
    (a.itemId < b.itemId ? -1 : a.itemId > b.itemId ? 1 : 0),
  );
  return candidates.slice(0, limit).map((candidate, index) => ({ itemId: candidate.itemId, score: candidate.prediction.score, rank: index + 1 }));
}

export function rmse(predictions: Array<{ actual: number; predicted: number }>): number | null {
  if (predictions.length === 0) return null;
  return Math.sqrt(predictions.reduce((sum, row) => sum + (row.actual - row.predicted) ** 2, 0) / predictions.length);
}

function dcg(rows: Array<{ actual: number; predicted: number }>, k: number): number {
  return [...rows]
    .sort((a, b) => b.predicted - a.predicted)
    .slice(0, k)
    .reduce((sum, row, index) => sum + (2 ** row.actual - 1) / Math.log2(index + 2), 0);
}

export function ndcgAtK(
  rows: Array<{ userId: string; itemId: string; actual: number; predicted: number }>,
  k: number,
): { users: Record<string, number | null>; macroNdcg: number | null; evaluatedUsers: number; emptyRelevantUsers: string[]; emptyCandidateUsers: string[] } {
  const byUser = new Map<string, typeof rows>();
  for (const row of rows) {
    const list = byUser.get(row.userId) ?? [];
    list.push(row);
    byUser.set(row.userId, list);
  }
  const scores: Record<string, number | null> = {};
  const emptyRelevantUsers: string[] = [];
  const emptyCandidateUsers: string[] = [];
  let total = 0;
  let count = 0;
  for (const [userId, list] of [...byUser.entries()].sort(([a], [b]) => (a < b ? -1 : 1))) {
    if (list.length === 0) {
      scores[userId] = null;
      emptyCandidateUsers.push(userId);
      continue;
    }
    const ideal = dcg(list, k);
    if (ideal === 0) {
      scores[userId] = null;
      emptyRelevantUsers.push(userId);
      continue;
    }
    const score = dcg(list, k) / ideal;
    scores[userId] = score;
    total += score;
    count += 1;
  }
  return { users: scores, macroNdcg: count === 0 ? null : total / count, evaluatedUsers: count, emptyRelevantUsers, emptyCandidateUsers };
}
