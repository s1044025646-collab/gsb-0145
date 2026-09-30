import { Rng } from './rng.ts';
import type { CurrentRating, PredictionContribution, TrainConfig } from './types.ts';

/**
 * Biased matrix factorization trained by explicit, deterministic SGD on the
 * squared error of OBSERVED ratings only. Missing ratings are never treated as
 * zero and contribute neither loss nor gradient.
 *
 * Prediction:  pred = mu + bu + bi + pu . qi
 * Loss:        sum (r - pred)^2
 *              + regBias * (bu^2 + bi^2)
 *              + regFactor * (||pu||^2 + ||qi||^2)
 *
 * Update order is fixed: ratings sorted by (user,item,event_time,event_seq),
 * and for each example biases before factors, factor indices ascending.
 */

export interface TrainStats {
  epochLosses: number[];
  finalLoss: number;
  observedCount: number;
}

export interface FactorModelData {
  globalMean: number;
  userBias: Map<string, number>;
  itemBias: Map<string, number>;
  userFactors: Map<string, number[]>;
  itemFactors: Map<string, number[]>;
  stats: TrainStats;
}

export interface FallbackStats {
  globalMean: number;
  itemMean: Map<string, number>;
  userBiasMean: Map<string, number>;
}

export function sortedExamples(ratings: CurrentRating[]): CurrentRating[] {
  return [...ratings].sort(
    (a, b) =>
      a.user_id.localeCompare(b.user_id) ||
      a.item_id.localeCompare(b.item_id) ||
      a.event_time - b.event_time ||
      a.event_seq - b.event_seq,
  );
}

export function trainMatrixFactorization(
  train: CurrentRating[],
  config: TrainConfig,
): FactorModelData {
  if (train.length === 0) {
    throw new Error('cannot train on empty training set');
  }
  const examples = sortedExamples(train);
  const users = new Set<string>();
  const items = new Set<string>();
  let sum = 0;
  for (const r of examples) {
    users.add(r.user_id);
    items.add(r.item_id);
    sum += r.rating;
  }
  const globalMean = sum / examples.length;

  const rng = new Rng(config.seed);
  const userBias = new Map<string, number>();
  const itemBias = new Map<string, number>();
  const userFactors = new Map<string, number[]>();
  const itemFactors = new Map<string, number[]>();
  // Deterministic key order for initialization.
  for (const user of [...users].sort()) {
    userBias.set(user, 0);
    userFactors.set(user, Array.from({ length: config.factors }, () => rng.small()));
  }
  for (const item of [...items].sort()) {
    itemBias.set(item, 0);
    itemFactors.set(item, Array.from({ length: config.factors }, () => rng.small()));
  }

  const epochLosses: number[] = [];
  for (let epoch = 0; epoch < config.epochs; epoch++) {
    let squaredError = 0;
    for (const r of examples) {
      const pu = userFactors.get(r.user_id)!;
      const qi = itemFactors.get(r.item_id)!;
      let dot = 0;
      for (let f = 0; f < config.factors; f++) dot += pu[f]! * qi[f]!;
      const bu = userBias.get(r.user_id)!;
      const bi = itemBias.get(r.item_id)!;
      const pred = globalMean + bu + bi + dot;
      const err = r.rating - pred;
      squaredError += err * err;

      // Biases first.
      const newBu = bu + config.learningRate * (err - config.regBias * bu);
      const newBi = bi + config.learningRate * (err - config.regBias * bi);
      userBias.set(r.user_id, newBu);
      itemBias.set(r.item_id, newBi);

      // Factors ascending; simultaneous update via saved old values.
      const oldPu = pu.slice();
      const oldQi = qi.slice();
      for (let f = 0; f < config.factors; f++) {
        pu[f] = oldPu[f]! + config.learningRate * (err * oldQi[f]! - config.regFactor * oldPu[f]!);
        qi[f] = oldQi[f]! + config.learningRate * (err * oldPu[f]! - config.regFactor * oldQi[f]!);
      }
    }
    epochLosses.push(squaredError);
  }

  return {
    globalMean,
    userBias,
    itemBias,
    userFactors,
    itemFactors,
    stats: {
      epochLosses,
      finalLoss: epochLosses[epochLosses.length - 1]!,
      observedCount: examples.length,
    },
  };
}

/** Training-only fallback statistics for cold start (never touches validation). */
export function buildFallback(train: CurrentRating[]): FallbackStats {
  const globalSum = train.reduce((acc, r) => acc + r.rating, 0);
  const globalMean = globalSum / train.length;
  const itemSum = new Map<string, number>();
  const itemCount = new Map<string, number>();
  for (const r of train) {
    itemSum.set(r.item_id, (itemSum.get(r.item_id) ?? 0) + r.rating);
    itemCount.set(r.item_id, (itemCount.get(r.item_id) ?? 0) + 1);
  }
  const itemMean = new Map<string, number>();
  for (const [item, sum] of itemSum) {
    itemMean.set(item, sum / itemCount.get(item)!);
  }
  // Average residual per user after global mean and item mean: pure training data.
  const userRes = new Map<string, number>();
  const userResCount = new Map<string, number>();
  for (const r of train) {
    const residual = r.rating - globalMean - (itemMean.get(r.item_id)! - globalMean);
    userRes.set(r.user_id, (userRes.get(r.user_id) ?? 0) + residual);
    userResCount.set(r.user_id, (userResCount.get(r.user_id) ?? 0) + 1);
  }
  const userBiasMean = new Map<string, number>();
  for (const [user, res] of userRes) {
    userBiasMean.set(user, res / userResCount.get(user)!);
  }
  return { globalMean, itemMean, userBiasMean };
}

export type KnownKind = 'known' | 'cold_user' | 'cold_item' | 'cold_both';

export function coldKind(
  data: FactorModelData,
  userId: string,
  itemId: string,
): KnownKind {
  const knownUser = data.userBias.has(userId);
  const knownItem = data.itemBias.has(itemId);
  if (knownUser && knownItem) return 'known';
  if (!knownUser && !knownItem) return 'cold_both';
  return knownUser ? 'cold_item' : 'cold_user';
}

export interface ExplainResult {
  prediction: number;
  contribution: PredictionContribution;
  kind: KnownKind;
}

/** Explain a score with bias/factor contributions and training-only fallback. */
export function explain(
  data: FactorModelData,
  fallback: FallbackStats,
  userId: string,
  itemId: string,
): ExplainResult {
  const kind = coldKind(data, userId, itemId);
  const globalMean = data.globalMean;
  if (kind === 'known') {
    const userBias = data.userBias.get(userId)!;
    const itemBias = data.itemBias.get(itemId)!;
    const pu = data.userFactors.get(userId)!;
    const qi = data.itemFactors.get(itemId)!;
    let dot = 0;
    for (let f = 0; f < pu.length; f++) dot += pu[f]! * qi[f]!;
    return {
      prediction: globalMean + userBias + itemBias + dot,
      contribution: { globalMean, userBias, itemBias, dot, source: 'model' },
      kind,
    };
  }

  const itemMean = fallback.itemMean.get(itemId);
  const userBiasFallback = fallback.userBiasMean.get(userId) ?? 0;
  const itemBias = itemMean === undefined ? 0 : itemMean - globalMean;
  const source: PredictionContribution['source'] =
    kind === 'cold_both' ? 'fallback' : 'mixed';
  const prediction =
    kind === 'cold_both'
      ? globalMean
      : kind === 'cold_item'
        ? globalMean + userBiasFallback
        : globalMean + itemBias;
  return {
    prediction,
    contribution: {
      globalMean,
      userBias: kind === 'cold_item' ? userBiasFallback : 0,
      itemBias: kind === 'cold_user' ? itemBias : 0,
      dot: 0,
      source,
    },
    kind,
  };
}
