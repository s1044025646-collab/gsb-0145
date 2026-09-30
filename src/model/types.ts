/** Shared domain types. */

export interface RatingEvent {
  userId: string;
  itemId: string;
  rating: number;
  eventTime: number;
  /** Client-supplied key; submitting the same key twice is a no-op. */
  idempotencyKey?: string;
}

/** Current (winning) rating for a (user, item) pair after revision rule. */
export interface CurrentRating {
  user_id: string;
  item_id: string;
  rating: number;
  event_time: number;
  event_seq: number;
}

export interface TrainConfig {
  /** Fraction of each user's chronologically latest ratings used for validation, in [0,1). */
  valFraction: number;
  factors: number;
  epochs: number;
  learningRate: number;
  regBias: number;
  regFactor: number;
  seed: number;
  /** Minimum number of ratings a user needs before any rating can move to validation. */
  minTrainPerUser: number;
}

export const DEFAULT_CONFIG: TrainConfig = {
  valFraction: 0.2,
  factors: 4,
  epochs: 20,
  learningRate: 0.01,
  regBias: 0.02,
  regFactor: 0.05,
  seed: 42,
  minTrainPerUser: 1,
};

export interface PredictionContribution {
  globalMean: number;
  userBias: number;
  itemBias: number;
  dot: number;
  /**
   * Human-readable provenance of the numeric parts.
   * `learned` parts come from training data only; `fallback` parts are
   * training-only averages used for cold start.
   */
  source: 'model' | 'fallback' | 'mixed';
}

export interface ScoredItem {
  itemId: string;
  score: number;
  /** Rank tie-break key; same scores always produce a stable order. */
  tieBreak: string;
}

export interface EvalReport {
  modelId: number;
  splitId: number;
  rmse: number | null;
  ndcgAtK: number | null;
  k: number;
  /** How empty-relevant-set and empty-candidate-set cases were handled. */
  notes: string[];
  evaluatedAt: number;
}
