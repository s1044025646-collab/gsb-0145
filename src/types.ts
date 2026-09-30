export type IsoDateTime = string;

export interface RatingInput {
  userId: string;
  itemId: string;
  rating: number;
  eventTime: IsoDateTime;
  idempotencyKey?: string;
}

export interface RatingRecord extends RatingInput {
  id: number;
  createdAt: IsoDateTime;
  active: boolean;
}

export interface Split {
  train: RatingRecord[];
  validation: RatingRecord[];
  cutoffTime: IsoDateTime;
  validationCount: number;
}

export interface TrainConfig {
  factors: number;
  epochs: number;
  learningRate: number;
  regularization: number;
  seed: number;
  validationCount: number;
  minRating: number;
  maxRating: number;
}

export interface FactorVector {
  bias: number;
  vector: number[];
}

export interface TrainedModel {
  globalMean: number;
  users: Map<string, FactorVector>;
  items: Map<string, FactorVector>;
  config: TrainConfig;
}

export interface EpochLoss {
  epoch: number;
  sse: number;
  rmse: number;
  objective: number;
}

export interface PredictionExplanation {
  userId: string;
  itemId: string;
  score: number;
  unclampedScore: number;
  contributions: {
    globalMean: number;
    userBias: number;
    itemBias: number;
    interaction: number;
  };
  fallbackSources: string[];
  fallback: boolean;
  modelVersion?: number;
  note: string;
}

export interface ScoredItem {
  itemId: string;
  score: number;
  rank: number;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}
