import type { Db } from './db.js';
import { ApiError } from './errors.js';
import { getSplitRatings } from './repository.js';
import { DEFAULT_CONFIG, explainPrediction, ndcgAtK, recommend, rmse, trainModel } from './recommender.js';
import type { EpochLoss, PredictionExplanation, RatingRecord, ScoredItem, TrainedModel, TrainConfig } from './types.js';
import { catalogItemIdsAtSplit } from './catalog.js';

interface ModelRow {
  id: number;
  split_id: number;
  config_json: string;
  global_mean: number;
  losses_json: string;
}

export function trainAndStoreModel(db: Db, splitId: number, overrides: Partial<TrainConfig> = {}): { modelId: number; losses: EpochLoss[] } {
  const split = getSplitRatings(db, splitId);
  const config = { ...DEFAULT_CONFIG, ...overrides, validationCount: split.validationCount };
  const { model, losses } = trainModel(split.train, config);
  db.exec('BEGIN IMMEDIATE;');
  try {
    const inserted = db
      .prepare('INSERT INTO models(split_id,config_json,global_mean,losses_json) VALUES (?,?,?,?)')
      .run(splitId, JSON.stringify(config), model.globalMean, JSON.stringify(losses));
    const modelId = Number(inserted.lastInsertRowid);
    const factorStatement = db.prepare('INSERT INTO model_factors(model_id,kind,key,bias,vector_json) VALUES (?,?,?,?,?)');
    for (const [key, factor] of model.users) factorStatement.run(modelId, 'user', key, factor.bias, JSON.stringify(factor.vector));
    for (const [key, factor] of model.items) factorStatement.run(modelId, 'item', key, factor.bias, JSON.stringify(factor.vector));
    db.exec('COMMIT;');
    return { modelId, losses };
  } catch (error) {
    db.exec('ROLLBACK;');
    throw error;
  }
}

export function loadModel(db: Db, modelId: number): TrainedModel & { modelId: number; splitId: number } {
  const row = db.prepare('SELECT * FROM models WHERE id = ? AND archived_at IS NULL').get(modelId) as ModelRow | undefined;
  if (!row) throw new ApiError(404, 'MODEL_NOT_FOUND', `model ${modelId} not found`);
  const factorRows = db.prepare('SELECT * FROM model_factors WHERE model_id = ?').all(modelId) as Array<{
    kind: 'user' | 'item';
    key: string;
    bias: number;
    vector_json: string;
  }>;
  const users = new Map<string, { bias: number; vector: number[] }>();
  const items = new Map<string, { bias: number; vector: number[] }>();
  for (const factor of factorRows) {
    const value = { bias: factor.bias, vector: JSON.parse(factor.vector_json) as number[] };
    (factor.kind === 'user' ? users : items).set(factor.key, value);
  }
  return { modelId: row.id, splitId: row.split_id, globalMean: row.global_mean, users, items, config: JSON.parse(row.config_json) as TrainConfig };
}

export function predictWithModel(db: Db, modelId: number, userId: string, itemId: string): PredictionExplanation {
  const model = loadModel(db, modelId);
  return { ...explainPrediction(model, userId, itemId), modelVersion: modelId };
}

export function recommendWithModel(db: Db, modelId: number, userId: string, topK: number, excludeSeen = true): ScoredItem[] {
  const model = loadModel(db, modelId);
  const split = getSplitRatings(db, model.splitId);
  const candidates = new Set<string>(catalogItemIdsAtSplit(db, model.splitId));
  const seen = new Set(split.train.filter((row) => row.userId === userId).map((row) => row.itemId));
  return recommend(model, userId, candidates, seen, topK, excludeSeen);
}

function fixedCandidates(split: { train: RatingRecord[]; validation: RatingRecord[] }, userId: string, model: TrainedModel): string[] {
  const itemIds = new Set<string>([...model.items.keys(), ...split.train.map((row) => row.itemId)]);
  for (const row of split.validation) if (row.userId === userId) itemIds.add(row.itemId);
  return [...itemIds].sort();
}

export function evaluateAndStoreReport(db: Db, modelId: number, k = 5) {
  const model = loadModel(db, modelId);
  const split = getSplitRatings(db, model.splitId);
  const predictions = split.validation.map((row) => ({
    ...row,
    predicted: explainPrediction(model, row.userId, row.itemId).score,
  }));
  const score = rmse(predictions.map((row) => ({ actual: row.rating, predicted: row.predicted })));

  const rankingRows: Array<{ userId: string; itemId: string; actual: number; predicted: number }> = [];
  for (const userId of [...new Set(split.validation.map((row) => row.userId))].sort()) {
    const seen = new Set(split.train.filter((row) => row.userId === userId).map((row) => row.itemId));
    for (const itemId of fixedCandidates(split, userId, model).filter((id) => !seen.has(id))) {
      const validation = split.validation.find((row) => row.userId === userId && row.itemId === itemId);
      rankingRows.push({ userId, itemId, actual: validation?.rating ?? 0, predicted: explainPrediction(model, userId, itemId).score });
    }
  }
  const ndcg = ndcgAtK(rankingRows, k);
  const details = {
    k,
    predictions,
    fixedCandidatePolicy: 'all items present in the immutable split catalog, plus validation items; training-seen items excluded for each user',
  };
  db.prepare('INSERT OR REPLACE INTO reports(model_id,rmse,ndcg_json,details_json) VALUES (?,?,?,?)').run(
    modelId,
    score,
    JSON.stringify(ndcg),
    JSON.stringify(details),
  );
  return { rmse: score, ndcg, details };
}

export function compareModels(db: Db, modelIds: number[], k = 5) {
  return modelIds.map((modelId) => {
    const model = loadModel(db, modelId);
    const row = db.prepare('SELECT * FROM reports WHERE model_id = ?').get(modelId) as
      | { rmse: number | null; ndcg_json: string; details_json: string }
      | undefined;
    if (!row) evaluateAndStoreReport(db, modelId, k);
    const report = db.prepare('SELECT * FROM reports WHERE model_id = ?').get(modelId) as {
      rmse: number | null;
      ndcg_json: string;
      details_json: string;
    };
    return {
      modelId,
      config: model.config,
      rmse: report.rmse,
      ndcg: JSON.parse(report.ndcg_json ?? 'null'),
    };
  });
}
