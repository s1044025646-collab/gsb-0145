import type { DatabaseSync } from 'node:sqlite';
import { loadSplit, createSplit } from './split.ts';
import { trainMatrixFactorization } from './mf.ts';
import { loadModel, saveModel } from './serialize.ts';
import { evaluateModel } from './evaluate.ts';
import type { EvalReport, TrainConfig } from './types.ts';
import { validation } from './errors.ts';

export function validateConfig(config: TrainConfig): void {
  const checks: Array<[boolean, string]> = [
    [Number.isInteger(config.factors) && config.factors > 0, 'factors must be a positive integer'],
    [Number.isInteger(config.epochs) && config.epochs > 0, 'epochs must be a positive integer'],
    [config.learningRate > 0, 'learningRate must be > 0'],
    [config.regBias >= 0, 'regBias must be >= 0'],
    [config.regFactor >= 0, 'regFactor must be >= 0'],
    [Number.isInteger(config.seed), 'seed must be an integer'],
    [config.valFraction >= 0 && config.valFraction < 1, 'valFraction in [0,1)'],
    [
      Number.isInteger(config.minTrainPerUser) && config.minTrainPerUser >= 0,
      'minTrainPerUser must be a non-negative integer',
    ],
  ];
  for (const [ok, message] of checks) if (!ok) throw validation(message);
}

export function trainFromCurrent(
  db: DatabaseSync,
  config: TrainConfig,
): { modelId: number; splitId: number; epochLosses: number[] } {
  validateConfig(config);
  const { splitId, train } = createSplit(db, config);
  const data = trainMatrixFactorization(train, config);
  const modelId = saveModel(db, splitId, config, train, data);
  return { modelId, splitId, epochLosses: data.stats.epochLosses };
}

export function trainOnExistingSplit(
  db: DatabaseSync,
  splitId: number,
  config: TrainConfig,
): { modelId: number; epochLosses: number[] } {
  validateConfig(config);
  const { train } = loadSplit(db, splitId);
  const data = trainMatrixFactorization(train, config);
  const modelId = saveModel(db, splitId, config, train, data);
  return { modelId, epochLosses: data.stats.epochLosses };
}

export function evaluateAndStore(db: DatabaseSync, modelId: number, k: number): EvalReport {
  if (!Number.isInteger(k) || k <= 0) throw validation('k must be a positive integer');
  const model = loadModel(db, modelId);
  const { val } = loadSplit(db, model.splitId);
  const report = evaluateModel(model, val, k);
  db.prepare(
    'INSERT INTO reports(model_id,split_id,evaluated_at,k,rmse,ndcg_at_k,notes) VALUES (?,?,?,?,?,?,?)',
  ).run(
    report.modelId,
    report.splitId,
    report.evaluatedAt,
    report.k,
    report.rmse,
    report.ndcgAtK,
    JSON.stringify(report.notes),
  );
  return report;
}

export function compareModels(
  db: DatabaseSync,
  modelIds: number[],
  k: number,
): EvalReport[] {
  return modelIds.map((id) => {
    const model = loadModel(db, id);
    const { val } = loadSplit(db, model.splitId);
    return evaluateModel(model, val, k);
  });
}
