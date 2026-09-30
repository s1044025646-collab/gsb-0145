import type { DatabaseSync } from 'node:sqlite';
import { notFound } from './errors.ts';
import {
  buildFallback,
  explain,
  type FactorModelData,
  type FallbackStats as FallbackData,
} from './mf.ts';
import type { CurrentRating, TrainConfig } from './types.ts';

export interface StoredModel {
  id: number;
  splitId: number;
  config: TrainConfig;
  data: FactorModelData;
  fallback: FallbackData;
  train: CurrentRating[];
}

export function saveModel(
  db: DatabaseSync,
  splitId: number,
  config: TrainConfig,
  train: CurrentRating[],
  data: FactorModelData,
  note?: string,
): number {
  const result = db
    .prepare(
      `INSERT INTO models(split_id,created_at,factors,epochs,learning_rate,reg_bias,reg_factor,seed,global_mean,update_order,final_train_loss,note)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`,
    )
    .run(
      splitId,
      Date.now(),
      config.factors,
      config.epochs,
      config.learningRate,
      config.regBias,
      config.regFactor,
      config.seed,
      data.globalMean,
      'sorted (user,item,event_time,event_seq); biases then factors',
      data.stats.finalLoss,
      note ?? null,
    );
  const modelId = Number(result.lastInsertRowid);

  const putBias = db.prepare(
    'INSERT INTO model_biases(model_id,kind,key,bias) VALUES (?,?,?,?)',
  );
  for (const [key, bias] of data.userBias) putBias.run(modelId, 'user', key, bias);
  for (const [key, bias] of data.itemBias) putBias.run(modelId, 'item', key, bias);

  const putFactor = db.prepare(
    'INSERT INTO model_factors(model_id,kind,key,factor_index,value) VALUES (?,?,?,?,?)',
  );
  for (const [key, vec] of data.userFactors) {
    vec.forEach((value, index) => putFactor.run(modelId, 'user', key, index, value));
  }
  for (const [key, vec] of data.itemFactors) {
    vec.forEach((value, index) => putFactor.run(modelId, 'item', key, index, value));
  }

  const putLoss = db.prepare(
    'INSERT INTO epoch_losses(model_id,epoch,train_loss) VALUES (?,?,?)',
  );
  data.stats.epochLosses.forEach((loss, epoch) =>
    putLoss.run(modelId, epoch, loss),
  );

  // Persist training-only fallback compactly in the model note envelope is not
  // needed: fallback can be rebuilt deterministically from split_ratings.
  return modelId;
}

export function listModels(db: DatabaseSync): Record<string, unknown>[] {
  return db
    .prepare('SELECT * FROM models ORDER BY id')
    .all() as Record<string, unknown>[];
}

export function loadModel(db: DatabaseSync, modelId: number): StoredModel {
  const row = db.prepare('SELECT * FROM models WHERE id=?').get(modelId) as
    | Record<string, unknown>
    | undefined;
  if (!row) throw notFound(`model ${modelId} not found`);

  const config: TrainConfig = {
    valFraction: 0,
    factors: Number(row.factors),
    epochs: Number(row.epochs),
    learningRate: Number(row.learning_rate),
    regBias: Number(row.reg_bias),
    regFactor: Number(row.reg_factor),
    seed: Number(row.seed),
    minTrainPerUser: 1,
  };
  const splitId = Number(row.split_id);
  const splitRow = db
    .prepare('SELECT val_fraction,min_train_per_user FROM splits WHERE id=?')
    .get(splitId) as { val_fraction: number; min_train_per_user: number };
  config.valFraction = Number(splitRow.val_fraction);
  config.minTrainPerUser = Number(splitRow.min_train_per_user);

  const train = db
    .prepare(
      'SELECT user_id,item_id,rating,event_time,event_seq FROM split_ratings WHERE split_id=? AND role=\'train\'',
    )
    .all(splitId) as CurrentRating[];

  const globalMean = Number(row.global_mean);
  const userBias = new Map<string, number>();
  const itemBias = new Map<string, number>();
  for (const b of db
    .prepare('SELECT kind,key,bias FROM model_biases WHERE model_id=?')
    .all(modelId) as { kind: string; key: string; bias: number }[]) {
    (b.kind === 'user' ? userBias : itemBias).set(b.key, b.bias);
  }
  const userFactors = new Map<string, number[]>();
  const itemFactors = new Map<string, number[]>();
  for (const f of db
    .prepare('SELECT kind,key,factor_index,value FROM model_factors WHERE model_id=?')
    .all(modelId) as {
      kind: string;
      key: string;
      factor_index: number;
      value: number;
    }[]) {
    const target = f.kind === 'user' ? userFactors : itemFactors;
    const vec = target.get(f.key) ?? [];
    vec[Number(f.factor_index)] = f.value;
    target.set(f.key, vec);
  }
  const epochLosses = (
    db
      .prepare('SELECT train_loss FROM epoch_losses WHERE model_id=? ORDER BY epoch')
      .all(modelId) as { train_loss: number }[]
  ).map((r) => Number(r.train_loss));

  const data: FactorModelData = {
    globalMean,
    userBias,
    itemBias,
    userFactors,
    itemFactors,
    stats: {
      epochLosses,
      finalLoss: epochLosses[epochLosses.length - 1] ?? NaN,
      observedCount: train.length,
    },
  };
  const fallback = buildFallback(train);
  return { id: modelId, splitId, config, data, fallback, train };
}

export { explain };
