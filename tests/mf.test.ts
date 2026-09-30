import { test } from 'node:test';
import assert from 'node:assert/strict';
import { trainMatrixFactorization, explain, buildFallback } from '../src/model/mf.ts';
import type { CurrentRating, TrainConfig } from '../src/model/types.ts';

function cr(user_id: string, item_id: string, r: number, t: number, seq = 1): CurrentRating {
  return { user_id, item_id, rating: r, event_time: t, event_seq: seq };
}

const config: TrainConfig = {
  valFraction: 0,
  factors: 2,
  epochs: 300,
  learningRate: 0.02,
  regBias: 0,
  regFactor: 0,
  seed: 1,
  minTrainPerUser: 1,
};

test('SGD reduces squared error on observed pairs only', () => {
  // Decomposable structure: rating = 3 + user tendency + item tendency.
  const rows = [
    cr('u1', 'i1', 5, 1),
    cr('u1', 'i2', 3, 2),
    cr('u2', 'i1', 4, 3),
    cr('u2', 'i2', 2, 4),
  ];
  const model = trainMatrixFactorization(rows, config);
  assert.ok(model.stats.epochLosses[0]! > model.stats.finalLoss);
  assert.ok(model.stats.finalLoss < 0.05, `loss too high: ${model.stats.finalLoss}`);

  const fallback = {
    globalMean: model.globalMean,
    itemMean: new Map<string, number>(),
    userBiasMean: new Map<string, number>(),
  };
  const p = explain(model, fallback, 'u1', 'i1');
  assert.ok(Math.abs(p.prediction - 5) < 0.2);
});

test('single observation: global mean equals rating and biases stay zero', () => {
  const one = [cr('u', 'i', 4, 1)];
  const cfg: TrainConfig = { ...config, epochs: 1, factors: 0, learningRate: 0.1, seed: 5 };
  const model = trainMatrixFactorization(one, cfg);
  assert.equal(model.globalMean, 4);
  assert.equal(model.stats.epochLosses[0], 0);
  assert.equal(model.userBias.get('u'), 0);
});

test('analytic first SGD bias step for a two-rating zero-factor case', () => {
  // mu=3, factors degenerate, reg=0. Example order (u1,i1): err=2, so
  // bu1, bi1 each move to lr*2. Loss is the pre-update SSE = 2^2 + (-2)^2.
  const rows = [cr('u1', 'i1', 5, 1), cr('u2', 'i2', 1, 2)];
  const cfg: TrainConfig = { ...config, factors: 0, epochs: 1, learningRate: 0.1 };
  const model = trainMatrixFactorization(rows, cfg);
  assert.equal(model.stats.epochLosses[0], 8);
  assert.ok(Math.abs(model.userBias.get('u1')! - 0.2) < 1e-9);
  assert.ok(Math.abs(model.itemBias.get('i1')! - 0.2) < 1e-9);
});

test('factor L2 regularization shrinks factor norms', () => {
  const rows = [cr('a', 'x', 5, 1), cr('a', 'y', 5, 2), cr('b', 'x', 1, 3), cr('b', 'y', 1, 4)];
  const norm = (vecs: Map<string, number[]>) =>
    Math.sqrt(
      [...vecs.values()]
        .flat()
        .reduce((sum, v) => sum + v * v, 0),
    );
  const base = trainMatrixFactorization(rows, { ...config, regFactor: 0, epochs: 200, seed: 5 });
  const reg = trainMatrixFactorization(rows, { ...config, regFactor: 0.5, epochs: 200, seed: 5 });
  assert.ok(norm(reg.userFactors) < norm(base.userFactors));
  assert.ok(norm(reg.itemFactors) < norm(base.itemFactors));
});

test('reproducible: same seed and order gives identical parameters', () => {
  const rows = [cr('a', 'x', 5, 1), cr('a', 'y', 1, 2), cr('b', 'x', 2, 3), cr('b', 'y', 4, 4)];
  const m1 = trainMatrixFactorization(rows, config);
  const m2 = trainMatrixFactorization(rows, config);
  assert.deepEqual(m1.userBias, m2.userBias);
  assert.deepEqual(m1.itemBias, m2.itemBias);
  assert.deepEqual([...m1.userFactors.values()], [...m2.userFactors.values()]);
  assert.deepEqual(m1.stats.epochLosses, m2.stats.epochLosses);
});

test('different seeds produce different factor initializations', () => {
  const rows = [cr('a', 'x', 5, 1), cr('b', 'y', 4, 2), cr('c', 'z', 2, 3)];
  const m1 = trainMatrixFactorization(rows, { ...config, seed: 1, epochs: 1 });
  const m2 = trainMatrixFactorization(rows, { ...config, seed: 2, epochs: 1 });
  assert.notDeepEqual([...m1.userFactors.values()], [...m2.userFactors.values()]);
});

test('regularization shrinks biases toward zero relative to no regularization', () => {
  const rows = [cr('a', 'x', 5, 1), cr('a', 'y', 5, 2), cr('b', 'x', 1, 3), cr('b', 'y', 1, 4)];
  const base = trainMatrixFactorization(rows, { ...config, regBias: 0, epochs: 200 });
  const reg = trainMatrixFactorization(rows, { ...config, regBias: 0.5, epochs: 200 });
  assert.ok(
    Math.abs(reg.userBias.get('a')!) < Math.abs(base.userBias.get('a')!),
  );
});

test('cold-start fallback uses training global mean and flags provenance', () => {
  const rows = [cr('a', 'x', 4, 1), cr('b', 'y', 2, 2)];
  const trained = trainMatrixFactorization(rows, { ...config, epochs: 5 });
  const fallback = buildFallback(rows);
  const bothCold = explain(trained, fallback, 'ghost', 'mystery');
  assert.equal(bothCold.kind, 'cold_both');
  assert.equal(bothCold.prediction, trained.globalMean);
  assert.equal(bothCold.contribution.source, 'fallback');

  const coldItem = explain(trained, fallback, 'a', 'mystery');
  assert.equal(coldItem.kind, 'cold_item');
  assert.equal(coldItem.contribution.dot, 0);
});
