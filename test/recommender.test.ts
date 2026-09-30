import assert from 'node:assert/strict';
import test from 'node:test';
import { trainModel, explainPrediction, ndcgAtK } from '../src/recommender.ts';
import type { RatingRecord } from '../src/types.ts';

function rating(id: number, userId: string, itemId: string, value: number, eventTime: string): RatingRecord {
  return { id, userId, itemId, rating: value, eventTime, createdAt: eventTime, active: true };
}

test('one epoch SGD matches hand-computed squared error update with regularization', () => {
  const rows = [
    rating(1, 'u1', 'i1', 1, '2024-01-01T00:00:00Z'),
    rating(2, 'u2', 'i2', 5, '2024-01-02T00:00:00Z'),
  ];
  const first = trainModel(rows, {
    factors: 1,
    epochs: 1,
    learningRate: 0.1,
    regularization: 0.2,
    seed: 123,
    minRating: 1,
    maxRating: 5,
  });
  assert.equal(first.losses[0]!.sse, 8);
  const second = trainModel(rows, {
    factors: 1,
    epochs: 2,
    learningRate: 0.1,
    regularization: 0.2,
    seed: 123,
    minRating: 1,
    maxRating: 5,
  });
  assert.equal(first.model.users.get('u1')!.bias, -0.2);
  assert.equal(first.model.items.get('i1')!.bias, -0.2);
  assert.ok(second.losses[1]!.sse < 8);
  assert.ok(Number.isFinite(second.losses[0]!.objective));
});

test('fixed seed and deterministic update order reproduce the same model', () => {
  const rows = [
    rating(1, 'u1', 'i1', 5, '2024-01-01T00:00Z'),
    rating(2, 'u1', 'i2', 1, '2024-01-02T00:00Z'),
    rating(3, 'u2', 'i1', 4, '2024-01-03T00:00Z'),
  ];
  const options = { factors: 2, epochs: 3, learningRate: 0.05, regularization: 0.02, seed: 7 } as const;
  const a = trainModel(rows, options).model;
  const b = trainModel([...rows].reverse(), options).model;
  assert.deepEqual(a, b);
});

test('cold users and items use only training global mean and disclose fallback', () => {
  const model = trainModel(
    [
      rating(1, 'u1', 'i1', 2, '2024-01-01T00:00Z'),
      rating(2, 'u2', 'i2', 4, '2024-01-02T00:00Z'),
    ],
    { factors: 2, epochs: 1, seed: 1 },
  ).model;
  const explanation = explainPrediction(model, 'new-user', 'new-item');
  assert.equal(explanation.score, 3);
  assert.equal(explanation.contributions.userBias, 0);
  assert.equal(explanation.contributions.itemBias, 0);
  assert.equal(explanation.contributions.interaction, 0);
  assert.equal(explanation.fallback, true);
  assert.match(explanation.note, /not causal/);
});

test('NDCG treats empty candidate and empty relevant sets separately', () => {
  assert.deepEqual(ndcgAtK([], 3).emptyCandidateUsers, []);
  assert.equal(ndcgAtK([], 3).macroNdcg, null);
  const result = ndcgAtK(
    [
      { userId: 'u1', itemId: 'i1', actual: 0, predicted: 1 },
      { userId: 'u1', itemId: 'i2', actual: 0, predicted: 2 },
    ],
    3,
  );
  assert.deepEqual(result.emptyRelevantUsers, ['u1']);
  assert.equal(result.users.u1, null);
});
