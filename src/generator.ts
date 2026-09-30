import { mulberry32 } from './prng.js';
import type { RatingInput } from './types.js';

export function generateRatings(seed = 42, users = 8, items = 10, density = 0.55): RatingInput[] {
  const random = mulberry32(seed);
  const userBias = Array.from({ length: users }, () => (random() - 0.5) * 1.4);
  const itemBias = Array.from({ length: items }, () => (random() - 0.5) * 1.4);
  const startTime = Date.parse('2024-01-01T00:00:00.000Z');
  const ratings: RatingInput[] = [];
  for (let user = 0; user < users; user += 1) {
    for (let item = 0; item < items; item += 1) {
      if (random() > density) continue;
      const noisy = 3 + userBias[user]! + itemBias[item]! + (random() - 0.5) * 0.5;
      const rating = Math.max(1, Math.min(5, Math.round(noisy * 2) / 2));
      ratings.push({
        userId: `u${String(user + 1).padStart(2, '0')}`,
        itemId: `i${String(item + 1).padStart(2, '0')}`,
        rating,
        eventTime: new Date(startTime + ratings.length * 3_600_000 + Math.floor(random() * 1000)).toISOString(),
      });
    }
  }
  return ratings.sort((a, b) => a.eventTime.localeCompare(b.eventTime));
}
