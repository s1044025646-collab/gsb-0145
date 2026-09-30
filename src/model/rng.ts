/**
 * Deterministic PRNG (mulberry32 over a 32-bit state) so every "random"
 * initialization is reproducible from an integer seed.
 */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform float in [min, max). */
  uniform(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  /** Small symmetric initialization value around zero. */
  small(scale = 0.1): number {
    return this.uniform(-scale, scale);
  }
}

export function hashSeed(value: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < value.length; i++) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
