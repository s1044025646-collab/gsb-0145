/** Typed application errors carrying stable machine-readable codes. */

export type ErrorCode =
  | 'VALIDATION_ERROR'
  | 'NOT_FOUND'
  | 'CONFLICT'
  | 'UNPROCESSABLE'
  | 'RANGE_ERROR'
  | 'INTERNAL';

export class AppError extends Error {
  readonly code: ErrorCode;
  readonly details?: unknown;

  constructor(code: ErrorCode, message: string, details?: unknown) {
    super(message);
    this.name = 'AppError';
    this.code = code;
    this.details = details;
  }
}

export function validation(message: string, details?: unknown): AppError {
  return new AppError('VALIDATION_ERROR', message, details);
}

export function notFound(message: string): AppError {
  return new AppError('NOT_FOUND', message);
}

export function conflict(message: string): AppError {
  return new AppError('CONFLICT', message);
}

export function unprocessable(message: string, details?: unknown): AppError {
  return new AppError('UNPROCESSABLE', message, details);
}
