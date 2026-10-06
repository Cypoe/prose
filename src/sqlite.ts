/**
 * Shared read-only SQLite access.
 *
 * Uses the builtin `node:sqlite` module (unflagged since Node 22.13, stable in
 * Node 24). Loaded via createRequire so this module stays importable on older
 * runtimes — openReadonly() just returns null there and every caller degrades
 * to "no sessions from this source" instead of crashing the CLI.
 */

import { createRequire } from 'node:module';

export interface SqliteRow {
  [key: string]: unknown;
}

export interface ReadonlyDb {
  prepare(sql: string): {
    all(...params: unknown[]): SqliteRow[];
    get(...params: unknown[]): SqliteRow | undefined;
  };
  close(): void;
}

let cachedDatabaseSync: any = null;
let resolved = false;

function loadDatabaseSync(): any {
  if (resolved) return cachedDatabaseSync;
  resolved = true;
  try {
    const req = createRequire(import.meta.url);
    // node:sqlite emits "ExperimentalWarning: SQLite is an experimental
    // feature" on first use — noise for a CLI whose stderr users read.
    // Swallow just that one warning while the module loads.
    const origEmit = process.emitWarning.bind(process);
    process.emitWarning = ((warning: unknown, ...args: unknown[]) => {
      const text = typeof warning === 'string' ? warning : '';
      if (text.includes('SQLite is an experimental feature')) return;
      return (origEmit as any)(warning, ...args);
    }) as typeof process.emitWarning;
    try {
      cachedDatabaseSync = req('node:sqlite').DatabaseSync ?? null;
    } finally {
      process.emitWarning = origEmit;
    }
  } catch {
    cachedDatabaseSync = null;
  }
  return cachedDatabaseSync;
}

/**
 * Open a SQLite database read-only. Returns null when node:sqlite is
 * unavailable or the file can't be opened (locked/foreign/missing).
 * WAL-mode databases (Devin, Antigravity trajectories) read fine while the
 * owning process is live.
 */
export function openReadonly(dbPath: string): ReadonlyDb | null {
  const DatabaseSync = loadDatabaseSync();
  if (!DatabaseSync) return null;
  try {
    return new DatabaseSync(dbPath, { readOnly: true }) as ReadonlyDb;
  } catch {
    return null;
  }
}

/**
 * Coerce a SQLite integer timestamp into a Date. Handles both second and
 * millisecond epochs (Devin stores seconds; others may store ms).
 */
export function sqliteTs(value: unknown, fallback: Date): Date {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return new Date(n < 1e11 ? n * 1000 : n);
}
