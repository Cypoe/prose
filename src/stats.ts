/**
 * Stats — quantitative readout over the parsed session stream (no LLM).
 *
 * Answers "how much am I actually doing this?": per-day active hours, message
 * volumes, session/project counts, and an hour-of-day histogram, across all
 * sources (Claude Code CLI, Codex, opencode, Cursor, pi, OMP). Default scope
 * is global — all cwds — because the interesting number is the combined one.
 *
 * Active time is computed from message timestamps: consecutive messages less
 * than the idle gap apart (default 15m) merge into one interval; the per-day
 * total is the union of those intervals, sliced at local midnight. Two
 * variants are reported:
 *   - activeMs: intervals over ALL messages — "an agent was working".
 *   - humanMs:  intervals over USER messages only — "you were at the keys".
 * A lone message contributes zero width, so quiet days are not inflated.
 */
import { createHash } from 'crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  discoverSessionFiles,
  parseSessionFile,
  type SourceType,
} from './session-parser.js';
import {
  discoverCodexSessionFiles,
  parseCodexSessionFile,
} from './codex-session-parser.js';
import {
  discoverOpencodeSessionFiles,
  parseOpencodeSessionFile,
} from './opencode-session-parser.js';
import {
  discoverCursorSessionFiles,
  parseCursorSessionFile,
} from './cursor-session-parser.js';
import {
  discoverPiSessionFiles,
  discoverOmpSessionFiles,
  parsePiSessionFile,
} from './pi-session-parser.js';
import {
  discoverDevinSessionFiles,
  parseDevinSessionFile,
} from './devin-session-parser.js';
import {
  discoverPerplexitySessionFiles,
  parsePerplexitySessionFile,
} from './perplexity-session-parser.js';
import { readSessionCwd } from './standup.js';

export interface StatsOptions {
  /** Restrict to sessions whose cwd matches (exact path). Default: all cwds. */
  cwd?: string;
  /** Restrict to specific source types. Default: all known sources. */
  sources?: SourceType[];
  /** Time window in ms from now. Default 30 days. */
  sinceMs?: number;
  /** Gap above which activity is considered idle and intervals split (default 15m). */
  idleGapMs?: number;
  /** Include `entrypoint: 'sdk-cli'` claude-code sessions (Claude Code's own automation). Default false. */
  includeSdkCli?: boolean;
  /** Cap on sessions parsed (performance guardrail, default 2000). */
  maxSessions?: number;
  /**
   * Use the per-file stamp cache in the OS temp dir (default true). The cache
   * is EXACT, not a TTL: entries are keyed on file size + mtime, and session
   * journals are append-only, so a hit means the file is byte-identical to
   * when it was parsed. Only changed files are re-parsed.
   */
  cache?: boolean;
  /** Clock override for tests. Default Date.now(). */
  now?: number;
}

/** One message timestamp with just enough context to aggregate. */
export interface MessageStamp {
  /** Epoch ms. */
  t: number;
  role: 'user' | 'assistant';
  sourceType: SourceType;
  sessionId: string;
  /** Resolved working directory (project identity). */
  cwd: string;
}

export interface DayStats {
  /** Local-time calendar day, YYYY-MM-DD. */
  day: string;
  /** Union of gap-merged intervals over all messages, sliced to this day. */
  activeMs: number;
  /** Same, over user messages only — presence, not agent runtime. */
  humanMs: number;
  userMessages: number;
  assistantMessages: number;
  /** Distinct sessions with at least one message this day. */
  sessions: number;
  /** Distinct cwds with at least one message this day. */
  projects: number;
  /** Message counts per source type this day. */
  sources: Partial<Record<SourceType, number>>;
}

export interface ProjectStats {
  /** Resolved working directory (project identity). */
  cwd: string;
  /** ISO timestamp of the project's last message in the window. */
  lastActivity: string;
  /** Union of gap-merged intervals over this project's messages. */
  activeMs: number;
  userMessages: number;
  assistantMessages: number;
  sessions: number;
  sources: Partial<Record<SourceType, number>>;
}

export interface StatsTotals {
  activeMs: number;
  humanMs: number;
  userMessages: number;
  assistantMessages: number;
  sessions: number;
  projects: number;
  /** Days in the window with at least one message. */
  activeDays: number;
  /** The single busiest day in the window by active time. Null when no activity. */
  peakDay: { day: string; activeMs: number; humanMs: number } | null;
  /**
   * Averages over CLOSED active days only — the current local day is excluded
   * because it's still accruing and would drag the average down. (Peak day
   * keeps today: a partial day can only undercount, never falsely win.)
   * Null when the window has no closed active days.
   */
  averages: { activeMsPerDay: number; humanMsPerDay: number; closedActiveDays: number } | null;
  sources: Partial<Record<SourceType, number>>;
}

export interface StatsResult {
  sinceMs: number;
  idleGapMs: number;
  days: DayStats[];
  /** All projects in the window, ordered by last activity, newest first. */
  projects: ProjectStats[];
  totals: StatsTotals;
  /** User messages per local hour-of-day, index 0-23. */
  hourHistogram: number[];
  sessionsScanned: number;
  /**
   * Session FILES served from the stamp cache rather than parsed fresh.
   * Counted before the empty/sdk-cli/cwd filters, so it can exceed
   * sessionsScanned (which counts sessions surviving those filters).
   */
  cacheHits: number;
  truncated: boolean;
  /** Pre-rendered table + histogram for terminal output. */
  text: string;
}

const DEFAULT_SINCE_MS = 30 * 86_400_000;
const DEFAULT_IDLE_GAP_MS = 15 * 60_000;
const DEFAULT_MAX_SESSIONS = 2000;

// ============================================================================
// Per-file stamp cache
// ============================================================================
//
// Parsing ~600 session files costs seconds; the stamps we need from each are
// tiny. Session journals are append-only, so (size, mtime) identifies file
// content exactly — a matching entry IS the file, not a stale approximation.
// One JSON per session file under the OS temp dir; only changed files re-parse.

const CACHE_DIR = join(tmpdir(), 'prose-stats-cache');
/** Bump when the entry shape or stamp extraction semantics change. */
const CACHE_VERSION = 1;

/** Exposed for tests. Not part of the public CLI surface. */
export interface StampCacheEntry {
  version: number;
  size: number;
  mtimeMs: number;
  sessionId: string;
  /** Resolved cwd, or null when the source doesn't record one. */
  cwd: string | null;
  entrypoint?: string;
  stamps: Array<{ t: number; role: 'user' | 'assistant' }>;
}

/** Exposed for tests. */
export function cachePathFor(sessionFilePath: string): string {
  const hash = createHash('sha1').update(sessionFilePath).digest('hex');
  return join(CACHE_DIR, `${hash}.json`);
}

/** Exposed for tests. Returns null on miss, mismatch, or unreadable entry. */
export function readCacheEntry(
  sessionFilePath: string,
  size: number,
  mtimeMs: number
): StampCacheEntry | null {
  try {
    const entry = JSON.parse(readFileSync(cachePathFor(sessionFilePath), 'utf-8')) as StampCacheEntry;
    if (entry.version !== CACHE_VERSION) return null;
    if (entry.size !== size || entry.mtimeMs !== mtimeMs) return null;
    return entry;
  } catch {
    return null; // missing or corrupt — caller re-parses and overwrites
  }
}

/** Exposed for tests. Write failures degrade to uncached operation (warned once per process). */
let cacheWriteWarned = false;
export function writeCacheEntry(sessionFilePath: string, entry: StampCacheEntry): void {
  try {
    mkdirSync(CACHE_DIR, { recursive: true });
    writeFileSync(cachePathFor(sessionFilePath), JSON.stringify(entry));
  } catch (err) {
    if (!cacheWriteWarned) {
      cacheWriteWarned = true;
      console.error(
        `prose stats: cache write to ${CACHE_DIR} failed (${err instanceof Error ? err.message : err}) — continuing uncached`
      );
    }
  }
}

/** Local-time calendar day key (YYYY-MM-DD) for an epoch-ms timestamp. */
export function localDayKey(t: number): string {
  const d = new Date(t);
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

/** Epoch ms of the next local midnight strictly after t. */
function nextLocalMidnight(t: number): number {
  const d = new Date(t);
  d.setHours(24, 0, 0, 0);
  return d.getTime();
}

/**
 * Merge sorted epoch-ms timestamps into [start, end] intervals, splitting
 * where consecutive stamps are more than gapMs apart. Exported for tests.
 */
export function mergeIntervals(sortedTimes: number[], gapMs: number): Array<[number, number]> {
  const intervals: Array<[number, number]> = [];
  for (const t of sortedTimes) {
    const last = intervals[intervals.length - 1];
    if (last && t - last[1] <= gapMs) {
      last[1] = t;
    } else {
      intervals.push([t, t]);
    }
  }
  return intervals;
}

/** Attribute interval durations to local calendar days, slicing at midnight. */
function attributeToDays(intervals: Array<[number, number]>, into: Map<string, number>): void {
  for (const [start, end] of intervals) {
    let cur = start;
    while (cur < end) {
      const sliceEnd = Math.min(end, nextLocalMidnight(cur));
      const key = localDayKey(cur);
      into.set(key, (into.get(key) ?? 0) + (sliceEnd - cur));
      cur = sliceEnd;
    }
  }
}

/**
 * Pure aggregation core: stamps in, day buckets + totals + histogram out.
 * Stamps need not be sorted. Exported for tests.
 */
export function computeStats(
  stamps: MessageStamp[],
  idleGapMs: number,
  now: number = Date.now()
): { days: DayStats[]; projects: ProjectStats[]; totals: StatsTotals; hourHistogram: number[] } {
  const sorted = [...stamps].sort((a, b) => a.t - b.t);

  const activeByDay = new Map<string, number>();
  const humanByDay = new Map<string, number>();
  attributeToDays(mergeIntervals(sorted.map(s => s.t), idleGapMs), activeByDay);
  attributeToDays(
    mergeIntervals(sorted.filter(s => s.role === 'user').map(s => s.t), idleGapMs),
    humanByDay
  );

  type DayAcc = {
    user: number;
    assistant: number;
    sessions: Set<string>;
    projects: Set<string>;
    sources: Partial<Record<SourceType, number>>;
  };
  const dayAcc = new Map<string, DayAcc>();
  type ProjectAcc = {
    times: number[];
    user: number;
    assistant: number;
    sessions: Set<string>;
    last: number;
    sources: Partial<Record<SourceType, number>>;
  };
  const projectAcc = new Map<string, ProjectAcc>();
  const hourHistogram = new Array(24).fill(0);
  const allSessions = new Set<string>();
  const allProjects = new Set<string>();
  const allSources: Partial<Record<SourceType, number>> = {};
  let totalUser = 0;
  let totalAssistant = 0;

  for (const s of sorted) {
    const key = localDayKey(s.t);
    let acc = dayAcc.get(key);
    if (!acc) {
      acc = { user: 0, assistant: 0, sessions: new Set(), projects: new Set(), sources: {} };
      dayAcc.set(key, acc);
    }
    if (s.role === 'user') {
      acc.user++;
      totalUser++;
      hourHistogram[new Date(s.t).getHours()]++;
    } else {
      acc.assistant++;
      totalAssistant++;
    }
    acc.sessions.add(s.sessionId);
    acc.projects.add(s.cwd);
    acc.sources[s.sourceType] = (acc.sources[s.sourceType] ?? 0) + 1;
    allSessions.add(s.sessionId);
    allProjects.add(s.cwd);
    allSources[s.sourceType] = (allSources[s.sourceType] ?? 0) + 1;

    let proj = projectAcc.get(s.cwd);
    if (!proj) {
      proj = { times: [], user: 0, assistant: 0, sessions: new Set(), last: 0, sources: {} };
      projectAcc.set(s.cwd, proj);
    }
    proj.times.push(s.t);
    proj[s.role === 'user' ? 'user' : 'assistant']++;
    proj.sessions.add(s.sessionId);
    proj.sources[s.sourceType] = (proj.sources[s.sourceType] ?? 0) + 1;
    if (s.t > proj.last) proj.last = s.t;
  }

  // Per-project active time: intervals merged within the project only —
  // a project's hours measure ITS activity, not the global union's slice.
  const projects: ProjectStats[] = [...projectAcc.entries()]
    .map(([cwd, acc]) => ({
      cwd,
      lastActivity: new Date(acc.last).toISOString(),
      activeMs: mergeIntervals(acc.times, idleGapMs).reduce((sum, [a, b]) => sum + (b - a), 0),
      userMessages: acc.user,
      assistantMessages: acc.assistant,
      sessions: acc.sessions.size,
      sources: acc.sources,
    }))
    .sort((a, b) => b.lastActivity.localeCompare(a.lastActivity));

  const days: DayStats[] = [...dayAcc.keys()].sort().map(day => {
    const acc = dayAcc.get(day)!;
    return {
      day,
      activeMs: activeByDay.get(day) ?? 0,
      humanMs: humanByDay.get(day) ?? 0,
      userMessages: acc.user,
      assistantMessages: acc.assistant,
      sessions: acc.sessions.size,
      projects: acc.projects.size,
      sources: acc.sources,
    };
  });

  // Peak = busiest HUMAN day (your messages), not active. Active pins toward 24h
  // for loop/background-driven days, which makes it a meaningless "busiest" pick.
  const peak = days.reduce(
    (best: DayStats | null, d) => (best === null || d.humanMs > best.humanMs ? d : best),
    null
  );

  const todayKey = localDayKey(now);
  const closedDays = days.filter(d => d.day !== todayKey);
  const averages =
    closedDays.length > 0
      ? {
          activeMsPerDay: closedDays.reduce((sum, d) => sum + d.activeMs, 0) / closedDays.length,
          humanMsPerDay: closedDays.reduce((sum, d) => sum + d.humanMs, 0) / closedDays.length,
          closedActiveDays: closedDays.length,
        }
      : null;

  const totals: StatsTotals = {
    activeMs: days.reduce((sum, d) => sum + d.activeMs, 0),
    humanMs: days.reduce((sum, d) => sum + d.humanMs, 0),
    userMessages: totalUser,
    assistantMessages: totalAssistant,
    sessions: allSessions.size,
    projects: allProjects.size,
    activeDays: days.length,
    peakDay: peak ? { day: peak.day, activeMs: peak.activeMs, humanMs: peak.humanMs } : null,
    averages,
    sources: allSources,
  };

  return { days, projects, totals, hourHistogram };
}

function hours(ms: number): string {
  return (ms / 3_600_000).toFixed(1);
}

function bar(value: number, max: number, width: number): string {
  if (max <= 0) return '';
  return '█'.repeat(Math.round((value / max) * width));
}

function shortenHome(path: string): string {
  const home = process.env.HOME;
  return home && path.startsWith(home) ? '~' + path.slice(home.length) : path;
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
}

function renderText(
  days: DayStats[],
  projects: ProjectStats[],
  totals: StatsTotals,
  hourHistogram: number[],
  now: number
): string {
  if (days.length === 0) return '';
  const out: string[] = [];

  out.push('day         active   human    user    asst  sess  proj');
  const maxActive = Math.max(...days.map(d => d.activeMs));
  for (const d of days) {
    out.push(
      [
        d.day,
        `${hours(d.activeMs).padStart(5)}h`,
        `${hours(d.humanMs).padStart(5)}h`,
        String(d.userMessages).padStart(6),
        String(d.assistantMessages).padStart(7),
        String(d.sessions).padStart(5),
        String(d.projects).padStart(5),
      ].join('  ') + `  ${bar(d.activeMs, maxActive, 24)}`
    );
  }

  out.push('');
  const sourceList = (Object.entries(totals.sources) as Array<[SourceType, number]>)
    .sort((a, b) => b[1] - a[1])
    .map(([src, n]) => `${src} ${n}`)
    .join(', ');
  out.push(
    `totals: ${hours(totals.activeMs)}h active (${hours(totals.humanMs)}h human) over ` +
      `${totals.activeDays} day(s) — ${totals.userMessages} user / ${totals.assistantMessages} assistant ` +
      `message(s), ${totals.sessions} session(s), ${totals.projects} project(s)`
  );
  out.push(
    (totals.averages
      ? `avg/closed-day: ${hours(totals.averages.activeMsPerDay)}h active, ` +
        `${hours(totals.averages.humanMsPerDay)}h human (${totals.averages.closedActiveDays} closed day(s))`
      : 'avg/closed-day: n/a (no closed days in window)') +
      (totals.peakDay ? ` — peak human day ${totals.peakDay.day} (${hours(totals.peakDay.humanMs)}h)` : '') +
      ` — sources: ${sourceList}`
  );

  out.push('');
  out.push('recently touched projects (10 most recent of ' + projects.length + '):');
  const recent = projects.slice(0, 10);
  const nameWidth = Math.max(...recent.map(p => shortenHome(p.cwd).length), 7);
  for (const p of recent) {
    out.push(
      [
        shortenHome(p.cwd).padEnd(nameWidth),
        formatAge(now - new Date(p.lastActivity).getTime()).padStart(8),
        `${hours(p.activeMs).padStart(6)}h`,
        `${String(p.userMessages).padStart(5)}u`,
        `${String(p.assistantMessages).padStart(6)}a`,
        `${String(p.sessions).padStart(4)} sess`,
      ].join('  ')
    );
  }

  out.push('');
  out.push('user messages by local hour:');
  const maxHour = Math.max(...hourHistogram, 1);
  for (let h = 0; h < 24; h++) {
    out.push(
      `${String(h).padStart(2, '0')}  ${bar(hourHistogram[h], maxHour, 40).padEnd(40)} ${hourHistogram[h]}`
    );
  }

  return out.join('\n') + '\n';
}

/** Render day buckets as CSV (header + one row per day) for external graphing. */
export function renderCsv(days: DayStats[]): string {
  const rows = [
    'day,active_hours,human_hours,user_messages,assistant_messages,sessions,projects',
  ];
  for (const d of days) {
    rows.push(
      [
        d.day,
        hours(d.activeMs),
        hours(d.humanMs),
        d.userMessages,
        d.assistantMessages,
        d.sessions,
        d.projects,
      ].join(',')
    );
  }
  return rows.join('\n') + '\n';
}

export function stats(opts: StatsOptions = {}): StatsResult {
  const sinceMs = opts.sinceMs ?? DEFAULT_SINCE_MS;
  const idleGapMs = opts.idleGapMs ?? DEFAULT_IDLE_GAP_MS;
  const maxSessions = opts.maxSessions ?? DEFAULT_MAX_SESSIONS;
  const includeSdkCli = opts.includeSdkCli ?? false;
  const useCache = opts.cache ?? true;
  const now = opts.now ?? Date.now();
  const cutoff = now - sinceMs;
  const sources: Set<SourceType> = new Set(
    opts.sources ?? ['claude-code', 'codex', 'opencode', 'cursor', 'pi', 'omp', 'devin', 'perplexity']
  );

  // Discover globally; a cwd filter applies after parse, since the JSONL's cwd
  // is the source of truth (vs. the indexer's project dir name). Same posture
  // as grep. mtime is an upper bound on content time for append-only logs, so
  // pre-filtering on it never drops in-window messages.
  const candidates: ReturnType<typeof discoverSessionFiles> = [];
  if (sources.has('claude-code')) candidates.push(...discoverSessionFiles());
  if (sources.has('codex')) candidates.push(...discoverCodexSessionFiles());
  if (sources.has('opencode')) candidates.push(...discoverOpencodeSessionFiles());
  if (sources.has('cursor')) candidates.push(...discoverCursorSessionFiles());
  if (sources.has('pi')) candidates.push(...discoverPiSessionFiles());
  if (sources.has('omp')) candidates.push(...discoverOmpSessionFiles());
  if (sources.has('devin')) candidates.push(...discoverDevinSessionFiles());
  if (sources.has('perplexity')) candidates.push(...discoverPerplexitySessionFiles());
  candidates.sort((a, b) => b.modifiedTime.getTime() - a.modifiedTime.getTime());

  const stamps: MessageStamp[] = [];
  let sessionsScanned = 0;
  let cacheHits = 0;
  let truncated = false;

  for (const f of candidates) {
    if (f.modifiedTime.getTime() < cutoff) continue;
    if (sessionsScanned >= maxSessions) {
      truncated = true;
      break;
    }

    const mtimeMs = f.modifiedTime.getTime();
    let entry = useCache ? readCacheEntry(f.path, f.fileSize, mtimeMs) : null;
    if (entry) {
      cacheHits += 1;
    } else {
      const conv =
        f.sourceType === 'codex'
          ? parseCodexSessionFile(f.path)
          : f.sourceType === 'opencode'
          ? parseOpencodeSessionFile(f.path)
          : f.sourceType === 'cursor'
          ? parseCursorSessionFile(f.path)
          : f.sourceType === 'pi' || f.sourceType === 'omp'
          ? parsePiSessionFile(f.path, f.sourceType)
          : f.sourceType === 'devin'
          ? parseDevinSessionFile(f.path, f.sessionId)
          : f.sourceType === 'perplexity'
          ? parsePerplexitySessionFile(f.path, f.sessionId)
          : parseSessionFile(f.path);
      entry = {
        version: CACHE_VERSION,
        size: f.fileSize,
        mtimeMs,
        sessionId: conv.sessionId,
        cwd:
          f.cwd ??
          (f.sourceType === 'claude-code' ? readSessionCwd(f.path, f.project) : null),
        entrypoint: conv.entrypoint,
        stamps: conv.messages.map(m => ({ t: m.timestamp.getTime(), role: m.role })),
      };
      if (useCache) writeCacheEntry(f.path, entry);
    }

    if (entry.stamps.length === 0) continue;
    if (!includeSdkCli && entry.entrypoint === 'sdk-cli') continue;
    if (opts.cwd && entry.cwd !== opts.cwd) continue;

    sessionsScanned += 1;

    const sourceType: SourceType = f.sourceType ?? 'claude-code';
    for (const s of entry.stamps) {
      if (Number.isNaN(s.t) || s.t < cutoff || s.t > now) continue;
      stamps.push({
        t: s.t,
        role: s.role,
        sourceType,
        sessionId: entry.sessionId,
        cwd: entry.cwd ?? f.project,
      });
    }
  }

  const { days, projects, totals, hourHistogram } = computeStats(stamps, idleGapMs, now);

  return {
    sinceMs,
    idleGapMs,
    days,
    projects,
    totals,
    hourHistogram,
    sessionsScanned,
    cacheHits,
    truncated,
    text: renderText(days, projects, totals, hourHistogram, now),
  };
}
