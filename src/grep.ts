/**
 * Grep — regex search over the parsed session line-stream, with grep-style
 * line context. NOT a filesystem search: we never grep against the JSONL on
 * disk. Sessions are loaded via the same parsers `snap` uses, normalized into
 * `Conversation` objects, then rendered to the same per-line text shape snap
 * prints (header + `[ts] ROLE:` + content). Matching happens in-process over
 * those rendered lines.
 *
 * Default scope is global: all sources (Claude Code CLI, Codex, opencode),
 * all cwds, all time. Use --cwd / --source / --since to narrow.
 */
import {
  discoverSessionFiles,
  isInvokingSession,
  parseSessionFile,
  type Conversation,
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

export interface GrepOptions {
  /** One or more patterns. Multiple patterns are OR'd (alternation). */
  patterns: string[];
  /** Restrict to sessions whose cwd matches (exact path). Default: all cwds. */
  cwd?: string;
  /** Restrict to specific source types. Default: all known sources. */
  sources?: SourceType[];
  /** Treat patterns as literal strings, not regex. Default false. */
  fixedStrings?: boolean;
  /** Case-insensitive matching. Default false. */
  ignoreCase?: boolean;
  /** Lines of context before and after each match (default 5). Overridden by before/after. */
  context?: number;
  /** Lines before the match (overrides context for the before half). */
  before?: number;
  /** Lines after the match (overrides context for the after half). */
  after?: number;
  /** Time window in ms from now. Sessions whose last message is older are skipped. Default: no window. */
  sinceMs?: number;
  /** Cap on total matches across all sessions (default 50). */
  maxMatches?: number;
  /** Cap on sessions scanned (default 500). Performance guardrail. */
  maxSessions?: number;
  /** Skip sessions whose file mtime is within this many ms (live-session filter). Default 10_000, claude-code only. */
  liveSessionWindowMs?: number;
  /** Include the actively-written session. Default false. */
  includeCurrent?: boolean;
  /** Include `entrypoint: 'sdk-cli'` claude-code sessions (Claude Code's own automation). Default false. */
  includeSdkCli?: boolean;
}

export interface GrepContextLine {
  /** 1-indexed line number within the rendered session text. */
  lineNumber: number;
  text: string;
  /** True if this line itself matched the pattern. Context lines are false. */
  isMatch: boolean;
}

export interface GrepMatchGroup {
  sessionId: string;
  sourceLabel: string;
  sourceType: SourceType;
  cwd: string | null;
  filePath: string;
  /** ISO timestamp of the session's last message. */
  sessionLastMessageTime: string;
  /** Age of the session in ms (now - last message time). */
  sessionAgeMs: number;
  /** The matching line numbers in this group (1+). */
  matchLineNumbers: number[];
  /** Lines in the group, with isMatch flagged for each match. Always includes context. */
  lines: GrepContextLine[];
}

export interface GrepResult {
  patterns: string[];
  /** True regex used (after any -F escaping and case-insensitive flag). */
  regex: string;
  matches: GrepMatchGroup[];
  /** Pre-rendered text suitable for terminal output. Empty when matches is empty. */
  text: string;
  /** Number of sessions actually parsed and scanned (after filters). */
  sessionsScanned: number;
  /** Sessions that produced at least one match. */
  sessionsWithMatches: number;
  /** Total match count across all sessions before maxMatches truncation. -1 if unknown (we capped scanning). */
  totalMatchesBeforeCap: number;
  /** True if max-matches or max-sessions clipped the output. */
  truncated: boolean;
}

// Exposed for tests. Not part of the public CLI surface.
export function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function compileRegex(opts: GrepOptions): RegExp {
  if (opts.patterns.length === 0) {
    throw new Error('grep: at least one pattern required');
  }
  const escaped = opts.fixedStrings
    ? opts.patterns.map(escapeRegex)
    : opts.patterns;
  // Multi-pattern alternation. Wrap each in a non-capturing group to avoid
  // ambiguity with internal alternation in user-supplied regexes.
  const body = escaped.map(p => `(?:${p})`).join('|');
  const flags = opts.ignoreCase ? 'i' : '';
  try {
    return new RegExp(body, flags);
  } catch (e) {
    throw new Error(
      `grep: invalid regex ${JSON.stringify(opts.patterns)}: ${
        e instanceof Error ? e.message : String(e)
      }`
    );
  }
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
}

/**
 * Render a Conversation as a flat list of lines — the same shape `snap` prints.
 * Header first, then per-message: a `[ts] ROLE:` marker, the message content
 * split on \n, and a blank line between messages. Returns lines without
 * trailing newlines; line N is `lines[N-1]` (1-indexed for grep semantics).
 */
export function renderSessionLines(
  conv: Conversation,
  sourceLabel: string,
  cwd: string | null,
  ageLabel: string,
  lastMessageTime: Date
): string[] {
  const lines: string[] = [];
  const cwdSegment = cwd ? `, cwd: ${cwd}` : '';
  lines.push(
    `=== ${sourceLabel} session ${conv.sessionId.slice(0, 8)} (${ageLabel}, last message ${lastMessageTime.toISOString()}${cwdSegment}) ===`
  );
  for (const msg of conv.messages) {
    lines.push(`[${msg.timestamp.toISOString()}] ${msg.role.toUpperCase()}:`);
    // Preserve internal blank lines inside a message — they're meaningful for
    // context windows. Split on '\n' (not /\r?\n/) since these are normalized
    // already by the JSON layer.
    for (const contentLine of msg.content.split('\n')) {
      lines.push(contentLine);
    }
    lines.push(''); // separator between messages
  }
  return lines;
}

/**
 * Find all matches in a rendered line array and group adjacent matches into
 * one block (grep merges context windows that overlap; we do the same).
 *
 * @returns { groups: array of { matchLineNumbers, lines: ContextLine[] }, totalMatches }
 */
export function findMatchGroups(
  lines: string[],
  regex: RegExp,
  before: number,
  after: number
): { groups: { matchLineNumbers: number[]; lines: GrepContextLine[] }[]; totalMatches: number } {
  const matchIndices: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    // RegExp with no /g flag: test() is safe to reuse; lastIndex isn't tracked.
    if (regex.test(lines[i])) matchIndices.push(i);
  }

  const groups: { matchLineNumbers: number[]; lines: GrepContextLine[] }[] = [];
  let i = 0;
  while (i < matchIndices.length) {
    // Start of a new group: this match plus all subsequent matches whose
    // pre-context overlaps with the previous match's post-context.
    const groupMatches: number[] = [matchIndices[i]];
    let groupStart = Math.max(0, matchIndices[i] - before);
    let groupEnd = Math.min(lines.length - 1, matchIndices[i] + after);
    let j = i + 1;
    while (j < matchIndices.length) {
      const nextMatch = matchIndices[j];
      const nextPreStart = Math.max(0, nextMatch - before);
      // If the next match's pre-context starts at or before our current
      // groupEnd + 1, the windows touch — merge.
      if (nextPreStart <= groupEnd + 1) {
        groupMatches.push(nextMatch);
        groupEnd = Math.min(lines.length - 1, nextMatch + after);
        j++;
      } else {
        break;
      }
    }

    const matchSet = new Set(groupMatches);
    const blockLines: GrepContextLine[] = [];
    for (let k = groupStart; k <= groupEnd; k++) {
      blockLines.push({
        lineNumber: k + 1,
        text: lines[k],
        isMatch: matchSet.has(k),
      });
    }
    groups.push({
      matchLineNumbers: groupMatches.map(idx => idx + 1),
      lines: blockLines,
    });
    i = j;
  }

  return { groups, totalMatches: matchIndices.length };
}

function renderText(matches: GrepMatchGroup[]): string {
  if (matches.length === 0) return '';
  const out: string[] = [];
  let prevSession: string | null = null;
  for (let m = 0; m < matches.length; m++) {
    const match = matches[m];
    // Session header on first hit, blank line between sessions.
    if (match.sessionId !== prevSession) {
      if (prevSession !== null) out.push('');
      const cwdSegment = match.cwd ? `, cwd: ${match.cwd}` : '';
      out.push(
        `=== ${match.sourceLabel} session ${match.sessionId.slice(0, 8)} (${formatAge(match.sessionAgeMs)}${cwdSegment}) ===`
      );
      prevSession = match.sessionId;
    } else {
      // Same session, separator between groups (mirrors `grep --` separator).
      out.push('--');
    }
    for (const line of match.lines) {
      const gutter = line.isMatch ? '>' : ' ';
      const lineNumStr = String(line.lineNumber).padStart(4, ' ');
      out.push(`${gutter} ${lineNumStr}: ${line.text}`);
    }
  }
  return out.join('\n') + '\n';
}

function sourceLabelOf(t: SourceType): string {
  switch (t) {
    case 'claude-code':
      return 'Claude Code';
    case 'codex':
      return 'Codex';
    case 'opencode':
      return 'opencode';
    case 'cursor':
      return 'Cursor';
    case 'omp':
      return 'OMP';
    case 'devin':
      return 'Devin';
    case 'perplexity':
      return 'Perplexity';
    default:
      return t;
  }
}

export function grep(opts: GrepOptions): GrepResult {
  const regex = compileRegex(opts);
  const context = opts.context ?? 5;
  const before = opts.before ?? context;
  const after = opts.after ?? context;
  const maxMatches = opts.maxMatches ?? 50;
  const maxSessions = opts.maxSessions ?? 500;
  const liveWindowMs = opts.liveSessionWindowMs ?? 10_000;
  const includeCurrent = opts.includeCurrent ?? false;
  const includeSdkCli = opts.includeSdkCli ?? false;
  const sources: Set<SourceType> = new Set(
    opts.sources ?? ['claude-code', 'codex', 'opencode', 'cursor', 'pi', 'omp', 'devin', 'perplexity']
  );

  const now = Date.now();
  const cutoff = opts.sinceMs ? now - opts.sinceMs : null;

  // Discover candidate session files. We do NOT scope discovery to opts.cwd —
  // grep is global by default and a cwd filter applies after parse, since the
  // JSONL's cwd is the source of truth (vs. the indexer's project dir name).
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

  const matches: GrepMatchGroup[] = [];
  let sessionsScanned = 0;
  let sessionsWithMatches = 0;
  let totalMatchesBeforeCap = 0;
  let truncated = false;

  for (const f of candidates) {
    if (sessionsScanned >= maxSessions) {
      truncated = true;
      break;
    }
    if (matches.length >= maxMatches) {
      truncated = true;
      break;
    }
    if (cutoff !== null && f.modifiedTime.getTime() < cutoff) {
      // mtime is an upper bound on content time for append-only logs.
      continue;
    }

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
    if (conv.messages.length === 0) continue;
    if (!includeSdkCli && conv.entrypoint === 'sdk-cli') continue;

    const lastMessageTime = conv.messages[conv.messages.length - 1].timestamp;
    if (cutoff !== null && lastMessageTime.getTime() < cutoff) continue;

    if (!includeCurrent && f.sourceType === 'claude-code') {
      if (isInvokingSession(f.path)) continue;
      const mtimeAgeMs = now - f.modifiedTime.getTime();
      if (liveWindowMs > 0 && mtimeAgeMs < liveWindowMs) continue;
    }

    // Resolve cwd. claude-code per-message cwd lives in the JSONL; codex /
    // opencode populate f.cwd at discovery time. For claude-code we peek the
    // first JSONL chunk because the dashy-name fallback is lossy on names with
    // dashes (e.g. `claude-prose` → `/Users/larsde/src/claude/prose` — wrong).
    const cwd =
      f.cwd ??
      (f.sourceType === 'claude-code'
        ? readSessionCwd(f.path, f.project)
        : null);
    if (opts.cwd && cwd !== opts.cwd) continue;

    sessionsScanned += 1;

    const sourceType: SourceType = f.sourceType ?? 'claude-code';
    const sourceLabel = sourceLabelOf(sourceType);
    const ageMs = now - lastMessageTime.getTime();
    const lines = renderSessionLines(
      conv,
      sourceLabel,
      cwd,
      formatAge(ageMs),
      lastMessageTime
    );
    const { groups, totalMatches } = findMatchGroups(lines, regex, before, after);
    if (groups.length === 0) continue;

    sessionsWithMatches += 1;
    totalMatchesBeforeCap += totalMatches;

    for (const g of groups) {
      if (matches.length >= maxMatches) {
        truncated = true;
        break;
      }
      matches.push({
        sessionId: conv.sessionId,
        sourceLabel,
        sourceType,
        cwd,
        filePath: f.path,
        sessionLastMessageTime: lastMessageTime.toISOString(),
        sessionAgeMs: ageMs,
        matchLineNumbers: g.matchLineNumbers,
        lines: g.lines,
      });
    }
  }

  return {
    patterns: opts.patterns,
    regex: regex.source,
    matches,
    text: renderText(matches),
    sessionsScanned,
    sessionsWithMatches,
    totalMatchesBeforeCap,
    truncated,
  };
}
