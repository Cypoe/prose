/**
 * Snap — verbatim readout of recent activity in the current cwd.
 *
 * Reads Claude Code session JSONL for the current cwd, returns the tail of
 * the most-recent N sessions (excluding any actively-being-written session)
 * up to a byte budget. Output is verbatim turn-pairs, chronological within
 * each session, sessions ordered newest-first.
 */

import { statSync } from 'fs';
import { discoverSessionFiles, isInvokingSession, parseSessionFile, type Conversation, type Message } from './session-parser.js';
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
import { listBatons, renderBatonHeader, type Baton } from './baton.js';

export interface SnapOptions {
  cwd?: string;
  /** Total byte budget for the assembled text (default 4000). */
  bytes?: number;
  /** Last N messages per session (default 4 → ~2 turn-pairs). */
  turnsPerSession?: number;
  /** Cap on how many sessions to include (default 10). */
  maxSessions?: number;
  /**
   * Per-message byte cap. Long messages (e.g. a 7500-line diff pasted into a
   * commit-message subagent) are truncated to this size with an `[N bytes
   * elided]` suffix so a single huge turn cannot exhaust the global budget.
   * Default 1500. Pass 0 to disable.
   */
  maxMessageBytes?: number;
  /**
   * Live-session filter: skip sessions whose file mtime is within this many
   * ms. mtime updates on every JSONL append (text messages, tool_use,
   * tool_result), so it's a reliable signal that Claude Code is actively in
   * this session — even between user-visible turns. Default 10_000 (10s) is
   * tight enough that a just-`/clear`'d session (whose mtime is frozen the
   * moment the session closed) still surfaces, and wide enough to cover the
   * gap between an assistant's tool_result write and the next tool_use.
   *
   * (file-growth during the parse loop is checked as a secondary signal too.)
   */
  liveSessionWindowMs?: number;
  /** Include the actively-written session (default false). */
  includeCurrent?: boolean;
  /**
   * Prepend the cwd's latest batons ("you are here" markers) as a compact
   * header. Default true — orientation should carry the last sign-off for
   * free. Pass false to suppress (e.g. when composing snap output elsewhere).
   */
  includeBatons?: boolean;
  /**
   * Include `entrypoint: 'sdk-cli'` Claude Code sessions. These are one-shot
   * SDK invocations Claude Code makes for its own automation (commit-message
   * generation, summaries, subagent dispatch). Default false — they are noise
   * for orientation.
   */
  includeSdkCli?: boolean;
}

export interface SnapSessionMeta {
  sessionId: string;
  /** ISO timestamp of the last message inside the session (content-based recency). */
  lastMessageTime: string;
  /** ISO timestamp of the file's last modification (filesystem recency, can be touched by indexers). */
  modifiedTime: string;
  ageMs: number;
  messageCount: number;
}

export interface SnapResult {
  cwd: string;
  text: string;
  bytes: number;
  sessionsIncluded: number;
  turnsIncluded: number;
  truncated: boolean;
  sessions: SnapSessionMeta[];
  /** Latest batons for this cwd (latest per type), newest-first. */
  batons: Baton[];
  /**
   * How many sessions were dropped for being the reader's own live session
   * (invoking-session id, live-mtime window, or file-growth). Load-bearing for
   * the empty case: "nothing has happened in this repo" and "the only thing
   * that happened here is the conversation you are already in" are different
   * facts, and only one of them should send an agent digging.
   */
  skippedLiveSessions: number;
}

function formatAge(ms: number): string {
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
}

function clipMessageContent(content: string, maxBytes: number): string {
  if (maxBytes <= 0) return content;
  const buf = Buffer.from(content, 'utf-8');
  if (buf.length <= maxBytes) return content;
  // Truncate on a UTF-8 boundary by decoding with `fatal: false` then slicing
  // by character. Cheap path: cut bytes and let Node fix up trailing partials.
  const head = buf.subarray(0, maxBytes).toString('utf-8');
  const elided = buf.length - maxBytes;
  return `${head}\n... [${elided} bytes elided]`;
}

export function renderSessionBlock(
  sessionId: string,
  sourceLabel: string,
  lastMessageTime: Date,
  ageLabel: string,
  tail: Message[],
  maxMessageBytes: number
): string {
  const lines: string[] = [];
  lines.push(
    `=== ${sourceLabel} session ${sessionId.slice(0, 8)} (${ageLabel}, last message ${lastMessageTime.toISOString()}) ===`
  );
  for (const msg of tail) {
    lines.push(`[${msg.timestamp.toISOString()}] ${msg.role.toUpperCase()}:`);
    lines.push(clipMessageContent(msg.content, maxMessageBytes));
    lines.push('');
  }
  return lines.join('\n') + '\n';
}

export function renderSessionWithinBudget(
  sessionId: string,
  sourceLabel: string,
  lastMessageTime: Date,
  ageLabel: string,
  messages: Message[],
  turnsPerSession: number,
  maxMessageBytes: number,
  blockBudget: number
): { block: string; tail: Message[] } | null {
  for (let n = turnsPerSession; n >= 1; n--) {
    const tail = messages.slice(-n);
    const headerBytes = Buffer.byteLength(
      renderSessionBlock(sessionId, sourceLabel, lastMessageTime, ageLabel, [], maxMessageBytes),
      'utf-8'
    );
    const contentBudget = Math.max(1, blockBudget - headerBytes - tail.length * 100);
    const fairMessageBytes = Math.max(1, Math.floor(contentBudget / tail.length));
    const block = renderSessionBlock(
      sessionId,
      sourceLabel,
      lastMessageTime,
      ageLabel,
      tail,
      Math.min(maxMessageBytes, fairMessageBytes)
    );
    if (Buffer.byteLength(block, 'utf-8') <= blockBudget) return { block, tail };
  }
  return null;
}

/**
 * Detect whether a session JSONL is being actively appended to. JSONL is
 * append-only, so any size growth between two stat calls flanking the parse
 * means the session is live. Returns true if the file grew (or vanished, or
 * stat fails — conservative: treat as live).
 */
function fileGrewDuringParse(filePath: string, sizeBefore: number): boolean {
  try {
    const sizeAfter = statSync(filePath).size;
    return sizeAfter > sizeBefore;
  } catch {
    return true;
  }
}

export function snap(opts: SnapOptions = {}): SnapResult {
  const cwd = opts.cwd ?? process.cwd();
  const bytesBudget = opts.bytes ?? 4000;
  const turnsPerSession = opts.turnsPerSession ?? 4;
  const maxSessions = opts.maxSessions ?? 10;
  const maxMessageBytes = opts.maxMessageBytes ?? 1500;
  const liveWindowMs = opts.liveSessionWindowMs ?? 10_000;
  const includeCurrent = opts.includeCurrent ?? false;
  const includeSdkCli = opts.includeSdkCli ?? false;
  const includeBatons = opts.includeBatons ?? true;

  // Pass cwd as both projectPath and currentCwd so we hit the cwd-match
  // primary scan AND the misfiled-session secondary scan.
  const claudeFiles = discoverSessionFiles(cwd, cwd);
  const codexFiles = discoverCodexSessionFiles(cwd);
  const opencodeFiles = discoverOpencodeSessionFiles(cwd);
  const cursorFiles = discoverCursorSessionFiles(cwd);
  const piFiles = discoverPiSessionFiles(cwd);
  const ompFiles = discoverOmpSessionFiles(cwd);
  const devinFiles = discoverDevinSessionFiles(cwd);
  const now = Date.now();

  // Parse a wider set of candidates than maxSessions so we can re-sort by
  // content recency (mtime can be skewed by indexers/touches). Take a per-
  // source slice so a noisy source can't crowd the others out — without this,
  // a project with hundreds of Claude Code JSONLs starves Codex sessions
  // from ever reaching the parse stage.
  const perSourceCap = Math.max(maxSessions * 3, 15);
  const candidatePool = [
    ...claudeFiles.slice(0, perSourceCap),
    ...codexFiles.slice(0, perSourceCap),
    ...opencodeFiles.slice(0, perSourceCap),
    ...cursorFiles.slice(0, perSourceCap),
    ...piFiles.slice(0, perSourceCap),
    ...ompFiles.slice(0, perSourceCap),
    ...devinFiles.slice(0, perSourceCap),
  ].sort((a, b) => b.modifiedTime.getTime() - a.modifiedTime.getTime());

  type Parsed = {
    file: typeof candidatePool[number];
    conv: Conversation;
    lastMessageTime: Date;
    contentAgeMs: number;
  };

  const parsed: Parsed[] = [];
  let skippedLiveSessions = 0;
  for (const f of candidatePool) {
    const sizeBefore = f.fileSize;
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
        : parseSessionFile(f.path);
    if (conv.messages.length === 0) continue;

    // Skip Claude Code's own automation: commit-message generation, summary
    // subagents, etc. all run as `entrypoint: sdk-cli` one-shots. They show up
    // as 1-user/1-assistant sessions whose user message is whatever Claude
    // Code passed in (often a giant paste). Not orientation material.
    if (!includeSdkCli && conv.entrypoint === 'sdk-cli') continue;

    // Live detection. Three signals, exact first:
    //   0. session id — the invoking session names itself in the environment,
    //      so we never have to guess about our own file. The clock-based
    //      signals below miss it whenever the last write is older than the
    //      window, which is the common case on an agent's first tool call.
    //   1. mtime within liveWindowMs — catches the current Claude Code session
    //      even when it's between user-visible turns (tool_use/tool_result
    //      lines bump mtime continuously, even when no text was just written).
    //   2. file-growth across the parse window — catches sessions that wrote
    //      a line while we were reading. Belt-and-suspenders for #1.
    const mtimeAgeMs = now - f.modifiedTime.getTime();
    const lastMessageTime = conv.messages[conv.messages.length - 1].timestamp;
    const contentAgeMs = now - lastMessageTime.getTime();
    if (!includeCurrent && f.sourceType === 'claude-code') {
      if (isInvokingSession(f.path)) { skippedLiveSessions++; continue; }
      if (liveWindowMs > 0 && mtimeAgeMs < liveWindowMs) { skippedLiveSessions++; continue; }
      if (fileGrewDuringParse(f.path, sizeBefore)) { skippedLiveSessions++; continue; }
    }

    parsed.push({ file: f, conv, lastMessageTime, contentAgeMs });
  }

  parsed.sort((a, b) => b.lastMessageTime.getTime() - a.lastMessageTime.getTime());

  const sections: string[] = [];
  const sessionsMeta: SnapSessionMeta[] = [];
  let bytes = 0;
  let turnsIncluded = 0;
  let truncated = false;

  for (const p of parsed) {
    if (sessionsMeta.length >= maxSessions) {
      truncated = true;
      break;
    }

    const sourceLabel =
      p.file.sourceType === 'codex'
        ? 'Codex'
        : p.file.sourceType === 'opencode'
        ? 'opencode'
        : p.file.sourceType === 'cursor'
        ? 'Cursor'
        : p.file.sourceType === 'pi'
        ? 'pi'
        : p.file.sourceType === 'omp'
        ? 'OMP'
        : p.file.sourceType === 'devin'
        ? 'Devin'
        : 'Claude Code';

    // Try the full tail first; if it overflows its fair share, shrink.
    // Reserving space for later sessions prevents a few verbose recent tails
    // from hiding every older session behind them.
    const remaining = bytesBudget - bytes;
    const sessionsRemaining = Math.min(maxSessions - sessionsMeta.length, parsed.length - sessionsMeta.length);
    const blockBudget = Math.floor(remaining / sessionsRemaining);
    const rendered = renderSessionWithinBudget(
      p.conv.sessionId,
      sourceLabel,
      p.lastMessageTime,
      formatAge(p.contentAgeMs),
      p.conv.messages,
      turnsPerSession,
      maxMessageBytes,
      blockBudget
    );

    if (!rendered) {
      truncated = true;
      continue;
    }

    const blockBytes = Buffer.byteLength(rendered.block, 'utf-8');
    sections.push(rendered.block);
    bytes += blockBytes;
    turnsIncluded += rendered.tail.length;
    sessionsMeta.push({
      sessionId: p.conv.sessionId,
      lastMessageTime: p.lastMessageTime.toISOString(),
      modifiedTime: p.file.modifiedTime.toISOString(),
      ageMs: p.contentAgeMs,
      messageCount: rendered.tail.length,
    });

    if (bytes >= bytesBudget) {
      truncated = sessionsMeta.length < parsed.length;
      break;
    }
  }

  // Batons are orientation metadata, not session content — prepend them as a
  // header without charging them against the byte budget.
  const batons = includeBatons ? listBatons({ cwd }) : [];
  const text = renderBatonHeader(batons) + sections.join('\n');

  return {
    cwd,
    text,
    bytes,
    sessionsIncluded: sessionsMeta.length,
    turnsIncluded,
    truncated,
    sessions: sessionsMeta,
    batons,
    skippedLiveSessions,
  };
}
