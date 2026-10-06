/**
 * Standup — cross-cwd, time-windowed compression of recent activity.
 *
 * Where `whisper` is "what's happening here," `standup` is "what have I been
 * doing across all my work." Crawls every Claude Code session JSONL whose
 * last message falls inside the time window, groups by working directory,
 * and runs a single streaming LLM pass to produce a project-by-project
 * narrative. No persistence.
 */

import { openSync, readSync, closeSync } from 'fs';
import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { resolveLlmParams } from './llm.js';

import { discoverSessionFiles, parseSessionFile, type Message } from './session-parser.js';
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
import { isGitRepo, getCommitsSince, type GitCommitSummary } from './source-parsers.js';
import { type StatusLine } from './status-line.js';

export interface StandupOptions {
  apiKey: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  /** Time window expressed as e.g. "4h", "30m", "1d", "2h30m". Default "4h". */
  since?: string;
  /** Last N messages per session that lands in the window. Default 6. */
  turnsPerSession?: number;
  /** Per-session byte cap on the rendered tail. Default 1500. */
  bytesPerSession?: number;
  /** Total byte cap on the assembled LLM input. Default 24000. */
  totalBytes?: number;
  /** Per-project byte cap on the injected git commit block. Default 800. */
  bytesPerCommitBlock?: number;
  /** Max commits per project surfaced in the commit block. Default 30. */
  commitsPerProject?: number;
  /** Max sessions across all projects. Default 30. */
  maxSessions?: number;
  /** Skip sessions whose last message is within this many ms (live session). Default 60_000. */
  liveSessionWindowMs?: number;
  /** Include the actively-written session. Default false. */
  includeCurrent?: boolean;
  /**
   * Include `entrypoint: 'sdk-cli'` Claude Code sessions. These are one-shot
   * SDK invocations (commit-message generation, summaries, subagents) — noise
   * for orientation. Default false.
   */
  includeSdkCli?: boolean;
  /** Stream destination. Defaults to process.stdout. */
  out?: NodeJS.WritableStream;
  /**
   * Transient progress reporter. Standup is a ~15s wait with nothing on screen
   * — the crawl, then the model — so it says what it is doing while it does it.
   * Omit for silence.
   */
  status?: StatusLine;
}

export interface StandupProjectMeta {
  cwd: string;
  sessionCount: number;
  messageCount: number;
  commitCount: number;
}

export interface StandupResult {
  windowMs: number;
  sessionsIncluded: number;
  projects: StandupProjectMeta[];
  promptBytes: number;
  text: string;
  emitted: boolean;
}

const SYSTEM_PROMPT = `You are giving a quick standup — a project-by-project readout of the user's recent Claude Code activity across multiple working directories. Two kinds of evidence are provided per project, grouped by cwd:

  1. \`### Commits in window\` — git log oneline for the project's repo within the time window. This is GROUND TRUTH for what shipped.
  2. \`--- session ... ---\` — verbatim tails (last N messages) from agent sessions. These are biased toward end-of-session framings — open threads, "next-step" notes, things-to-revisit. They reflect what was top-of-mind at session close, NOT necessarily what's actually pending now.

When commits and session tails conflict, trust commits. If a session tail says "next up: X" but a later commit message describes finishing X, X shipped. If commits show meaningful work that the tails don't recap, surface it from the commits.

Produce a project-by-project narrative. For each project that had real activity:
  - One header line: "**<short project name>**" (derive from the cwd path; just the basename)
  - 2–4 sentences: what was being worked on, what landed (cite commit evidence), anything genuinely open or pending (not just "noted for later")

Skip projects with only trivial activity. Do not restate session IDs, commit hashes, or timestamps. No preamble, no closing summary. Voice: direct, daily-standup register — what changed, what's next.`;

const DURATION_RE = /(\d+)([smhd])/g;

export function parseDuration(input: string): number {
  let total = 0;
  let any = false;
  for (const m of input.matchAll(DURATION_RE)) {
    any = true;
    const n = parseInt(m[1], 10);
    switch (m[2]) {
      case 's': total += n * 1000; break;
      case 'm': total += n * 60_000; break;
      case 'h': total += n * 3_600_000; break;
      case 'd': total += n * 86_400_000; break;
    }
  }
  if (!any && /^\d+$/.test(input.trim())) {
    return parseInt(input.trim(), 10) * 3_600_000;
  }
  if (total <= 0) {
    throw new Error(`Could not parse duration: "${input}". Try "4h", "30m", "1d", or "2h30m".`);
  }
  return total;
}

/**
 * Scan the first chunk of a session JSONL file for a line carrying a cwd field
 * (user/assistant messages have it; meta lines like `queue-operation` don't).
 * Falls back to deriving a path from the project dir name if no line yields one,
 * which is lossy when basenames contain dashes — the JSONL cwd is authoritative.
 */
export function readSessionCwd(filePath: string, projectDirName: string): string {
  try {
    const fd = openSync(filePath, 'r');
    const buf = Buffer.alloc(16384);
    readSync(fd, buf, 0, buf.length, 0);
    closeSync(fd);

    for (const line of buf.toString('utf-8').split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const parsed = JSON.parse(trimmed);
        if (typeof parsed.cwd === 'string' && parsed.cwd.length > 0) {
          return parsed.cwd;
        }
      } catch {
        // partial line at end of buffer is expected; keep scanning previous lines
      }
    }
  } catch {
    // fall through to dashy-name fallback
  }
  return '/' + projectDirName.replace(/^-/, '').replace(/-/g, '/');
}

function basenameOf(path: string): string {
  const trimmed = path.replace(/\/+$/, '');
  const idx = trimmed.lastIndexOf('/');
  return idx === -1 ? trimmed : trimmed.slice(idx + 1);
}

function renderMessage(msg: Message): string {
  const ts = msg.timestamp.toISOString();
  return `[${ts}] ${msg.role.toUpperCase()}:\n${msg.content}\n`;
}

function renderTail(messages: Message[], byteCap: number): string {
  const blocks = messages.map(renderMessage);
  let out = '';
  let bytes = 0;
  for (const b of blocks) {
    const bn = Buffer.byteLength(b, 'utf-8');
    if (bytes + bn > byteCap && out.length > 0) break;
    out += b + '\n';
    bytes += bn + 1;
  }
  return out;
}

/**
 * Render a project's git commits into a compact block, byte-capped. Newest-first.
 * Returns empty string when there are no commits or the repo can't be read.
 */
function renderCommitBlock(commits: GitCommitSummary[], byteCap: number): string {
  if (commits.length === 0) return '';
  const lines: string[] = ['### Commits in window'];
  let included = 0;
  let bytes = Buffer.byteLength(lines[0] + '\n', 'utf-8');
  for (const c of commits) {
    const line = `  ${c.hash.slice(0, 8)} ${c.subject}`;
    const bn = Buffer.byteLength(line + '\n', 'utf-8');
    if (bytes + bn > byteCap && included > 0) {
      const remaining = commits.length - included;
      if (remaining > 0) lines.push(`  (+${remaining} more commits)`);
      break;
    }
    lines.push(line);
    bytes += bn;
    included += 1;
  }
  return lines.join('\n') + '\n';
}

export async function standup(opts: StandupOptions): Promise<StandupResult> {
  const out = opts.out ?? process.stdout;
  // 7d default: a project rhythm window, not an "in the last few hours"
  // window. The byte cap controls volume; the window only decides which
  // sessions are recent enough to be worth surfacing.
  const windowMs = parseDuration(opts.since ?? '7d');
  const turnsPerSession = opts.turnsPerSession ?? 10;
  const bytesPerSession = opts.bytesPerSession ?? 1500;
  const totalBytes = opts.totalBytes ?? 2_000_000;
  const bytesPerCommitBlock = opts.bytesPerCommitBlock ?? 800;
  const commitsPerProject = opts.commitsPerProject ?? 30;
  const maxSessions = opts.maxSessions ?? 80;
  const liveWindowMs = opts.liveSessionWindowMs ?? 60_000;
  const includeCurrent = opts.includeCurrent ?? false;
  const includeSdkCli = opts.includeSdkCli ?? false;

  const status = opts.status ?? { show() {}, clear() {} };

  const cutoff = Date.now() - windowMs;
  status.show(`scanning agent journals from the last ${opts.since ?? '7d'}…`);
  const claudeFiles = discoverSessionFiles();
  const codexFiles = discoverCodexSessionFiles();
  const opencodeFiles = discoverOpencodeSessionFiles();
  const cursorFiles = discoverCursorSessionFiles();
  const piFiles = discoverPiSessionFiles();
  const ompFiles = discoverOmpSessionFiles();
  const devinFiles = discoverDevinSessionFiles();
  const perplexityFiles = discoverPerplexitySessionFiles();
  // No per-source cap here: standup is already time-windowed by mtime, so
  // Claude Code can't structurally crowd the others out the way snap's
  // flat candidate slice did. Sort by mtime, drop everything below the cutoff
  // in the parse loop.
  const allFiles = [
    ...claudeFiles,
    ...codexFiles,
    ...opencodeFiles,
    ...cursorFiles,
    ...piFiles,
    ...ompFiles,
    ...devinFiles,
    ...perplexityFiles,
  ].sort((a, b) => b.modifiedTime.getTime() - a.modifiedTime.getTime());

  type Kept = {
    cwd: string;
    sessionId: string;
    sourceType: 'claude-code' | 'codex' | 'opencode' | 'cursor' | 'pi' | 'omp' | 'devin' | 'perplexity';
    messages: Message[];
    lastMessageTime: Date;
  };

  const kept: Kept[] = [];
  for (const f of allFiles) {
    if (kept.length >= maxSessions) break;
    if (f.modifiedTime.getTime() < cutoff) {
      // mtime can be touched, but if it's well before the cutoff we can skip
      // the parse safely — content time is bounded above by mtime for append-only logs.
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
    const last = conv.messages[conv.messages.length - 1];
    // Window controls inclusion (did this session do anything recently?),
    // not which messages we surface. A session that drifted into the window
    // for a few messages should still get its full tail of context, even if
    // most of that tail predates the cutoff.
    if (last.timestamp.getTime() < cutoff) continue;
    if (!includeCurrent && Date.now() - last.timestamp.getTime() < liveWindowMs) continue;

    const tail = conv.messages.slice(-turnsPerSession);
    // For opencode, the synthetic path has no cwd encoded; the SessionFile.cwd
    // populated at discovery is authoritative. readSessionCwd's first-line
    // JSONL probe doesn't apply.
    const cwd =
      f.cwd ??
      (f.sourceType === 'opencode' || f.sourceType === 'devin'
        ? '/'
        : f.sourceType === 'perplexity'
        ? 'perplexity'
        : readSessionCwd(f.path, f.project));
    kept.push({
      cwd,
      sessionId: conv.sessionId,
      sourceType:
        f.sourceType === 'codex'
          ? 'codex'
          : f.sourceType === 'opencode'
          ? 'opencode'
          : f.sourceType === 'cursor'
          ? 'cursor'
          : f.sourceType === 'pi'
          ? 'pi'
          : f.sourceType === 'omp'
          ? 'omp'
          : f.sourceType === 'devin'
          ? 'devin'
          : f.sourceType === 'perplexity'
          ? 'perplexity'
          : 'claude-code',
      messages: tail,
      lastMessageTime: last.timestamp,
    });
  }

  if (kept.length === 0) {
    status.clear();
    return {
      windowMs,
      sessionsIncluded: 0,
      projects: [],
      promptBytes: 0,
      text: '',
      emitted: false,
    };
  }

  // Group by cwd, sessions newest-first within each project.
  const byCwd = new Map<string, Kept[]>();
  for (const k of kept) {
    const arr = byCwd.get(k.cwd) ?? [];
    arr.push(k);
    byCwd.set(k.cwd, arr);
  }
  for (const arr of byCwd.values()) {
    arr.sort((a, b) => b.lastMessageTime.getTime() - a.lastMessageTime.getTime());
  }

  // Assemble prompt body, project sections newest-activity-first.
  const projectSections = [...byCwd.entries()]
    .map(([cwd, sessions]) => ({
      cwd,
      sessions,
      mostRecent: Math.max(...sessions.map(s => s.lastMessageTime.getTime())),
    }))
    .sort((a, b) => b.mostRecent - a.mostRecent);

  const lines: string[] = [];
  let promptBytes = 0;
  const projectsMeta: StandupProjectMeta[] = [];

  for (const [idx, p] of projectSections.entries()) {
    const header = `## ${p.cwd}\n`;

    // The git shell-outs below are the slow half of the crawl — one repo at a
    // time is exactly the granularity a waiting reader can feel progress at.
    status.show(
      `reading ${basenameOf(p.cwd)} — project ${idx + 1} of ${projectSections.length}…`
    );

    // Pull commits for this project, time-windowed. isGitRepo guards against
    // synthetic cwd fallbacks (e.g. derived dashy-name paths) and non-git dirs.
    const commits = isGitRepo(p.cwd)
      ? getCommitsSince(p.cwd, new Date(cutoff), commitsPerProject)
      : [];
    const commitBlock = renderCommitBlock(commits, bytesPerCommitBlock);

    let sectionBody = '';
    let sessionCount = 0;
    let messageCount = 0;

    for (const s of p.sessions) {
      const block = `--- session ${s.sessionId.slice(0, 8)} (last: ${s.lastMessageTime.toISOString()}) ---\n` +
        renderTail(s.messages, bytesPerSession);
      const blockBytes = Buffer.byteLength(block, 'utf-8');
      const fixedBytes = Buffer.byteLength(header + commitBlock, 'utf-8');
      if (promptBytes + fixedBytes + blockBytes > totalBytes && lines.length > 0) {
        break;
      }
      sectionBody += block + '\n';
      sessionCount += 1;
      messageCount += s.messages.length;
    }

    if (sessionCount === 0) continue;

    const section = header + commitBlock + sectionBody;
    const sectionBytes = Buffer.byteLength(section, 'utf-8');
    if (promptBytes + sectionBytes > totalBytes && lines.length > 0) break;

    lines.push(section);
    promptBytes += sectionBytes;
    projectsMeta.push({
      cwd: p.cwd,
      sessionCount,
      messageCount,
      commitCount: commits.length,
    });
  }

  const promptBody = lines.join('\n');

  const llm = resolveLlmParams(opts);
  const client = createOpenAI({
    apiKey: llm.apiKey,
    baseURL: llm.baseUrl,
    headers: { 'X-Title': 'prose' },
  });
  const modelId = llm.model;
  const model = client(modelId);

  status.show(
    `${kept.length} session(s) across ${projectsMeta.length} project(s), ` +
    `${Math.round(promptBytes / 1024)} KB — asking ${modelId}…`
  );

  const result = await streamText({
    model,
    temperature: opts.temperature ?? 0.4,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: promptBody },
    ],
  });

  let text = '';
  for await (const chunk of result.textStream) {
    // The first token is the moment the wait is over; drop the status before
    // the answer starts landing on the same terminal.
    status.clear();
    out.write(chunk);
    text += chunk;
  }
  out.write('\n');

  return {
    windowMs,
    sessionsIncluded: kept.length,
    projects: projectsMeta,
    promptBytes,
    text,
    emitted: true,
  };
}

// Used by the CLI to derive a short project label without re-importing path utils.
export const __internal = { basenameOf };
