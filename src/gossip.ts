/**
 * Gossip — casual LLM compaction over a `whisper` (neighborhood verbatim).
 *
 * Where `whisper` collects verbatim across the project family, `gossip`
 * runs one streaming LLM pass over that material to produce a short,
 * low-volume paragraph — like a colleague catching you up over coffee.
 * Sits between `whisper` (verbatim, no LLM) and `standup` (formal cross-
 * project structured rundown). No persistence.
 */

import { createOpenAI } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { resolveLlmParams } from './llm.js';

import { whisper, type WhisperOptions, type WhisperResult } from './whisper.js';
import { type NeighborhoodEntry } from './neighborhood.js';

export interface GossipOptions extends WhisperOptions {
  apiKey: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  /** Stream destination for the LLM paragraph. Defaults to process.stdout. */
  out?: NodeJS.WritableStream;
}

export interface GossipResult {
  /** The verbatim whisper that was fed to the LLM. */
  source: WhisperResult;
  /** Members that actually contributed sessions (subset of source.neighborhood). */
  contributing: NeighborhoodEntry[];
  /** The LLM-emitted gossip paragraph. */
  text: string;
  /** False when no sessions were found (LLM was not called). */
  emitted: boolean;
}

function buildSystemPrompt(self: NeighborhoodEntry, contributing: NeighborhoodEntry[]): string {
  const others = contributing.filter((n) => n.path !== self.path);

  if (others.length === 0) {
    return `You are gossiping — quietly catching the user up on their own recent work in **${self.name}**. The verbatim tail of recent agent sessions in that directory is provided below.

Produce ONE short paragraph (3–5 sentences). Capture:
  - what's actively being worked on
  - any open thread, blocker, or decision in flight

No bullet points. No headers. No preamble like "Here's a summary". No restating session IDs or timestamps — the reader already sees those. Voice: direct, low-volume, like a colleague catching them up over coffee. If the sessions look incoherent or empty, say so plainly in one sentence.`;
  }

  const otherList = others.map((o) => o.name).join(', ');
  return `You are gossiping — quietly catching the user up on their own recent work across the **${self.name}** project family. The verbatim tail of recent agent sessions is provided below, grouped by sibling repo.

The user is anchored on **${self.name}**, but its conceptual neighbors share infrastructure, intent, and active development: ${otherList}. **Treat the whole family as one project.** Cross-repo work is the norm, not bleed.

Produce ONE short paragraph (3–5 sentences). Capture:
  - what's actively being worked on across the family
  - any open thread, blocker, or decision in flight
  - notable cross-repo movement when it's the actual story

No bullet points. No headers. No preamble like "Here's a summary". No restating session IDs or timestamps — the reader already sees those. Voice: direct, low-volume, like a colleague catching them up over coffee. If sessions look incoherent or empty, say so plainly in one sentence.`;
}

export async function gossip(opts: GossipOptions): Promise<GossipResult> {
  const out = opts.out ?? process.stdout;

  const source = whisper({
    cwd: opts.cwd,
    bytes: opts.bytes,
    turnsPerSession: opts.turnsPerSession,
    maxSessions: opts.maxSessions,
    liveSessionWindowMs: opts.liveSessionWindowMs,
    includeCurrent: opts.includeCurrent,
    cwdOnly: opts.cwdOnly,
  });

  if (!source.emitted) {
    return { source, contributing: [], text: '', emitted: false };
  }

  const self = source.neighborhood[0];
  // Only siblings that actually contributed sessions get named in the prompt
  // — naming a silent sibling biases the LLM toward fabricating activity for it.
  const contributing = source.neighborhood.filter((n) =>
    source.blocks.some((b) => b.member.path === n.path)
  );

  const llm = resolveLlmParams(opts);
  const client = createOpenAI({
    apiKey: llm.apiKey,
    baseURL: llm.baseUrl,
    headers: { 'X-Title': 'prose' },
  });

  const model = client(llm.model);

  const result = await streamText({
    model,
    temperature: opts.temperature ?? 0.4,
    messages: [
      { role: 'system', content: buildSystemPrompt(self, contributing) },
      { role: 'user', content: source.text },
    ],
  });

  let text = '';
  for await (const chunk of result.textStream) {
    out.write(chunk);
    text += chunk;
  }
  out.write('\n');

  return { source, contributing, text, emitted: true };
}
