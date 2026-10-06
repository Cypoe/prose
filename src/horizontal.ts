/**
 * Horizontal Evolution - Evolve across sessions
 *
 * Takes fragments from the last N sessions and evolves them into
 * a compressed current state. The window naturally prevents crud
 * buildup - old sessions age out, no algorithmic pruning needed.
 */

import { createOpenAI } from '@ai-sdk/openai';
import { generateObject } from 'ai';
import { z } from 'zod';

import type { AllFragments } from './schemas.js';
import { resolveLlmParams } from './llm.js';
import {
  DecisionSchema,
  InsightSchema,
  FocusSchema,
  NarrativeSchema,
  emptyFragments,
} from './schemas.js';

// ============================================================================
// Types
// ============================================================================

export interface SessionSnapshot {
  sessionId: string;
  timestamp: Date;
  fragments: AllFragments;
}

export interface HorizontalEvolutionConfig {
  apiKey: string;
  baseUrl?: string;
  model?: string;
  windowSize?: number;  // How many sessions to include (default: 3)
  currentFragments?: AllFragments; // Baseline to evolve from
  timeoutMs?: number; // Timeout in milliseconds (default: 60000)
  externalFragments?: Array<{ projectName: string; fragments: AllFragments }>; // Linked projects
}

export interface HorizontalEvolutionResult {
  current: AllFragments;
  tokensUsed: number;
  sessionsIncluded: number;
  musings?: string;
}

// ============================================================================
// LLM Client
// ============================================================================

function createLLMClient(config: HorizontalEvolutionConfig) {
  const llm = resolveLlmParams(config);
  return createOpenAI({
    apiKey: llm.apiKey,
    baseURL: llm.baseUrl,
    headers: { 'X-Title': 'prose' },
  });
}

// ============================================================================
// Format helpers
// ============================================================================

function formatFragmentsForEvolution(snapshots: SessionSnapshot[]): string {
  return snapshots
    .map((s, i) => {
      const age = i === 0 ? '(most recent)' : `(${i} sessions ago)`;
      return `## Session ${s.sessionId.slice(0, 8)} ${age}

### Decisions
${JSON.stringify(s.fragments.decisions, null, 2)}

### Insights
${JSON.stringify(s.fragments.insights, null, 2)}

### Focus
${JSON.stringify(s.fragments.focus, null, 2)}
`;
    })
    .join('\n---\n\n');
}

function formatExternalFragments(external?: Array<{ projectName: string; fragments: AllFragments }>): string {
  if (!external || external.length === 0) return '';

  return `\n## INTEGRATED CONTEXT (Linked Projects):
These fragments come from higher-level or peer projects that this project is linked to.
Use this for consistency and to avoid reinventing architectural or strategic wheels.

${external.map(ex => `### Project: ${ex.projectName}
${JSON.stringify({
    decisions: ex.fragments.decisions,
    insights: ex.fragments.insights,
    focus: ex.fragments.focus
  }, null, 2)}
`).join('\n---\n')}
`;
}

// ============================================================================
// Horizontal Evolution
// ============================================================================

/**
 * Evolve decisions horizontally across sessions
 */
async function evolveDecisionsHorizontal(
  snapshots: SessionSnapshot[],
  config: HorizontalEvolutionConfig
) {
  const client = createLLMClient(config);

  const baseline = config.currentFragments?.decisions;
  const prompt = `You are evolving a project's decision memory across multiple recent sessions.
${baseline ? `\n## CURRENT Decisions (Baseline):
${JSON.stringify(baseline, null, 2)}
` : ''}
${formatExternalFragments(config.externalFragments)}
## Recent Session Fragments (In reverse chronological order):
${formatFragmentsForEvolution(snapshots)}

Synthesize these into the CURRENT set of active project decisions.
Sessions prefixed with "INTELLIGENT DESIGN SESSION" are direct human corrections and must be treated as absolute ground truth, overriding any conflicting automated summaries from regular sessions.
Evolve the baseline using these new session fragments. The schema describes what we need.`;

  const { object, usage } = await generateObject({
    model: client(resolveLlmParams(config).model),
    schema: DecisionSchema,
    prompt,
    temperature: 0.3,
    abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
  });

  return { data: object, tokensUsed: usage.totalTokens };
}

/**
 * Evolve insights horizontally across sessions
 */
async function evolveInsightsHorizontal(
  snapshots: SessionSnapshot[],
  config: HorizontalEvolutionConfig
) {
  const client = createLLMClient(config);

  const baseline = config.currentFragments?.insights;
  const prompt = `Synthesize project insights from recent sessions.
${baseline ? `\n## CURRENT Insights (Baseline):
${JSON.stringify(baseline, null, 2)}
` : ''}
${formatExternalFragments(config.externalFragments)}
## Recent Session Fragments:
${formatFragmentsForEvolution(snapshots)}

Synthesize these into what's CURRENTLY valuable to know.
Sessions prefixed with "INTELLIGENT DESIGN SESSION" are direct human corrections and must be treated as absolute ground truth.
Evolve the baseline using these new session fragments. The schema describes what we need.`;

  const { object, usage } = await generateObject({
    model: client(resolveLlmParams(config).model),
    schema: InsightSchema,
    prompt,
    temperature: 0.3,
    abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
  });

  return { data: object, tokensUsed: usage.totalTokens };
}

/**
 * Evolve focus horizontally - this should just be the most recent
 */
async function evolveFocusHorizontal(
  snapshots: SessionSnapshot[],
  config: HorizontalEvolutionConfig
) {
  const client = createLLMClient(config);

  const baseline = config.currentFragments?.focus;
  const prompt = `You are determining the CURRENT focus of a project based on recent sessions.
${baseline ? `\n## CURRENT Focus (Baseline):
${JSON.stringify(baseline, null, 2)}
` : ''}
${formatExternalFragments(config.externalFragments)}
## Recent Session Fragments:
${formatFragmentsForEvolution(snapshots)}

The most recent session's focus is likely most relevant, but synthesize if needed to produce the current state.`;

  const { object, usage } = await generateObject({
    model: client(resolveLlmParams(config).model),
    schema: FocusSchema,
    prompt,
    temperature: 0.3,
    abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
  });

  return { data: object, tokensUsed: usage.totalTokens };
}

/**
 * Evolve narrative horizontally
 */
async function evolveNarrativeHorizontal(
  snapshots: SessionSnapshot[],
  config: HorizontalEvolutionConfig
) {
  const client = createLLMClient(config);

  const baseline = config.currentFragments?.narrative;
  const prompt = `You are synthesizing the narrative arc across multiple development sessions.
${baseline ? `\n## CURRENT Narrative (Baseline):
${JSON.stringify(baseline, null, 2)}
` : ''}
## Recent Session Fragments:
${formatFragmentsForEvolution(snapshots)}

Focus on the overall arc, key moments, and memorable quotes. Evolve the existing narrative history with these new beats.`;

  const { object, usage } = await generateObject({
    model: client(resolveLlmParams(config).model),
    schema: NarrativeSchema,
    prompt,
    temperature: 0.5,
    abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
  });

  return { data: object, tokensUsed: usage.totalTokens };
}

// ============================================================================
// Main horizontal evolution
// ============================================================================

/**
 * Perform horizontal evolution across session snapshots
 */
export async function evolveHorizontal(
  snapshots: SessionSnapshot[],
  config: HorizontalEvolutionConfig
): Promise<HorizontalEvolutionResult> {
  const windowSize = config.windowSize || 3;

  // Take only the most recent N sessions
  const window = snapshots
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    .slice(0, windowSize);

  if (window.length === 0) {
    return {
      current: emptyFragments(),
      tokensUsed: 0,
      sessionsIncluded: 0,
    };
  }

  // If only one session and NO baseline, just return its fragments directly (bootstrap)
  // If we HAVE a baseline, we ALWAYS want to evolve even if it's just one session (Integration)
  if (window.length === 1 && !config.currentFragments) {
    return {
      current: window[0].fragments,
      tokensUsed: 0,
      sessionsIncluded: 1,
    };
  }

  // Evolve each fragment type in parallel
  const [decisions, insights, focus, narrative] = await Promise.all([
    evolveDecisionsHorizontal(window, config),
    evolveInsightsHorizontal(window, config),
    evolveFocusHorizontal(window, config),
    evolveNarrativeHorizontal(window, config),
  ]);

  const totalTokens =
    decisions.tokensUsed +
    insights.tokensUsed +
    focus.tokensUsed +
    narrative.tokensUsed;

  return {
    current: {
      decisions: decisions.data,
      insights: insights.data,
      focus: focus.data,
      narrative: narrative.data,
      vocabulary: null, // Vocabulary can be merged algorithmically later
    },
    tokensUsed: totalTokens,
    sessionsIncluded: window.length,
    musings: (decisions.data as any).musings,
  };
}
