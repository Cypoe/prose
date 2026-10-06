/**
 * Fragment Evolution Engine
 *
 * The core of prose: evolve semantic fragments as new messages
 * are processed. Each fragment sees its own previous state and produces
 * an updated version - evolution, not extraction.
 *
 * Uses a cheap LLM (Gemini 3 Flash via OpenRouter) for fast, affordable
 * compression of conversation into structured memory.
 */

import { createOpenAI } from '@ai-sdk/openai';
import { generateObject } from 'ai';
import { z } from 'zod';

import type { Message } from './session-parser.js';
import type { AllFragments, FragmentType, DecisionFragment, InsightFragment, FocusFragment, NarrativeFragment, VocabularyFragment } from './schemas.js';
import {
  DecisionSchema,
  InsightSchema,
  FocusSchema,
  NarrativeSchema,
  VocabularySchema,
  emptyDecisions,
  emptyInsights,
  emptyFocus,
  emptyNarrative,
  emptyVocabulary,
} from './schemas.js';
import { getJinaEmbeddings } from './jina.js';
import { loadProjectVectors, saveProjectVectors, calculateFragmentHash } from './memory.js';
import { resolveLlmParams } from './llm.js';

// ============================================================================
// Types
// ============================================================================

export interface EvolutionContext {
  /** Messages to process in this evolution step */
  messages: Message[];
  /** All fragments (for cross-pollination) */
  allFragments: AllFragments;
  /** Project name/path for context */
  project?: string;
  /** Session ID for context */
  sessionId?: string;
}

export interface EvolutionResult<T> {
  success: boolean;
  data?: T;
  error?: string;
  tokensUsed?: number;
}

export interface EvolutionConfig {
  apiKey: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  timeoutMs?: number; // Timeout in milliseconds (default: 60000)
  jinaApiKey?: string;
}

// ============================================================================
// LLM Client
// ============================================================================

function createLLMClient(config: EvolutionConfig) {
  const llm = resolveLlmParams(config);
  return createOpenAI({
    apiKey: llm.apiKey,
    baseURL: llm.baseUrl,
    headers: { 'X-Title': 'prose' },
  });
}

// ============================================================================
// Message Formatting
// ============================================================================

function formatMessages(messages: Message[]): string {
  return messages
    .map(m => {
      const role = m.role === 'user' ? 'Human' : 'Claude';
      const time = m.timestamp.toLocaleTimeString();
      const content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
      return `[${time}] ${role}: ${content}`;
    })
    .join('\n\n---\n\n');
}

function formatFragmentState(fragment: unknown): string {
  if (!fragment) return 'null (no previous state)';
  return JSON.stringify(fragment, null, 2);
}

// ============================================================================
// Evolution Functions
// ============================================================================

/**
 * Evolve the decisions fragment
 */
export async function evolveDecisions(
  currentState: DecisionFragment | null,
  context: EvolutionContext,
  config: EvolutionConfig
): Promise<EvolutionResult<DecisionFragment>> {
  try {
    const client = createLLMClient(config);

    const systemPrompt = `You are analyzing a development conversation to extract and evolve a record of DECISIONS made.

${currentState ? `## Your Previous State (EVOLVE this - keep relevant decisions, update changed ones, add new ones):
${formatFragmentState(currentState)}` : '## No previous state - create initial decisions from the conversation.'}

## Project Context
Project: ${context.project || 'Unknown'}
Session: ${context.sessionId || 'Unknown'}

## Other Fragments (for context, cross-pollination):
Focus: ${formatFragmentState(context.allFragments.focus)}
Insights: ${formatFragmentState(context.allFragments.insights)}

## New Messages to Process:
${formatMessages(context.messages)}

## Instructions:
Evolve the decisions based on the new messages. The schema describes what we need.`;

    const { object, usage } = await generateObject({
      model: client(resolveLlmParams(config).model),
      schema: DecisionSchema,
      prompt: systemPrompt,
      temperature: config.temperature ?? 0.3,
      abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
    });

    return {
      success: true,
      data: object as DecisionFragment,
      tokensUsed: usage.totalTokens,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Evolve the insights fragment
 */
export async function evolveInsights(
  currentState: InsightFragment | null,
  context: EvolutionContext,
  config: EvolutionConfig
): Promise<EvolutionResult<InsightFragment>> {
  try {
    const client = createLLMClient(config);

    const systemPrompt = `You are analyzing a development conversation to extract and evolve INSIGHTS and LEARNINGS.

${currentState ? `## Your Previous State (EVOLVE this - keep relevant insights, add new ones):
${formatFragmentState(currentState)}` : '## No previous state - create initial insights from the conversation.'}

## Project Context
Project: ${context.project || 'Unknown'}

## Other Fragments (for context):
Decisions: ${formatFragmentState(context.allFragments.decisions)}

## New Messages to Process:
${formatMessages(context.messages)}

## Instructions:
Evolve the insights based on the new messages. The schema describes what we need.`;

    const { object, usage } = await generateObject({
      model: client(resolveLlmParams(config).model),
      schema: InsightSchema,
      prompt: systemPrompt,
      temperature: config.temperature ?? 0.3,
      abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
    });

    return {
      success: true,
      data: object as InsightFragment,
      tokensUsed: usage.totalTokens,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Evolve the focus fragment
 */
export async function evolveFocus(
  currentState: FocusFragment | null,
  context: EvolutionContext,
  config: EvolutionConfig
): Promise<EvolutionResult<FocusFragment>> {
  try {
    const client = createLLMClient(config);

    const systemPrompt = `You are analyzing a development conversation to track the CURRENT FOCUS.

${currentState ? `## Your Previous State (UPDATE based on new messages):
${formatFragmentState(currentState)}` : '## No previous state - determine current focus from the conversation.'}

## New Messages to Process:
${formatMessages(context.messages)}

## Instructions:
Determine what is CURRENTLY being worked on:
- current_goal: The main objective right now
- active_tasks: What's actively in progress
- blockers: What's blocking progress (if any)
- next_steps: What's planned next

This fragment is EPHEMERAL - it reflects the current state, not history.
Update completely based on what the conversation shows as the current focus.`;

    const { object, usage } = await generateObject({
      model: client(resolveLlmParams(config).model),
      schema: FocusSchema,
      prompt: systemPrompt,
      temperature: config.temperature ?? 0.3,
      abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
    });

    return {
      success: true,
      data: object as FocusFragment,
      tokensUsed: usage.totalTokens,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Evolve the narrative fragment
 */
export async function evolveNarrative(
  currentState: NarrativeFragment | null,
  context: EvolutionContext,
  config: EvolutionConfig
): Promise<EvolutionResult<NarrativeFragment>> {
  try {
    const client = createLLMClient(config);

    const systemPrompt = `You are analyzing a development conversation to capture its NARRATIVE arc.

${currentState ? `## Your Previous State (EVOLVE the story):
${formatFragmentState(currentState)}` : '## No previous state - begin the narrative.'}

## Project Context
Project: ${context.project || 'Unknown'}

## New Messages to Process:
${formatMessages(context.messages)}

## Instructions:
Capture the STORY of this development session:
- story_beats: Key moments (setup, conflict, breakthrough, pivot, resolution, cliffhanger)
- memorable_quotes: Lines worth preserving verbatim (especially moments of insight, frustration, or celebration)
- themes: Recurring patterns or themes

Think like a documentarian - what would make this session interesting to read about later?
Be selective - only capture truly notable moments and quotes.`;

    const { object, usage } = await generateObject({
      model: client(resolveLlmParams(config).model),
      schema: NarrativeSchema,
      prompt: systemPrompt,
      temperature: config.temperature ?? 0.5, // Slightly higher for creative narrative
      abortSignal: AbortSignal.timeout(config.timeoutMs ?? 60000),
    });

    return {
      success: true,
      data: object as NarrativeFragment,
      tokensUsed: usage.totalTokens,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Evolve the vocabulary fragment (simpler - can be done locally)
 */
export function evolveVocabulary(
  currentState: VocabularyFragment | null,
  context: EvolutionContext
): EvolutionResult<VocabularyFragment> {
  const state = currentState || emptyVocabulary();

  // Simple tokenization for term frequency
  const allText = context.messages.map(m => m.content).join(' ').toLowerCase();
  const tokens = allText
    .replace(/[^a-z0-9\s-]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2 && t.length < 30);

  // Update term frequency
  const terms = { ...state.terms };
  for (const token of tokens) {
    terms[token] = (terms[token] || 0) + 1;
  }

  // Extract file paths mentioned
  const filePatterns = allText.match(/[a-z0-9_-]+\.(ts|js|tsx|jsx|py|rs|go|md|json|yaml|zig|gd)/gi) || [];
  const files = [...new Set([...state.files, ...filePatterns])];

  // Simple concept/technology extraction (could be enhanced with LLM)
  const techPatterns = /\b(typescript|javascript|python|rust|go|react|svelte|node|npm|git|docker|claude|gpt|llm|ai|zod|zontax)\b/gi;
  const techMatches = allText.match(techPatterns) || [];
  const technologies = [...new Set([...state.technologies, ...techMatches.map(t => t.toLowerCase())])];

  return {
    success: true,
    data: {
      terms,
      concepts: state.concepts, // Would need LLM to extract concepts
      technologies,
      files,
    },
  };
}

// ============================================================================
// Batch Evolution
// ============================================================================

export interface BatchEvolutionResult {
  fragments: AllFragments;
  tokensUsed: number;
  errors: string[];
}

/**
 * Evolve all fragments in parallel
 */
export async function evolveAllFragments(
  currentFragments: AllFragments,
  context: EvolutionContext,
  config: EvolutionConfig
): Promise<BatchEvolutionResult> {
  const contextWithFragments = { ...context, allFragments: currentFragments };

  // Run evolution in parallel
  const [decisionsResult, insightsResult, focusResult, narrativeResult] = await Promise.all([
    evolveDecisions(currentFragments.decisions, contextWithFragments, config),
    evolveInsights(currentFragments.insights, contextWithFragments, config),
    evolveFocus(currentFragments.focus, contextWithFragments, config),
    evolveNarrative(currentFragments.narrative, contextWithFragments, config),
  ]);

  // Vocabulary can be done locally
  const vocabularyResult = evolveVocabulary(currentFragments.vocabulary, context);

  const errors: string[] = [];
  if (!decisionsResult.success) errors.push(`decisions: ${decisionsResult.error}`);
  if (!insightsResult.success) errors.push(`insights: ${insightsResult.error}`);
  if (!focusResult.success) errors.push(`focus: ${focusResult.error}`);
  if (!narrativeResult.success) errors.push(`narrative: ${narrativeResult.error}`);

  const tokensUsed =
    (decisionsResult.tokensUsed || 0) +
    (insightsResult.tokensUsed || 0) +
    (focusResult.tokensUsed || 0) +
    (narrativeResult.tokensUsed || 0);

  const finalFragments = {
    decisions: decisionsResult.data || currentFragments.decisions,
    insights: insightsResult.data || currentFragments.insights,
    focus: focusResult.data || currentFragments.focus,
    narrative: narrativeResult.data || currentFragments.narrative,
    vocabulary: vocabularyResult.data || currentFragments.vocabulary,
  };

  // Generate embeddings for new fragments if Jina key is provided
  if (config.jinaApiKey && context.project) {
    const vectors = loadProjectVectors(context.project);
    const toEmbed: { hash: string; text: string }[] = [];

    // Check decisions
    for (const d of finalFragments.decisions?.decisions || []) {
      const hash = calculateFragmentHash('decision', d.what, d.why);
      if (!vectors[hash]) toEmbed.push({ hash, text: `${d.what} ${d.why}` });
    }
    // Check insights
    for (const i of finalFragments.insights?.insights || []) {
      const hash = calculateFragmentHash('insight', i.learning, i.context);
      if (!vectors[hash]) toEmbed.push({ hash, text: `${i.learning} ${i.context}` });
    }
    // Check story beats
    for (const b of finalFragments.narrative?.story_beats || []) {
      const hash = calculateFragmentHash('narrative', b.summary, b.beat_type);
      if (!vectors[hash]) toEmbed.push({ hash, text: b.summary });
    }

    if (toEmbed.length > 0) {
      try {
        const embeddings = await getJinaEmbeddings(toEmbed.map(e => e.text), config.jinaApiKey);
        toEmbed.forEach((item, index) => {
          vectors[item.hash] = embeddings[index];
        });
        saveProjectVectors(context.project, vectors);
      } catch (e) {
        errors.push(`Jina embedding failed during evolution: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }

  return {
    fragments: finalFragments,
    tokensUsed,
    errors,
  };
}
