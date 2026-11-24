import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tokenize } from '../src/shared/text'
import knowledge from '../data/knowledge.json'
import facts from '../data/facts.json'
import casualPhrasings from '../data/casual.json'
import intents from '../data/intents.json'

type Entry = {
  id: string
  title: string
  keywords: string[]
  examples: string[]
  /** Casual and misspelled phrasings carried inside the entry itself. */
  casual?: string[]
  answer: string
  more?: string
  related?: string
}

// Two layers: the deep topic entries and the broader short-fact layer.
const docs: Entry[] = [...(knowledge as Entry[]), ...(facts as Entry[])]
const casual = casualPhrasings as Record<string, string[]>
const maxTerms = 4000
/** Kept in sync with src/worker/model.ts. */
const MATCH_THRESHOLD = 0.12

/** Curated examples plus the casual and misspelled versions people actually type. */
const phrasings = (doc: Entry) => [
  ...doc.examples,
  ...(doc.casual ?? []),
  ...(casual[doc.id] ?? []),
]

function terms(doc: Entry, list: string[]) {
  // Question side of the corpus only: the answers are never used as features.
  return tokenize(
    [...list, ...list, doc.title, doc.title, ...doc.keywords, ...doc.keywords].join(' '),
  )
}

/* --- Retrieval: TF-IDF over the question side --- */

function tfidf(termsList: string[], idf: Record<string, number>, vocabulary: Set<string>) {
  const counts = new Map<string, number>()
  for (const term of termsList)
    if (vocabulary.has(term)) counts.set(term, (counts.get(term) ?? 0) + 1)
  const vector: Record<string, number> = {}
  let norm = 0
  for (const [term, count] of counts) {
    const weight = (1 + Math.log(count)) * (idf[term] ?? 1)
    vector[term] = weight
    norm += weight * weight
  }
  norm = Math.sqrt(norm) || 1
  for (const term of Object.keys(vector))
    vector[term] = Math.round((vector[term] / norm) * 1e4) / 1e4
  return vector
}

function buildIndex(items: { doc: Entry; terms: string[] }[]) {
  const df = new Map<string, number>()
  for (const item of items)
    for (const term of new Set(item.terms)) df.set(term, (df.get(term) ?? 0) + 1)

  const idf: Record<string, number> = {}
  for (const [term, count] of df)
    idf[term] = Math.log((items.length + 1) / (count + 1)) + 1

  const features = [...df.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, maxTerms)
    .map(([term]) => term)
  const vocabulary = new Set(features)
  return {
    idf,
    features,
    vocabulary,
    vectors: items.map((item) => tfidf(item.terms, idf, vocabulary)),
  }
}

function embed(text: string, index: { idf: Record<string, number> }) {
  const counts = new Map<string, number>()
  for (const term of tokenize(text)) {
    if (!(term in index.idf)) continue
    counts.set(term, (counts.get(term) ?? 0) + 1)
  }
  const vector: Record<string, number> = {}
  let norm = 0
  for (const [term, count] of counts) {
    const weight = (1 + Math.log(count)) * index.idf[term]
    vector[term] = weight
    norm += weight * weight
  }
  norm = Math.sqrt(norm) || 1
  for (const term of Object.keys(vector)) vector[term] /= norm
  return vector
}

function bestMatch(text: string, index: ReturnType<typeof buildIndex>, ids: string[]) {
  const query = embed(text, index)
  const terms = tokenize(text)
  let best = { id: ids[0], score: 0, rank: 0, docIndex: 0 }
  index.vectors.forEach((vector, docIndex) => {
    let score = 0
    for (const term in query) {
      const weight = vector[term]
      if (weight) score += query[term] * weight
    }
    const matched = terms.filter((term) => vector[term] !== undefined).length
    // Mirrors the coverage-weighted ranking in src/worker/model.ts.
    const rank = score * (0.6 + matched / Math.max(terms.length, 1))
    if (rank > best.rank) best = { id: ids[docIndex], score, rank, docIndex }
  })
  const vector = index.vectors[best.docIndex]
  const matched = terms.filter((term) => vector[term] !== undefined).length
  return { ...best, matched, content: terms.length }
}

/** Mirrors hasOverlap() in src/worker/model.ts. */
function acceptable(match: { score: number; matched: number; content: number }, threshold: number) {
  return match.score >= threshold && match.matched >= (match.content >= 3 ? 2 : 1)
}

/* --- Intent classifier: softmax regression over a union vocabulary --- */

type Classifier = {
  labels: string[]
  features: string[]
  weights: number[][]
  biases: number[]
  classify: (text: string) => string
}

function featurize(text: string, index: Map<string, number>, size: number) {
  const counts = new Array<number>(size).fill(0)
  for (const term of tokenize(text)) {
    const position = index.get(term)
    if (position !== undefined) counts[position] += 1
  }
  let norm = 0
  for (let position = 0; position < counts.length; position++) {
    if (!counts[position]) continue
    counts[position] = 1 + Math.log(counts[position])
    norm += counts[position] * counts[position]
  }
  norm = Math.sqrt(norm) || 1
  for (let position = 0; position < counts.length; position++)
    counts[position] /= norm
  return counts
}

function trainClassifier(
  sets: Record<string, string[]>,
  extraTerms: string[],
  labels: string[],
): Classifier {
  const vocabulary = new Set<string>(extraTerms)
  for (const examples of Object.values(sets))
    for (const example of examples)
      for (const term of tokenize(example)) vocabulary.add(term)
  const features = [...vocabulary].sort()
  const index = new Map(features.map((term, position) => [term, position]))

  const samples: { x: number[]; active: { i: number; v: number }[]; y: number; weight: number }[] =
    []
  for (const [label, examples] of Object.entries(sets)) {
    const y = labels.indexOf(label)
    // Balance classes: a short greeting should not be swamped by hundreds of questions.
    const weight = 1 / Math.max(examples.length, 1)
    for (const example of examples) {
      const x = featurize(example, index, features.length)
      const active = x.flatMap((value, position) =>
        value ? [{ i: position, v: value }] : [],
      )
      if (active.length) samples.push({ x, active, y, weight })
    }
  }

  const weights = labels.map(() => new Array<number>(features.length).fill(0))
  const biases = new Array<number>(labels.length).fill(0)

  const epochs = 400
  const learningRate = 1.2
  const l2 = 1e-5
  let seed = 42
  const random = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296
    return seed / 4294967296
  }

  const order = samples.map((_, position) => position)
  const logits = new Array<number>(labels.length).fill(0)

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let position = order.length - 1; position > 0; position--) {
      const swap = Math.floor(random() * (position + 1))
      ;[order[position], order[swap]] = [order[swap], order[position]]
    }
    for (const position of order) {
      const sample = samples[position]
      const sampleLogits = logits.fill(0)
      for (let classIndex = 0; classIndex < labels.length; classIndex++) {
        const row = weights[classIndex]
        let score = biases[classIndex]
        for (const { i, v } of sample.active) score += row[i] * v
        sampleLogits[classIndex] = score
      }
      const max = Math.max(...sampleLogits)
      let total = 0
      for (let classIndex = 0; classIndex < labels.length; classIndex++) {
        sampleLogits[classIndex] = Math.exp(sampleLogits[classIndex] - max)
        total += sampleLogits[classIndex]
      }
      for (let classIndex = 0; classIndex < labels.length; classIndex++) {
        const gradient =
          (sampleLogits[classIndex] / total - (sample.y === classIndex ? 1 : 0)) *
          sample.weight
        const row = weights[classIndex]
        for (const { i, v } of sample.active)
          row[i] -= learningRate * (gradient * v + l2 * row[i])
        biases[classIndex] -= learningRate * gradient
      }
    }
  }

  return {
    labels,
    features,
    weights,
    biases,
    classify(text: string) {
      const x = featurize(text, index, features.length)
      if (!x.some((value) => value !== 0)) return 'question'
      let best = { label: 'question', score: -Infinity }
      labels.forEach((label, classIndex) => {
        const row = weights[classIndex]
        let score = biases[classIndex]
        for (let position = 0; position < row.length; position++)
          if (x[position]) score += row[position] * x[position]
        if (score > best.score) best = { label, score }
      })
      return best.label
    },
  }
}

/* --- Build the shipped model from every phrasing --- */

const items = docs.map((doc) => ({ doc, terms: terms(doc, phrasings(doc)) }))
const index = buildIndex(items)

const trainedDocs = docs.map((doc, position) => ({
  index: position,
  id: doc.id,
  title: doc.title,
  answer: doc.answer,
  ...(doc.more ? { more: doc.more } : {}),
  ...(doc.related ? { related: doc.related } : {}),
  vector: index.vectors[position],
}))

const intentData = intents as Record<string, string[]>
const labels = [...Object.keys(intentData), 'question']

function buildSets(questionExamples: string[]): Record<string, string[]> {
  return { ...intentData, question: questionExamples }
}

const allQuestionExamples = docs.flatMap((doc) => [...phrasings(doc), doc.title])
const classifier = trainClassifier(
  buildSets(allQuestionExamples),
  index.features,
  labels,
)

let intentCorrect = 0
let intentTotal = 0
const intentMisses: string[] = []
for (const [label, examples] of Object.entries(buildSets(allQuestionExamples)))
  for (const example of examples) {
    intentTotal++
    const predicted = classifier.classify(example)
    if (predicted === label) intentCorrect++
    else if (label !== 'question') intentMisses.push(`${label} -> ${predicted}: ${example}`)
  }

/* --- Held-out evaluation: every 4th phrasing is unseen during fitting --- */

const heldOut: { doc: Entry; text: string }[] = []
const evalItems = docs.map((doc) => {
  const list = phrasings(doc)
  const seen = list.filter((_, position) => position % 4 !== 3)
  list.forEach((text, position) => {
    if (position % 4 === 3) heldOut.push({ doc, text })
  })
  return { doc, terms: terms(doc, seen) }
})
const evalIndex = buildIndex(evalItems)
const evalIds = evalItems.map((item) => item.doc.id)
const evalHeldOut = heldOut.filter((item) => phrasings(item.doc).length >= 6)

let retrievalHits = 0
let retrievalScored = 0
let answeredCorrectly = 0
const retrievalMisses: string[] = []
const acceptedInDomain: number[] = []
for (const item of evalHeldOut) {
  const match = bestMatch(item.text, evalIndex, evalIds)
  retrievalScored++
  if (match.id === item.doc.id) retrievalHits++
  else
    retrievalMisses.push(
      `${item.doc.id} -> ${match.id} (${match.score.toFixed(3)}, ` +
        `${match.matched}/${match.content} terms): ${item.text}`,
    )
  if (match.id === item.doc.id && acceptable(match, MATCH_THRESHOLD)) {
    answeredCorrectly++
    acceptedInDomain.push(match.score)
  }
}

const offTopic = [
  'who won the 1998 world cup final',
  'what is the capital of mongolia',
  'recommend a good pizza topping',
  'how do i change a car tyre',
  'what is the plot of hamlet',
  'when is the next solar eclipse in europe',
  'how do i train my dog to sit',
  'what is the best phone to buy',
  'who painted the mona lisa',
  'how do i grow tomatoes on a balcony',
  'what is the airspeed velocity of an unladen swallow',
  'which team won the super bowl',
]
let offTopicFellBack = 0
let offTopicTotalScore = 0
const offTopicMatches: { score: number; matched: number; content: number }[] = []
const offTopicLines: string[] = []
for (const text of offTopic) {
  const match = bestMatch(text, evalIndex, evalIds)
  offTopicTotalScore += match.score
  offTopicMatches.push(match)
  offTopicLines.push(
    `  ${match.score.toFixed(3)} (${match.matched}/${match.content} terms) -> ${match.id}: ${text}`,
  )
  if (!acceptable(match, MATCH_THRESHOLD)) offTopicFellBack++
}

// Pick the threshold that separates real questions from off-topic ones.
let bestThreshold = MATCH_THRESHOLD
let bestF1 = -1
for (let threshold = 0.12; threshold <= 0.4; threshold += 0.005) {
  const recalled = acceptedInDomain.filter((score) => score >= threshold).length
  const rejected = offTopicMatches.filter((match) => !acceptable(match, threshold)).length
  const precision = rejected + recalled ? recalled / (recalled + rejected) : 0
  const recall = recalled / Math.max(acceptedInDomain.length, 1)
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0
  if (f1 > bestF1) {
    bestF1 = f1
    bestThreshold = threshold
  }
}

/* Held-out intent accuracy: hold out every 4th phrasing per class too. */
const holdoutSets: Record<string, string[]> = {}
const evalSets: Record<string, string[]> = {}
for (const [label, examples] of Object.entries(buildSets(allQuestionExamples))) {
  evalSets[label] = examples.filter((_, position) => position % 4 !== 3)
  holdoutSets[label] = examples.filter((_, position) => position % 4 === 3)
}
const evalClassifier = trainClassifier(evalSets, evalIndex.features, labels)

let heldOutCorrect = 0
let heldOutTotal = 0
const heldOutMisses: string[] = []
for (const [label, examples] of Object.entries(holdoutSets))
  for (const example of examples) {
    heldOutTotal++
    const predicted = evalClassifier.classify(example)
    if (predicted === label) heldOutCorrect++
    else heldOutMisses.push(`${label} -> ${predicted}: ${example}`)
  }

/* --- Write the trained model --- */

const model = {
  version: 2,
  trainedAt: new Date().toISOString(),
  documents: docs.length,
  terms: index.features.length,
  idf: index.idf,
  docs: trainedDocs,
  intentModel: {
    labels,
    features: classifier.features,
    weights: classifier.weights.map((row) =>
      row.map((value) => Math.round(value * 1e4) / 1e4),
    ),
    biases: classifier.biases,
  },
}

const outputDirectory = join(import.meta.dir, '..', 'src', 'model')
const outputFile = join(outputDirectory, 'model.json')
mkdirSync(outputDirectory, { recursive: true })
writeFileSync(outputFile, `${JSON.stringify(model)}\n`)

const size = readFileSync(outputFile).length
const percent = (part: number, whole: number) => ((part / whole) * 100).toFixed(1)

console.log(
  `trained ${docs.length} documents from ${docs.reduce((sum, doc) => sum + phrasings(doc).length, 0)} phrasings, ` +
    `${index.features.length} retrieval features, ${labels.length} intents`,
)
console.log(
  `held-out retrieval top-1 ${percent(retrievalHits, retrievalScored)}% ` +
    `(${retrievalHits}/${retrievalScored})`,
)
console.log(
  `held-out questions answered ${percent(answeredCorrectly, retrievalScored)}% ` +
    `(${answeredCorrectly}/${retrievalScored})`,
)
console.log(
  `off-topic questions that correctly fell back ${offTopicFellBack}/${offTopic.length} ` +
    `(mean best score ${(offTopicTotalScore / offTopic.length).toFixed(3)})`,
)
console.log(
  `suggested match threshold ${bestThreshold.toFixed(2)} ` +
    `(current ${MATCH_THRESHOLD}, accepted in-domain mean score ` +
    `${(acceptedInDomain.reduce((sum, score) => sum + score, 0) / Math.max(acceptedInDomain.length, 1)).toFixed(3)})`,
)
console.log(
  `in-sample intent accuracy ${percent(intentCorrect, intentTotal)}% (${intentCorrect}/${intentTotal})`,
)
console.log(
  `held-out intent accuracy ${percent(heldOutCorrect, heldOutTotal)}% (${heldOutCorrect}/${heldOutTotal})`,
)
if (process.argv.includes('--diagnostics') || process.env.MODEL_DIAGNOSTICS) {
  console.log(`off-topic probe scores:\n${offTopicLines.join('\n')}`)
  if (retrievalMisses.length)
    console.log(`held-out retrieval misses:\n${retrievalMisses.join('\n')}`)
}
if (intentMisses.length)
  console.log(`in-sample misses (non-question labels):\n${intentMisses.join('\n')}`)
if (heldOutMisses.length)
  console.log(`held-out misses:\n${heldOutMisses.join('\n')}`)
console.log(`model size ${(size / 1024).toFixed(0)} KB`)
