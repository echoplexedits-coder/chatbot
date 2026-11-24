import primary from '../../data/curriculum/primary.json'
import middle from '../../data/curriculum/middle.json'
import secondary from '../../data/curriculum/secondary.json'
import senior from '../../data/curriculum/senior.json'
import extraCoverage from '../../data/curriculum/extra-coverage.json'
import extraFoundation from '../../data/curriculum/extra-foundation.json'
import extraSecondary from '../../data/curriculum/extra-secondary.json'
import extraSeniorApplied from '../../data/curriculum/extra-senior-applied.json'
import extraSeniorBiology from '../../data/curriculum/extra-senior-biology.json'
import extraSeniorChemistry from '../../data/curriculum/extra-senior-chemistry.json'
import extraSeniorMathsPhysics from '../../data/curriculum/extra-senior-maths-physics.json'
import extraSeniorHumanities from '../../data/curriculum/extra-senior-humanities.json'
import extraCommerce from '../../data/curriculum/extra-commerce.json'
import extraElectives from '../../data/curriculum/extra-electives.json'
import extraLiterature from '../../data/curriculum/extra-literature.json'
import extraPractical from '../../data/curriculum/extra-practical.json'
import extraBoards from '../../data/curriculum/extra-boards.json'
import extraSeniorSocial from '../../data/curriculum/extra-senior-social.json'
import { tokenize } from '../shared/text'

export type Formula = { name: string; expr: string; symbols?: string }

export type Topic = {
  id: string
  title: string
  subject: string
  classes: number[]
  summary: string
  points: string[]
  formulas?: Formula[]
  example?: string
  mistakes?: string[]
  aliases?: string[]
}

export type EducationMode =
  | 'lesson'
  | 'notes'
  | 'revision'
  | 'formulas'
  | 'quiz'
  | 'answers'
  | 'check'
  | 'questions'
  | 'deepen'

const ALL_TOPICS = [
  ...primary,
  ...middle,
  ...secondary,
  ...senior,
  ...extraFoundation,
  ...extraSecondary,
  ...extraSeniorMathsPhysics,
  ...extraSeniorChemistry,
  ...extraSeniorBiology,
  ...extraSeniorSocial,
  ...extraSeniorApplied,
  ...extraCoverage,
  ...extraSeniorHumanities,
  ...extraCommerce,
  ...extraElectives,
  ...extraLiterature,
  ...extraPractical,
  ...extraBoards,
] as unknown as Topic[]

export const TOPICS: Topic[] = ALL_TOPICS

/* --- Looking up a topic --- */

type IndexedTopic = {
  topic: Topic
  title: Set<string>
  body: Set<string>
  phrases: string[]
}

function normalizeQuery(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** A whole chapter title is a stronger signal than one shared word. */
function phraseForms(topic: Topic): string[] {
  return [topic.title, ...(topic.aliases ?? [])]
    .map(normalizeQuery)
    .filter((phrase) => phrase.split(' ').length >= 2)
    .map((phrase) => ` ${phrase} `)
}

const INDEX: IndexedTopic[] = ALL_TOPICS.map((topic) => ({
  topic,
  title: new Set(tokenize(topic.title)),
  body: new Set(
    tokenize(
      [
        ...(topic.aliases ?? []),
        topic.summary,
        ...topic.points,
        topic.subject,
        ...(topic.formulas?.map((formula) => `${formula.name} ${formula.expr}`) ?? []),
      ].join(' '),
    ),
  ),
  phrases: phraseForms(topic),
}))

type SubjectMatch = { label: string; subjects: string[] }

const SUBJECT_PATTERNS: { pattern: RegExp; match: SubjectMatch }[] = [
  {
    pattern: /\b(?:maths?|mathematics|ganit)\b/i,
    match: { label: 'Mathematics', subjects: ['Mathematics'] },
  },
  { pattern: /\bphysics\b/i, match: { label: 'Physics', subjects: ['Physics'] } },
  { pattern: /\bchemistry\b/i, match: { label: 'Chemistry', subjects: ['Chemistry'] } },
  {
    pattern: /\b(?:biology|botany|zoology)\b/i,
    match: { label: 'Biology', subjects: ['Biology'] },
  },
  {
    pattern: /\b(?:science|vigyan|evs|environmental studies)\b/i,
    match: {
      label: 'Science',
      subjects: ['Science', 'Physics', 'Chemistry', 'Biology'],
    },
  },
  { pattern: /\bhistory\b/i, match: { label: 'History', subjects: ['History'] } },
  { pattern: /\bgeography\b/i, match: { label: 'Geography', subjects: ['Geography'] } },
  {
    pattern: /\b(?:civics|political science|polity)\b/i,
    match: { label: 'Civics and Political Science', subjects: ['Civics', 'Political Science'] },
  },
  { pattern: /\beconomics?\b/i, match: { label: 'Economics', subjects: ['Economics'] } },
  {
    pattern: /\b(?:accountancy|accounts?|accounting)\b/i,
    match: { label: 'Accountancy', subjects: ['Accountancy'] },
  },
  {
    pattern: /\bbusiness studies\b/i,
    match: { label: 'Business Studies', subjects: ['Business Studies'] },
  },
  {
    pattern: /\bphysical education\b/i,
    match: { label: 'Physical Education', subjects: ['Physical Education'] },
  },
  {
    pattern: /\b(?:fine arts?|visual arts?|art history)\b/i,
    match: { label: 'Fine Arts', subjects: ['Fine Arts'] },
  },
  {
    pattern: /\bhome science\b/i,
    match: { label: 'Home Science', subjects: ['Home Science'] },
  },
  {
    pattern: /\bentrepreneurship\b/i,
    match: { label: 'Entrepreneurship', subjects: ['Entrepreneurship'] },
  },
  {
    pattern: /\blegal studies\b/i,
    match: { label: 'Legal Studies', subjects: ['Legal Studies'] },
  },
  {
    pattern: /\binformatics practices\b/i,
    match: { label: 'Informatics Practices', subjects: ['Informatics Practices', 'Computer Science'] },
  },
  {
    pattern: /\b(?:computer science|computers?|informatics)\b/i,
    match: { label: 'Computer Science', subjects: ['Computer Science'] },
  },
  { pattern: /\benglish\b/i, match: { label: 'English', subjects: ['English'] } },
  { pattern: /\b(?:hindi|हिंदी)\b/i, match: { label: 'Hindi', subjects: ['Hindi'] } },
  {
    pattern: /\b(?:sociology|sociological)\b/i,
    match: { label: 'Sociology', subjects: ['Sociology'] },
  },
  {
    pattern: /\bpsychology\b/i,
    match: { label: 'Psychology', subjects: ['Psychology'] },
  },
]

/** Subject, class and request words that never identify a chapter on their own. */
const GENERIC_TERMS = new Set([
  ...SUBJECT_PATTERNS.flatMap((entry) => tokenize(entry.match.label)),
  ...tokenize(
    'class grade std standard chapter lesson topic notes note formula formulas question questions ' +
      'important exam examination board sample practice revision revise test quiz teach explain ' +
      'define describe syllabus cbse ncert',
  ),
])

const BOARD_PATTERN =
  /\b(ncert|cbse|icse|isc|igcse|ib|state board|up board|maharashtra board|kerala board|karnataka board|tamil nadu board)\b/i

const CLASS_PATTERN = /\b(?:class|grade|std|standard)\s*[-:]?\s*(\d{1,2})(?:st|nd|rd|th)?\b/i
const CLASS_SUFFIX_PATTERN = /\b(\d{1,2})(?:st|nd|rd|th)\s*(?:class|grade|standard)\b/i

export function detectClass(text: string): number | null {
  const match = text.match(CLASS_PATTERN) ?? text.match(CLASS_SUFFIX_PATTERN)
  if (!match) return null
  const value = Number(match[1])
  return value >= 1 && value <= 12 ? value : null
}

function detectBoard(text: string): string | null {
  const match = text.match(BOARD_PATTERN)
  return match ? match[1].toUpperCase() : null
}

function detectSubject(text: string): SubjectMatch | null {
  for (const entry of SUBJECT_PATTERNS) if (entry.pattern.test(text)) return entry.match
  return null
}

/* --- What kind of teaching response is being asked for --- */

const ANSWERS_MODE =
  /^\s*(?:the\s+)?(?:answers?|answer\s+key|solutions?|show\s+(?:me\s+)?(?:the\s+)?answers?|reveal\s+(?:the\s+)?answers?)\s*[?.!]*$/i
const CHECK_MODE =
  /\b(?:check|mark|grade|correct|evaluate)\b[^?]*\b(?:my|this|the)\b[^?]*\banswer\b|\b(?:is|was)\s+my\s+answer\s+(?:correct|right)\b/i
const QUIZ_MODE =
  /\b(?:test|quiz)\s+me\b|\bask\s+me\s+(?:some\s+)?questions?\b|\btake\s+(?:a|the)\s+(?:test|quiz)\b|\bpractice\s+test\b/i
const EXAM_QUESTIONS =
  /\b(?:important|exam|examination|board|sample|expected|likely|previous\s+year)\b[^?]*\bquestions?\b|\bquestion\s+(?:paper|bank)\b/i
const REVISION_MODE = /\b(?:revision|revise|last[-\s]minute)\b/i
const FORMULA_MODE = /\bformulas?\b|\bequations?\s+sheet\b|\bcheat\s+sheet\b/i
const NOTES_MODE =
  /\b(?:make|write|give|create|prepare|show|need|want|revise\s+from)\b[^?]*\bnotes?\b|\bnotes?\s+(?:on|about|for)\b|\bkey\s+points\b|\bsummari[sz]e\s+(?:this\s+)?(?:topic|chapter)\b/i
const DEEPEN_MODE =
  /\b(?:explain|go|teach)\b[^?]*\b(?:deeply|in\s+depth|in\s+detail|detailed|thoroughly)\b|\bdeep\s+dive\b/i
const LESSON_MODE =
  /^(?:please\s+)?(?:teach(?:\s+me)?|explain|help\s+me\s+(?:understand|learn|study)|i\s+(?:want|need|would\s+like)\s+to\s+(?:learn|understand|study)|(?:start|begin)\s+(?:the\s+)?chapter|lesson\s+(?:on|about)|introduction\s+to)\b/i

function detectMode(text: string): EducationMode | null {
  if (ANSWERS_MODE.test(text)) return 'answers'
  if (CHECK_MODE.test(text)) return 'check'
  if (QUIZ_MODE.test(text)) return 'quiz'
  if (EXAM_QUESTIONS.test(text)) return 'questions'
  if (REVISION_MODE.test(text)) return 'revision'
  if (FORMULA_MODE.test(text)) return 'formulas'
  if (NOTES_MODE.test(text)) return 'notes'
  if (DEEPEN_MODE.test(text)) return 'deepen'
  if (LESSON_MODE.test(text)) return 'lesson'
  return null
}

/** Removes the request framing so only the topic name is left. */
const STRIP_PREFIXES: RegExp[] = [
  /^(?:please\s+)?(?:can|could|would)\s+you\s+(?:please\s+)?/i,
  /^(?:please\s+)?(?:check|mark|grade|correct|evaluate)\s+(?:my|this|the)?\s*(?:answer|work|response)?\s*(?:on|about|for|to)?\s*/i,
  /^(?:teach|explain|describe|define|understand|learn|show|tell)\s+(?:me\s+)?/i,
  /^(?:help\s+me\s+)?/i,
  /^(?:to\s+me|me)\s+/i,
  /^(?:i\s+(?:want|need|would\s+like)\s+to\s+(?:learn|understand|study)\s+(?:about\s+)?)/i,
  /^(?:make|write|give|create|prepare|generate|provide|list|do)\s+(?:me\s+)?/i,
  /^(?:complete|detailed|full|quick|short|brief|important|exam|examination|board|sample|practice|expected|previous\s+year)\s+/i,
  /^(?:revision\s+)?(?:notes?|summary|formula\s+sheet|formulas?|questions?|question\s+paper|lesson|chapter|quiz|test)\s*(?:on|about|for|of|in)?\s*/i,
  /^(?:on|about|for|of|in|to)\s+/i,
  /^(?:the|a|an)\s+/i,
]

function extractTopicText(text: string): string {
  let out = text.trim()
  for (let pass = 0; pass < 6; pass++) {
    const before = out
    for (const pattern of STRIP_PREFIXES) out = out.replace(pattern, '').trim()
    if (out === before) break
  }
  return out
    .replace(/\b(?:class|grade|std|standard)\s*[-:]?\s*\d{1,2}(?:st|nd|rd|th)?\b/gi, ' ')
    .replace(/\b\d{1,2}(?:st|nd|rd|th)\s*(?:class|grade|standard)\b/gi, ' ')
    .replace(/\b(?:ncert|cbse|icse|isc|igcse|ib|state board)\b/gi, ' ')
    .replace(/\bchapter\s*\d+\b/gi, ' ')
    .replace(/\bformula\s+sheet\b/gi, ' ')
    .replace(/\bformulas?\b/gi, ' ')
    .replace(/^(?:on|about|for|of|in|to)\s+/i, '')
    .replace(/\s+(?:on|about|for|of|in|to)$/i, '')
    .replace(/[\s,.:;?!-]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function findTopic(
  terms: string[],
  options: { classLevel: number | null; subjects: string[] | null; phraseTexts?: string[] },
): Topic | null {
  const phraseTexts = (options.phraseTexts ?? [])
    .map((text) => ` ${normalizeQuery(text)} `)
    .filter((text) => text.trim())
  if (!terms.length && !phraseTexts.length) return null
  let best: { topic: Topic; rank: number } | null = null
  for (const item of INDEX) {
    let phraseBonus = 0
    for (const phrase of phraseTexts)
      for (const form of item.phrases)
        if (phrase.includes(form)) {
          const words = form.trim().split(' ').length
          phraseBonus = Math.max(phraseBonus, 6 + words)
        }
    // An exact chapter title outranks a guessed subject, so only filter when the
    // phrase did not name this topic.
    if (!phraseBonus && options.subjects && !options.subjects.includes(item.topic.subject)) continue
    let score = 0
    let hits = 0
    for (const term of terms) {
      if (item.title.has(term)) {
        score += 3
        hits += 1
      } else if (item.body.has(term)) {
        score += 1
        hits += 1
      }
    }
    if (!hits && !phraseBonus) continue
    const coverage = terms.length ? hits / terms.length : 0
    if (!phraseBonus && terms.length > 2 && coverage < 0.5) continue
    let rank = score * (0.6 + coverage) + phraseBonus
    if (options.classLevel && item.topic.classes.includes(options.classLevel)) rank += 2
    if (!best || rank > best.rank) best = { topic: item.topic, rank }
  }
  return best?.topic ?? null
}

/* --- Formatting helpers --- */

function hash(text: string) {
  let value = 0
  for (const char of text) value = (value * 31 + char.charCodeAt(0)) % 100003
  return value
}

function seededOrder<T>(list: T[], seed: number): T[] {
  const out = [...list]
  let state = (seed || 1) % 4294967296
  const next = () => {
    state = (state * 1664525 + 1013904223) % 4294967296
    return state / 4294967296
  }
  for (let index = out.length - 1; index > 0; index--) {
    const swap = Math.floor(next() * (index + 1))
    ;[out[index], out[swap]] = [out[swap], out[index]]
  }
  return out
}

function classLabel(classes: number[]): string {
  const low = Math.min(...classes)
  const high = Math.max(...classes)
  return low === high ? `Class ${low}` : `Classes ${low}–${high}`
}

function firstSentence(text: string): string {
  const sentence = text.match(/^[^.!?]+[.!?]?/)?.[0]?.trim() ?? text
  return sentence.length > 160 ? `${sentence.slice(0, 157).trimEnd()}…` : sentence
}

function bullets(items: string[]): string {
  return items.map((item) => `- ${item}`).join('\n')
}

function formulaLines(formulas: Formula[]): string {
  return formulas
    .map((formula) => `- \`${formula.expr}\` — ${formula.name}${formula.symbols ? ` (${formula.symbols})` : ''}`)
    .join('\n')
}

function pointsFor(topic: Topic, classLevel: number | null): string[] {
  if (classLevel !== null) {
    if (classLevel <= 2) return topic.points.slice(0, 2)
    if (classLevel <= 5) return topic.points.slice(0, 3)
  }
  return topic.points
}

function levelNote(topic: Topic, classLevel: number | null): string {
  if (classLevel === null) return ''
  const low = Math.min(...topic.classes)
  const high = Math.max(...topic.classes)
  if (classLevel < low)
    return `This is usually introduced in ${classLabel(topic.classes)}, so I will keep the language simple and leave out the harder parts.\n\n`
  if (classLevel > high)
    return `This topic usually appears in ${classLabel(topic.classes)}. Here is the fuller explanation, since you are in Class ${classLevel}.\n\n`
  return ''
}

function relatedTopics(topic: Topic, limit = 4): Topic[] {
  const own = new Set(tokenize(topic.title))
  const scored = ALL_TOPICS.filter(
    (other) => other.id !== topic.id && other.subject === topic.subject,
  ).map((other) => {
    const shared = tokenize(other.title).filter((term) => own.has(term)).length
    const classOverlap = other.classes.filter((value) => topic.classes.includes(value)).length
    return { other, score: shared * 3 + classOverlap }
  })
  return seededOrder(scored, hash(topic.id))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((item) => item.other)
}

/* --- Teaching responses --- */

function lesson(topic: Topic, options: { classLevel: number | null; deep: boolean; board: string | null }): string {
  const points = pointsFor(topic, options.classLevel)
  const parts: string[] = []

  parts.push(`**${topic.title}** · ${topic.subject} · ${classLabel(topic.classes)}`)
  parts.push(levelNote(topic, options.classLevel) + topic.summary)

  parts.push(`**Key ideas**\n${bullets(points)}`)

  if (topic.formulas?.length) parts.push(`**Formulas**\n${formulaLines(topic.formulas)}`)
  if (topic.example) parts.push(`**Worked example**\n${topic.example}`)
  if (topic.mistakes?.length)
    parts.push(`**Common mistakes**\n${bullets(topic.mistakes.slice(0, options.deep ? 3 : 2))}`)

  if (options.deep) {
    const related = relatedTopics(topic, 3)
    if (related.length)
      parts.push(
        `**Where it connects**\n${bullets(related.map((other) => `**${other.title}** — ${firstSentence(other.summary)}`))}`,
      )
  }

  parts.push(
    `**Try this**\n${questionFor(topic)}`,
  )

  if (options.board && !/^(NCERT|CBSE)$/.test(options.board))
    parts.push(
      `*Note: I default to NCERT/CBSE. For ${options.board}, check your own syllabus — the concept is the same, but the sequence and depth can differ.*`,
    )

  return parts.join('\n\n')
}

function questionFor(topic: Topic): string {
  if (topic.formulas?.length)
    return `State the formula for **${topic.formulas[0].name}** and explain what each symbol means.`
  if (topic.example) return `Explain **${topic.title}** in two sentences, then give one example of your own.`
  return `In two or three sentences, explain **${topic.title}** without looking at the notes above.`
}

function notes(topic: Topic, classLevel: number | null): string {
  const points = pointsFor(topic, classLevel)
  const parts = [
    `**Notes — ${topic.title}**\n**Subject:** ${topic.subject} · **Classes:** ${classLabel(topic.classes)}`,
    `**Definition**\n${topic.summary}`,
    `**Key points**\n${bullets(points)}`,
  ]
  if (topic.formulas?.length) parts.push(`**Formulas**\n${formulaLines(topic.formulas)}`)
  if (topic.example) parts.push(`**Example**\n${topic.example}`)
  if (topic.mistakes?.length) parts.push(`**Avoid these mistakes**\n${bullets(topic.mistakes)}`)
  parts.push(`**Recall check**\n${questionFor(topic)}`)
  return parts.join('\n\n')
}

function revisionSheet(topic: Topic, classLevel: number | null): string {
  const points = pointsFor(topic, classLevel)
  const parts = [
    `**Revision sheet — ${topic.title}**\n**Subject:** ${topic.subject} · **Classes:** ${classLabel(topic.classes)}`,
    `**Definition:** ${topic.summary}`,
    `**Remember**\n${bullets(points)}`,
  ]
  if (topic.formulas?.length) parts.push(`**Formulas**\n${formulaLines(topic.formulas)}`)
  if (topic.mistakes?.length) parts.push(`**Watch out**\n${bullets(topic.mistakes)}`)
  parts.push(`**One-line recall:** ${points[0] ?? topic.summary}`)
  return parts.join('\n\n')
}

function formulaSheet(topic: Topic): string {
  if (!topic.formulas?.length)
    return `**${topic.title}** is mainly conceptual, so there is no formula to memorise for it.\n\n**What to remember instead**\n${bullets(pointsFor(topic, null))}`
  return `**Formulas — ${topic.title}** · ${topic.subject} · ${classLabel(topic.classes)}\n\n${formulaLines(topic.formulas)}\n\nUse a formula only when the conditions in the symbols column are met.`
}

function subjectFormulaSheet(match: SubjectMatch, classLevel: number | null): string | null {
  const pool = ALL_TOPICS.filter(
    (topic) => match.subjects.includes(topic.subject) && topic.formulas?.length,
  )
  if (!pool.length) return null
  const nearby = classLevel
    ? pool.filter((topic) => topic.classes.some((value) => Math.abs(value - classLevel) <= 1))
    : pool
  const ordered = [...(nearby.length ? nearby : pool)].sort((a, b) => {
    const aInClass = classLevel && a.classes.includes(classLevel) ? 0 : 1
    const bInClass = classLevel && b.classes.includes(classLevel) ? 0 : 1
    if (aInClass !== bInClass) return aInClass - bInClass
    const aDistance = classLevel
      ? Math.min(...a.classes.map((value) => Math.abs(value - classLevel)))
      : 0
    const bDistance = classLevel
      ? Math.min(...b.classes.map((value) => Math.abs(value - classLevel)))
      : 0
    if (aDistance !== bDistance) return aDistance - bDistance
    return a.title.localeCompare(b.title)
  })
  const chosen = ordered.slice(0, 8)

  const sections: string[] = []
  let count = 0
  for (const topic of chosen) {
    if (!topic.formulas?.length || count >= 18) continue
    const formulas = topic.formulas.slice(0, 4)
    count += formulas.length
    sections.push(`**${topic.title}** · ${classLabel(topic.classes)}\n${formulaLines(formulas)}`)
  }

  const heading =
    `**Formula sheet — ${classLevel ? `Class ${classLevel} ` : ''}${match.label}**` +
    '\n\nThese are the formulas stored for this level. Check your own textbook for the exact syllabus and notation.'
  return `${heading}\n\n${sections.join('\n\n')}`
}

type QuizItem = { prompt: string; answer: string }

function buildQuiz(topic: Topic): QuizItem[] {
  const items: QuizItem[] = []

  const pool = ALL_TOPICS.filter((other) => other.id !== topic.id && other.subject === topic.subject)
  const distractorPool = pool.length >= 3 ? pool : ALL_TOPICS.filter((other) => other.id !== topic.id)
  const distractors = seededOrder(distractorPool, hash(topic.id)).slice(0, 3)
  const options = seededOrder(
    [
      { text: firstSentence(topic.summary), correct: true },
      ...distractors.map((other) => ({ text: firstSentence(other.summary), correct: false })),
    ],
    hash(`${topic.id} options`),
  )
  const letters = ['A', 'B', 'C', 'D']
  const correctIndex = options.findIndex((option) => option.correct)

  items.push({
    prompt:
      `Which statement best describes **${topic.title}**?\n` +
      options.map((option, index) => `- ${letters[index]}) ${option.text}`).join('\n'),
    answer: `${letters[correctIndex]}) ${firstSentence(topic.summary)}`,
  })

  if (topic.formulas?.length) {
    const formula = topic.formulas[0]
    items.push({
      prompt: `State the formula for **${formula.name}**.`,
      answer: `\`${formula.expr}\`${formula.symbols ? ` — ${formula.symbols}` : ''}`,
    })
  } else {
    items.push({
      prompt: `Name two key ideas about **${topic.title}**.`,
      answer: topic.points.slice(0, 2).join(' '),
    })
  }

  items.push({
    prompt: `Explain **${topic.title}** in two or three sentences.`,
    answer: `${topic.summary} ${topic.points[0] ?? ''}`.trim(),
  })

  if (topic.example) {
    items.push({
      prompt: `Give one example of **${topic.title}**.`,
      answer: topic.example,
    })
  } else {
    items.push({
      prompt: `Which subject does **${topic.title}** belong to, and in which classes is it usually taught?`,
      answer: `${topic.subject}, ${classLabel(topic.classes)}.`,
    })
  }

  if (topic.mistakes?.length) {
    items.push({
      prompt: `A student makes this mistake: "${topic.mistakes[0]}" What should they do instead?`,
      answer: `Avoid it: ${topic.mistakes[0]}`,
    })
  } else {
    items.push({
      prompt: `In one line, say why **${topic.title}** matters.`,
      answer: topic.points.at(-1) ?? topic.summary,
    })
  }

  return items
}

function quiz(topic: Topic): string {
  const items = buildQuiz(topic)
  return (
    `**Quick test — ${topic.title}** · ${topic.subject} · ${classLabel(topic.classes)}\n\n` +
    `Write your answers on paper first. Type **answers** when you want the key.\n\n` +
    items
      .map((item, index) => `**Q${index + 1}.** ${item.prompt}`)
      .join('\n\n') +
    `\n\nI cannot mark a free-text answer like a teacher, but if you paste one I will check it against the key points.`
  )
}

function answerKey(topic: Topic): string {
  const items = buildQuiz(topic)
  return (
    `**Answer key — ${topic.title}**\n\n` +
    items.map((item, index) => `**${index + 1}.** ${item.answer}`).join('\n\n') +
    `\n\nFor a written answer, the marks usually come from naming the concept, explaining it and giving an example.`
  )
}

function examQuestions(topic: Topic): string {
  const questions = [
    `Define **${topic.title}** in one or two sentences. *(1 mark)*`,
    `Explain **${topic.title}** with a suitable example. *(3 marks)*`,
  ]
  if (topic.formulas?.length)
    questions.push(
      `State the formula for **${topic.formulas[0].name}** and explain what each symbol means. *(2 marks)*`,
    )
  else questions.push(`Write any two important points about **${topic.title}**. *(2 marks)*`)

  if (topic.mistakes?.length)
    questions.push(
      `A student writes: "${topic.mistakes[0]}" Identify the error and correct it. *(3 marks)*`,
    )
  else questions.push(`Give one real-life application of **${topic.title}**. *(2 marks)*`)

  return (
    `**Practice questions — ${topic.title}** · ${topic.subject} · ${classLabel(topic.classes)}\n\n` +
    questions.map((question, index) => `**${index + 1}.** ${question}`).join('\n\n') +
    `\n\nThese are practice questions built from this topic, not a prediction of any examination paper.`
  )
}

function checkAnswer(topic: Topic, answerText: string): string {
  const expected = tokenize(
    [topic.summary, ...topic.points, ...(topic.formulas?.map((formula) => formula.expr) ?? [])].join(
      ' ',
    ),
  )
  const given = new Set(tokenize(answerText))
  if (given.size < 3)
    return `Paste your full answer after the question and I will compare it with the key points for **${topic.title}**. I will not pretend to award marks, but I can show what is missing.`

  const expectedSet = new Set(expected)
  const matched = [...expectedSet].filter((term) => given.has(term))
  const missing = [...expectedSet].filter((term) => !given.has(term)).slice(0, 6)

  return (
    `**Checking your answer on ${topic.title}**\n\n` +
    `I cannot mark like a teacher, but I can compare your wording with the key ideas. Your answer used **${matched.length} of the ${expectedSet.size} terms** I look for in a full answer.\n\n` +
    `**Key points a full answer should include**\n${bullets(topic.points)}\n\n` +
    (missing.length ? `**Not mentioned yet**\n${bullets(missing.map((term) => `\`${term}\``))}\n\n` : '') +
    `**A model answer**\n${topic.summary}\n\n` +
    `If your answer has the definition and the key points in your own words, that is usually what earns the marks.`
  )
}

function guidance(
  match: SubjectMatch | null,
  classLevel: number | null,
  mode: EducationMode | null,
): string {
  const subjectLine = match ? match.label : null
  const label = `${classLevel ? `Class ${classLevel} ` : ''}${subjectLine ?? ''}`.trim()

  let pool = ALL_TOPICS.filter(
    (topic) =>
      (!match || match.subjects.includes(topic.subject)) &&
      (!classLevel || topic.classes.includes(classLevel)),
  )
  if (!pool.length)
    pool = ALL_TOPICS.filter((topic) => !match || match.subjects.includes(topic.subject))
  if (!pool.length) pool = ALL_TOPICS

  const preview = seededOrder(pool, hash(label || 'all')).slice(0, 10)
  const examples =
    mode === 'formulas'
      ? 'try **"formulas for class 10 maths"** or **"notes on quadratic equations"**'
      : mode === 'quiz'
        ? 'try **"test me on class 8 science"** or **"test me on integers"**'
        : 'try **"notes on integers for class 7"** or **"explain photosynthesis for class 7"**'

  return (
    `I can teach across the NCERT/CBSE Classes 1–12 map, but I need a topic name to be accurate. For example, ${examples}.\n\n` +
    (label ? `**${label}** topics I have:\n` : '**Some topics I have:**\n') +
    bullets(preview.map((topic) => `${topic.title} — ${topic.subject}, ${classLabel(topic.classes)}`)) +
    `\n\nTell me the chapter or topic and I can make a lesson, notes, a revision sheet, formulas, practice questions or a short test.`
  )
}

/* --- Entry point --- */

export function educationReply(text: string, previousUser?: string): string | null {
  const mode = detectMode(text)
  const classLevel = detectClass(text)
  const board = detectBoard(text)
  const subject = detectSubject(text)

  // Without an explicit teaching request or a class/board reference, the normal
  // knowledge lookup should answer. This keeps ordinary questions unchanged.
  if (!mode && classLevel === null && !board) return null

  const answerText =
    mode === 'check' && text.includes(':') ? text.slice(text.indexOf(':') + 1).trim() : text
  const topicSource = mode === 'answers' || mode === 'check' ? previousUser ?? text : text
  const topicHint =
    mode === 'check' && text.includes(':') ? text.slice(0, text.indexOf(':')) : topicSource
  const topicText = extractTopicText(topicHint)
  // The raw message keeps board names and class numbers that topic extraction
  // strips, so a chapter title written in full still matches exactly.
  const options = { classLevel, subjects: subject?.subjects ?? null, phraseTexts: [topicText, text] }
  const terms = tokenize(topicText).filter(
    (term) => /\p{L}/u.test(term) && !GENERIC_TERMS.has(term),
  )

  let topic = findTopic(terms, options)
  // Only when the framing words were all that was left does the full sentence
  // get a second chance; otherwise words like "explain" can pick a chapter.
  if (!topic && !terms.length)
    topic = findTopic(
      tokenize(text).filter((term) => /\p{L}/u.test(term) && !GENERIC_TERMS.has(term)),
      options,
    )

  // "formulas for class 10 maths" names a subject, not a single chapter, so it
  // gets a sheet rather than whichever chapter happened to score highest.
  if (mode === 'formulas' && subject) {
    const terms = tokenize(topicText)
    const titleHit = topic && terms.some((term) => tokenize(topic!.title).includes(term))
    if (!titleHit) {
      const sheet = subjectFormulaSheet(subject, classLevel)
      if (sheet) return sheet
    }
  }

  if (!topic) {
    if (mode === 'formulas' && subject) {
      const sheet = subjectFormulaSheet(subject, classLevel)
      if (sheet) return sheet
    }
    if (mode === 'answers' || mode === 'check')
      return `I need the topic first. Ask me for a test or a question, then send **answers** or **check my answer** and I will use that topic.`
    // A general "explain X in detail" for a topic outside the curriculum map
    // should reach the normal knowledge lookup rather than a topic list.
    if ((mode === 'lesson' || mode === 'deepen') && !subject && classLevel === null) return null
    return guidance(subject, classLevel, mode)
  }

  switch (mode) {
    case 'answers':
      return answerKey(topic)
    case 'quiz':
      return quiz(topic)
    case 'check':
      return checkAnswer(topic, answerText)
    case 'formulas':
      return formulaSheet(topic)
    case 'revision':
      return revisionSheet(topic, classLevel)
    case 'notes':
      return notes(topic, classLevel)
    case 'questions':
      return examQuestions(topic)
    case 'deepen':
      return lesson(topic, { classLevel, deep: true, board })
    default:
      return lesson(topic, { classLevel, deep: false, board })
  }
}
