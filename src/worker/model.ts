import modelData from '../model/model.json'
import { tokenize } from '../shared/text'
import { educationReply } from './education'

export type ChatMessage = { role: 'user' | 'assistant'; content: string }

type Doc = {
  index: number
  id: string
  title: string
  answer: string
  more?: string
  related?: string
  vector: Record<string, number>
}

type IntentModel = {
  labels: string[]
  features: string[]
  weights: number[][]
  biases: number[]
}

type Model = {
  version: number
  trainedAt: string
  documents: number
  terms: number
  idf: Record<string, number>
  docs: Doc[]
  intentModel: IntentModel
}

const model = modelData as unknown as Model
const docs = model.docs
const byId = new Map(docs.map((doc) => [doc.id, doc]))
const intent = model.intentModel
const intentIndex = new Map(intent.features.map((term, index) => [term, index]))
const MATCH_THRESHOLD = 0.12
const FALLBACK_TOPICS = [
  'school topics from Classes 1–10: integers, fractions, algebra, geometry, light, electricity, photosynthesis, the freedom movement',
  'senior subjects: quadratic equations, calculus, trigonometry, thermodynamics, organic chemistry, genetics, accountancy',
  'photosynthesis, gravity, black holes, why the sky is blue, earthquakes, volcanoes',
  'the internet, encryption, passwords, scams, APIs, HTTPS, databases, Git, Docker',
  'recursion, Big O, async code, testing, regex, HTML and CSS, machine learning',
  'sleep, focus, writing clearly, studying, budgeting, exercise, habits, interviews',
]

/* --- Text features (identical to the training script) --- */

function embed(text: string) {
  const counts = new Map<string, number>()
  for (const term of tokenize(text)) {
    if (!(term in model.idf)) continue
    counts.set(term, (counts.get(term) ?? 0) + 1)
  }
  const vector: Record<string, number> = {}
  let norm = 0
  for (const [term, count] of counts) {
    const weight = (1 + Math.log(count)) * model.idf[term]
    vector[term] = weight
    norm += weight * weight
  }
  norm = Math.sqrt(norm) || 1
  for (const term of Object.keys(vector)) vector[term] /= norm
  return vector
}

function search(text: string) {
  const terms = tokenize(text)
  const query = embed(text)
  let best: { doc: Doc; score: number; rank: number; matched: number } | null = null
  for (const doc of docs) {
    let score = 0
    for (const term in query) {
      const weight = doc.vector[term]
      if (weight) score += query[term] * weight
    }
    // One shared word can carry a cosine score, so also count how much of the
    // question the entry actually covers and rank on both.
    const matched = terms.filter((term) => doc.vector[term] !== undefined).length
    const rank = score * (0.6 + matched / Math.max(terms.length, 1))
    if (!best || rank > best.rank) best = { doc, score, rank, matched }
  }
  const match = best ?? { doc: docs[0], score: 0, rank: 0, matched: 0 }
  return { ...match, content: terms.length }
}

/** A single shared word is enough only for very short questions like "gravity?". */
function hasOverlap(match: { matched: number; content: number }) {
  return match.matched >= (match.content >= 3 ? 2 : 1)
}

/** "can you tell me about X" should look up X, not the words around it. */
function stripLead(text: string) {
  return text
    .trim()
    .replace(
      /^(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:tell|explain|show|teach)\s+me\s+(?:about|how|what|why)?\s*/i,
      '',
    )
    .replace(/^(?:please\s+)?(?:tell|explain|show|teach)\s+me\s+(?:about|how|what|why)?\s*/i, '')
    .replace(
      /^i\s+(?:want|would like|'d like)\s+to\s+(?:know|learn|understand)\s+(?:about|how|what|why)?\s*/i,
      '',
    )
    .replace(/^(?:let'?s|can we|could we)\s+(?:talk|chat)\s+about\s*/i, '')
    .replace(/^do you know\s+(?:about|anything about)\s*/i, '')
    .replace(/^what do you know about\s*/i, '')
    .replace(/^explain\s+(.+?)\s+to\s+me\s*$/i, '$1')
    .trim()
}

/* --- Trained intent classifier (softmax regression) --- */

function classify(text: string) {
  const x = new Array<number>(intent.features.length).fill(0)
  for (const term of tokenize(text)) {
    const index = intentIndex.get(term)
    if (index !== undefined) x[index] += 1
  }
  let norm = 0
  for (let index = 0; index < x.length; index++) {
    if (!x[index]) continue
    x[index] = 1 + Math.log(x[index])
    norm += x[index] * x[index]
  }
  if (!norm) return 'question'
  norm = Math.sqrt(norm)
  for (let index = 0; index < x.length; index++) x[index] /= norm

  let best = { label: 'question', score: -Infinity }
  intent.labels.forEach((label, classIndex) => {
    const row = intent.weights[classIndex]
    let score = intent.biases[classIndex]
    for (let feature = 0; feature < row.length; feature++)
      if (x[feature]) score += row[feature] * x[feature]
    if (score > best.score) best = { label, score }
  })
  return best.label
}

/* --- Structured facts that must not be guessed --- */

const CAPITALS: Record<string, string> = {
  france: 'Paris',
  japan: 'Tokyo',
  australia: 'Canberra',
  canada: 'Ottawa',
  brazil: 'Brasília',
  egypt: 'Cairo',
  india: 'New Delhi',
  kenya: 'Nairobi',
  norway: 'Oslo',
  argentina: 'Buenos Aires',
  'united states': 'Washington, D.C.',
  usa: 'Washington, D.C.',
  spain: 'Madrid',
  italy: 'Rome',
  germany: 'Berlin',
  china: 'Beijing',
  russia: 'Moscow',
  mexico: 'Mexico City',
  'south korea': 'Seoul',
  'united kingdom': 'London',
  uk: 'London',
  portugal: 'Lisbon',
  netherlands: 'Amsterdam',
  sweden: 'Stockholm',
  poland: 'Warsaw',
  turkey: 'Ankara',
  'south africa': 'Pretoria',
  nigeria: 'Abuja',
  'saudi arabia': 'Riyadh',
  greece: 'Athens',
  ireland: 'Dublin',
  'new zealand': 'Wellington',
}

const CAPITAL_QUERY = /capital (?:city )?of (?:the )?([a-z][a-z .'-]*?)(?:\s*[?.!]|$)/i

function capital(text: string): string | null {
  const match = text.match(CAPITAL_QUERY)
  if (!match) return null
  const country = match[1].trim().replace(/\s+/g, ' ').toLowerCase()
  const city = CAPITALS[country]
  if (city)
    return speak(
      `The capital of ${country.replace(/\b\w/g, (letter) => letter.toUpperCase())} is **${city}**.`,
      country,
      text,
    )
  const title = (value: string) =>
    value.replace(/\b\w/g, (letter) => letter.toUpperCase())
  const known = Object.keys(CAPITALS)
    .filter((name) => name.length > 3)
    .slice(0, 6)
    .map(title)
  return `I do not have the capital of ${title(country)} in my knowledge base, so I would rather say that than guess. I do know ${known.join(', ')} and a few others.`
}

/* --- Local arithmetic --- */

function evaluateExpression(source: string): number | null {
  const tokens = source.match(/\d+(?:\.\d+)?|[+\-*/^%()]|sqrt|sin|cos|tan|log|ln|abs|round|floor|ceil|pi|e/g)
  if (!tokens) return null
  let position = 0

  const peek = () => tokens[position]
  const take = () => tokens[position++]

  const parsePrimary = (): number => {
    const token = take()
    if (token === undefined) throw new Error('bad')
    if (token === '(') {
      const value = parseSum()
      if (take() !== ')') throw new Error('bad')
      return value
    }
    if (token === '-') return -parsePrimary()
    if (token === '+') return parsePrimary()
    if (token === 'pi') return Math.PI
    if (token === 'e') return Math.E
    if (token === 'sqrt') return Math.sqrt(parsePrimary())
    if (token === 'abs') return Math.abs(parsePrimary())
    if (token === 'round') return Math.round(parsePrimary())
    if (token === 'floor') return Math.floor(parsePrimary())
    if (token === 'ceil') return Math.ceil(parsePrimary())
    if (token === 'ln') return Math.log(parsePrimary())
    if (token === 'log') return Math.log10(parsePrimary())
    if (token === 'sin') return Math.sin(parsePrimary())
    if (token === 'cos') return Math.cos(parsePrimary())
    if (token === 'tan') return Math.tan(parsePrimary())
    return Number(token)
  }

  const parsePower = (): number => {
    const base = parsePrimary()
    if (peek() === '^') {
      take()
      return base ** parsePower()
    }
    return base
  }

  const parseProduct = (): number => {
    let value = parsePower()
    while (peek() === '*' || peek() === '/' || peek() === '%') {
      const operator = take()
      const right = parsePower()
      if (operator === '*') value *= right
      else if (operator === '/') value /= right
      else value %= right
    }
    return value
  }

  const parseSum = (): number => {
    let value = parseProduct()
    while (peek() === '+' || peek() === '-') {
      const operator = take()
      const right = parseProduct()
      value = operator === '+' ? value + right : value - right
    }
    return value
  }

  try {
    const result = parseSum()
    if (position !== tokens.length || !Number.isFinite(result)) return null
    return result
  } catch {
    return null
  }
}

function formatNumber(value: number) {
  if (Number.isInteger(value)) return value.toLocaleString('en-US')
  return String(Math.round(value * 1e8) / 1e8)
}

function arithmetic(text: string): string | null {
  const cleaned = text
    .toLowerCase()
    .replace(/[?=]/g, ' ')
    .replace(/^(what is|what's|whats|calculate|compute|solve|evaluate|how much is|work out)\s+/, '')
    .trim()
  const percent = cleaned.match(/^(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)$/)
  if (percent) {
    const value = (Number(percent[1]) / 100) * Number(percent[2])
    return `${percent[1]}% of ${percent[2]} is **${formatNumber(value)}**.`
  }
  if (!/\d/.test(cleaned) || !/[+\-*/^%]|sqrt|sin|cos|tan|log|ln|abs/.test(cleaned)) return null
  const value = evaluateExpression(cleaned)
  if (value === null) return null
  return `\`${cleaned}\` = **${formatNumber(value)}**`
}

/* --- Extractive summariser --- */

function summarize(text: string): string {
  const body = text
    .replace(/^\s*(summarize|summarise|summary|tl;?dr|condense)\s*[:\-]?\s*/i, '')
    .trim()
  if (body.length < 80)
    return 'Paste the text you want summarised after the word *summarize* and I will pull out its key sentences.'
  const sentences = body.match(/[^.!?]+[.!?]+/g) ?? [body]
  if (sentences.length < 3) return `**Summary:** ${body.slice(0, 320)}${body.length > 320 ? '…' : ''}`

  const frequency = new Map<string, number>()
  for (const term of tokenize(body)) frequency.set(term, (frequency.get(term) ?? 0) + 1)

  const scored = sentences.map((sentence, index) => {
    const terms = tokenize(sentence)
    const score = terms.reduce((sum, term) => sum + (frequency.get(term) ?? 0), 0) / (terms.length + 3)
    return { sentence: sentence.trim(), score, index }
  })
  const chosen = [...scored]
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.min(3, Math.ceil(sentences.length / 3)))
    .sort((a, b) => a.index - b.index)

  return `**Summary**\n\n${chosen.map((item) => `- ${item.sentence}`).join('\n')}`
}

/* --- Idea generation from constraints --- */

const IDEA_TEMPLATES = [
  'Build a tiny version of {topic} you can finish in one afternoon, then show it to one person and watch where they get stuck.',
  'Take something you already like and change one constraint around {topic}: no budget, one week, or a single user.',
  'List the three most annoying parts of {topic}, then design the smallest possible fix for the worst one.',
  'Pair {topic} with an unrelated field: what would it look like as a game, a newsletter, a physical object, or a service?',
  'Ask who is currently underserved by {topic} and what a version made only for them would include.',
  'Find the step in {topic} that everyone complains about, and remove or automate just that step.',
]

function ideas(text: string): string {
  const topic =
    text
      .replace(/^.*?\b(ideas?|brainstorm(?:ing)?|suggest(?:ion)?s?)\b\s*(?:for|about|on)?\s*/i, '')
      .replace(/^(me|us|a|an|some)\s+/i, '')
      .replace(/[?.!]+$/, '')
      .trim() || 'a small weekend project'
  let seed = 0
  for (const char of topic) seed = (seed * 31 + char.charCodeAt(0)) % 997
  const picks = [0, 1, 2].map((offset) => IDEA_TEMPLATES[(seed + offset * 2) % IDEA_TEMPLATES.length])
  return `A few starting points for **${topic}**:\n\n${picks.map((idea) => `- ${idea.replace('{topic}', topic)}`).join('\n')}\n\nPick one and make it smaller still.`
}

/* --- Speech: turn a stored answer into a spoken reply --- */

const CONTRACTIONS: [string, string][] = [
  ['do not', "don't"],
  ['does not', "doesn't"],
  ['did not', "didn't"],
  ['is not', "isn't"],
  ['are not', "aren't"],
  ['was not', "wasn't"],
  ['were not', "weren't"],
  ['cannot', "can't"],
  ['can not', "can't"],
  ['will not', "won't"],
  ['would not', "wouldn't"],
  ['should not', "shouldn't"],
  ['it is', "it's"],
  ['that is', "that's"],
  ['there is', "there's"],
  ['they are', "they're"],
  ['we are', "we're"],
  ['you are', "you're"],
  ['have not', "haven't"],
  ['has not', "hasn't"],
  ['here is', "here's"],
  ['could not', "couldn't"],
  ['they will', "they'll"],
  ['we will', "we'll"],
  ['you will', "you'll"],
  ['i will', "I'll"],
  ['let us', "let's"],
  ['i have', "I've"],
]

/** Small rewrites that make stored prose sound more like speech. */
const TIDY: [RegExp, string][] = [
  [/\bin order to\b/gi, 'to'],
  [/\bdue to the fact that\b/gi, 'because'],
  [/\ba large number of\b/gi, 'many'],
  [/\butilis[ez]e\b/gi, 'use'],
  [/\bapproximately\b/gi, 'about'],
  [/However, /g, 'But '],
]

type Kind = 'what' | 'how' | 'why' | 'compare' | 'general'

const OPENERS: Record<Kind, string[]> = {
  what: ['Here is the short version.', 'Short version:', 'Right, the gist of it:'],
  how: ['Here is how it works.', 'Roughly, it goes like this.'],
  why: ['There is a clear reason for it.', 'It comes down to this.'],
  compare: ['The difference is worth being precise about.', 'They differ in a couple of ways.'],
  general: ['Right.', 'Here is what I have on that.'],
}

const CLOSERS = [
  'The longer version is there if you want it, just say "tell me more".',
  'There is more detail behind that if you say "tell me more".',
]

function kindOf(question: string): Kind {
  if (/\b(difference|versus|vs\.?|compared? (?:to|with)|better)\b/i.test(question)) return 'compare'
  if (/^\s*(what|what'?s|define|meaning of|who|which|where|when)\b/i.test(question)) return 'what'
  if (/^\s*how\b/i.test(question)) return 'how'
  if (/^\s*why\b/i.test(question)) return 'why'
  return 'general'
}

function hash(text: string) {
  let value = 0
  for (const char of text) value = (value * 31 + char.charCodeAt(0)) % 9973
  return value
}

function relax(text: string) {
  let out = text
  for (const [from, to] of CONTRACTIONS) {
    const capital = from[0].toUpperCase() + from.slice(1)
    const capitalTo = to[0].toUpperCase() + to.slice(1)
    out = out.replace(new RegExp(`\\b${from}\\b`, 'g'), to)
    out = out.replace(new RegExp(`\\b${capital}\\b`, 'g'), capitalTo)
  }
  for (const [pattern, replacement] of TIDY) out = out.replace(pattern, replacement)
  return out
}

/** Answers read as speech: contractions, plainer wording, and no stock preamble. */
function speak(answer: string, key: string, question = '') {
  if (answer.includes('`')) return answer
  const seed = hash(key)
  const body = relax(answer)
  // A lead-in should only orient the reader when the explanation is long. Short
  // factual answers get to the point on the first word.
  const openers = OPENERS[kindOf(question)]
  const opener = body.length > 260 && seed % 4 === 0 ? `${openers[seed % openers.length]}\n\n` : ''
  const closer = body.length > 220 && seed % 5 === 0 ? ` ${CLOSERS[seed % CLOSERS.length]}` : ''
  return `${opener}${body}${closer}`
}

/* --- Conversational replies that are not knowledge lookups --- */

const HOW_ARE_YOU =
  /^\s*(how are you|how r u|how are u|how'?s it going|how you doing|how do you feel)\b/i
const ASK_NAME = /^\s*(what'?s your name|what is your name|do you have a name|who named you)\b/i
const ASK_JOKE = /^\s*(tell me a joke|say something funny|make me laugh|know any jokes?|joke)\b/i
const GOODBYE = /^\s*(bye|goodbye|good ?night|goodnight|see you|see ya|cya|later)\b/i
const CAN_SPEAK = /^\s*(can you (?:talk|speak)|do you speak|are you able to speak|do you have a voice)\b/i
const ASK_HOW_OLD = /^\s*(how old are you|what'?s your age|when were you (?:made|born|created))\b/i
const ASK_WHERE = /^\s*(where are you(?: from| based)?|where do you live|where do you run)\b/i
const ASK_BODY =
  /^\s*(do you (?:sleep|eat|dream|have feelings|feel)|are you (?:real|alive|conscious|sentient))\b/i
const ASK_LIKE = /^\s*(what do you like|what'?s your favou?rite|do you have (?:hobbies|feelings))\b/i

const JOKES = [
  'Why do programmers prefer dark mode? Because light attracts bugs.',
  'There are two hard problems in computing: cache invalidation, naming things, and off-by-one errors.',
  'A SQL query walks into a bar, approaches two tables and asks: may I join you?',
]

function chatter(text: string): string | null {
  if (HOW_ARE_YOU.test(text))
    return 'Doing well, thanks, or at least running properly. I do not have moods to report, since I am a small trained model, but I am ready. What are you working on?'
  if (ASK_NAME.test(text))
    return 'I am Lumen, the study assistant in this project. The name belongs to the interface rather than the model, which has no identity beyond the files it was fitted on.'
  if (ASK_HOW_OLD.test(text))
    return 'I do not have an age in any meaningful sense. I was fitted from the files in this project and rebuilt whenever the data changes, so I am younger than this conversation.'
  if (ASK_WHERE.test(text))
    return 'I run inside this project as a Worker with no server of my own and no outside calls, so I do not really live anywhere.'
  if (ASK_BODY.test(text))
    return 'I am not alive or conscious, and I do not sleep, eat or dream. I match your question against a fixed knowledge base and phrase the result, which is why I would rather admit a gap than invent an answer.'
  if (ASK_LIKE.test(text))
    return 'I have no preferences, sadly. I can tell you what I am built to do: teach topics across the Classes 1–12 curriculum, make notes and revision sheets, give formulas, set practice questions, check an answer against the key, do arithmetic, summarise text and suggest starting ideas.'
  if (ASK_JOKE.test(text))
    return `${JOKES[hash(text) % JOKES.length]}\n\nThat one is canned, not generated, so I will not pretend it is original.`
  if (GOODBYE.test(text))
    return 'Goodbye, and thanks for the chat. My knowledge stays exactly where it is, so come back with any chapter you want to work through.'
  if (CAN_SPEAK.test(text))
    return 'In text, yes. I build sentences from a trained knowledge base rather than generating them from a language model, and I have no voice output: nothing here sends audio anywhere.'
  return null
}

/* --- Basic conversation: reacting to what someone says --- */

const ACK_ONLY =
  /^\s*(?:ok(?:ay)?|k|kk|cool|nice|great|awesome|perfect|good|right|sure|yeah|yep|yup|nah|nope|alright|fine|i see|makes sense|got it|fair enough|hmm+|haha+|lol|lmao|hehe|oh|ah|wow|interesting|true|same|indeed)\s*[.!]*\s*$/i

const FEELINGS: { pattern: RegExp; replies: string[] }[] = [
  {
    pattern:
      /^\s*i(?:'m| am|'m feeling| feel| am feeling)\s+(?:so |very |really |kind of |a bit |pretty )?(?:tired|exhausted|drained|sleepy|wiped out)\b/i,
    replies: [
      'Tired is tired, and no amount of chat fixes it. The boring fixes usually win: water, a short walk, and an earlier night. Want a five-minute distraction, or something practical for winding down?',
      'That sounds rough. I will not pretend to know your day, but being tired usually means the small things matter: food, a walk, and sleep. Want a distraction, or a few things that actually help at bedtime?',
    ],
  },
  {
    pattern: /^\s*i(?:'m| am)\s+(?:so |very |really |kind of |a bit |pretty )?bored\b/i,
    replies: [
      'Bored is fixable, at least. Pick a lane: something to learn, something to make, or a silly question to pass the time. Which one?',
      'Being bored is a decent excuse to try something small. I can explain a topic in about a minute, throw you three project ideas, or tell you a terrible joke. Pick one.',
    ],
  },
  {
    pattern:
      /^\s*i(?:'m| am| feel| am feeling)\s+(?:so |very |really |kind of |a bit |pretty )?(?:sad|down|low|upset|unhappy|depressed|heartbroken)\b/i,
    replies: [
      'I am sorry you are feeling that way. I am a small model, so I will not pretend to be a friend or a therapist, but I can keep you company for a bit. If it is heavy, please lean on someone real. Do you want a distraction, or to talk it through?',
      'That sounds genuinely hard. I am only text, and I am not a substitute for people who know you. But I am here: what happened, or would you rather I find something lighter to think about?',
    ],
  },
  {
    pattern:
      /^\s*i(?:'m| am| feel| am feeling)\s+(?:so |very |really |kind of |a bit |pretty )?(?:stressed|anxious|nervous|overwhelmed|panicking|worried)\b/i,
    replies: [
      'That is a lot to carry. One thing that helps is shrinking it: write down the single next action instead of the whole pile. What is the first thing on the list?',
      'Fair feeling. When everything feels urgent, pick the one thing that is due soonest and ignore the rest for twenty minutes. Want to tell me what is on your plate?',
    ],
  },
  {
    pattern:
      /^\s*i(?:'m| am| feel| am feeling)\s+(?:so |very |really |kind of |a bit |pretty )?(?:happy|good|great|excited|proud|relieved|glad)\b/i,
    replies: [
      'That is good to hear. What is the best part of it?',
      'Nice. I have no mood of my own to compare it to, but I am glad it is going well. What happened?',
    ],
  },
  {
    pattern:
      /^\s*i(?:'m| am| feel| am feeling)\s+(?:so |very |really |kind of |a bit |pretty )?(?:angry|annoyed|frustrated|furious|fed up)\b/i,
    replies: [
      'That sounds frustrating. Do you want to vent, or would it help to work out the next step?',
      'Fair. Anger usually points at something that needs changing. What set it off?',
    ],
  },
  {
    pattern: /^\s*i(?:'m| am)\s+(?:so |really |very )?(?:lonely|alone)\b/i,
    replies: [
      'That is a heavy feeling. I am only text, but I am here while you want company. What is on your mind?',
      'Being lonely is rough. I cannot replace people, but I can keep you talking for a while. How has your day been, good or bad?',
    ],
  },
  {
    pattern: /\b(?:long|rough|hard|tough|crazy) day\b/i,
    replies: [
      'Long days deserve a low bar for the evening: food, something mindless, and sleep. What made it long?',
      'Sorry to hear that. Do you want to decompress by talking about it, or should I find something easy to take your mind off it?',
    ],
  },
]

const LEARNING =
  /^\s*i(?:'m| am|'m just| am just|'m currently| am currently)?\s*(?:learning|studying|trying to learn|getting into|new to|starting to learn|reading about)\s+(.+?)\s*[.!]*$/i
const WANT_TO_LEARN =
  /^\s*i\s+(?:want|would like|'d like|need)\s+to\s+(?:learn|understand|get better at|study)\s+(.+?)\s*[.!]*$/i
const NEED_HELP =
  /^\s*i(?:'m| am|m)?\s*(?:need|struggling with|stuck on|stuck with)\s+(?:some\s+)?(?:help\s+(?:with|on)\s+)?(.+?)\s*[.!]*$/i
const OPINION =
  /^\s*(?:what do you think (?:about|of)|what'?s your (?:opinion|take) on|do you like|how do you feel about|any thoughts on)\s+(.+?)\s*[?.!]*$/i
const CRITICISM =
  /^\s*(?:(?:you'?re|you are|that'?s|that was|this is|u r)\s+(?:so |really |kind of |pretty )?(?:dumb|stupid|useless|wrong|bad|terrible|awful|pathetic|nonsense|garbage|trash)\b|(?:wrong|nope,? wrong|you didn'?t answer|that doesn'?t make sense|that makes no sense|you missed the point|not helpful)\s*[.!]*\s*$)/i
const CONFUSED =
  /^\s*(?:i\s+(?:do not|don'?t|dont)\s+(?:understand|get|follow)(?:\s+(.+?))?|i'?m\s+(?:confused|lost)|(?:that|this)\s+(?:does not|doesn'?t)\s+make\s+sense|what\s+do\s+you\s+mean|huh|eli5|in\s+simple\s+terms|explain\s+(?:that\s+)?(?:again|simpler|more\s+simply))\s*[?.!]*$/i
const WHY_ONLY = /^\s*why\s*[?.!]*$/i
const ASK_CLOCK =
  /^\s*(?:what(?:'s| is) the (?:time|date)|what time is it|what day is it(?: today)?|what(?:'s| is) today(?:'s date)?|current (?:time|date))\b/i
const ASK_WEATHER =
  /^\s*(?:what(?:'s| is) the weather|how(?:'s| is) the weather|is it (?:raining|snowing|sunny|cold|hot|warm)|will it rain)\b/i
const ASK_MEMORY = /^\s*(?:do|can|will) you remember\b/i
const ASK_PRESENCE = /^\s*(?:are you (?:there|awake|online)|you there|anyone (?:there|here))\s*[?.!]*$/i
const ASK_CHAT =
  /^\s*(?:let'?s (?:chat|talk)|can we (?:chat|talk)|i want to (?:chat|talk)|talk to me|keep me company|chat with me)\b/i
const QUESTION_INVITE =
  /^\s*(?:i have a question|can i ask (?:you )?(?:a question|something)|may i ask (?:a question|something)|quick question)\s*[?.!]*$/i
const AND_YOU = /^\s*(?:and you|what about you|how about you|you)\s*[?.!]*$/i
const SAY_NAME = /^\s*(?:my name is|i'?m|i am|call me)\s+([A-Za-z][A-Za-z'-]{1,20})\s*[.!]*$/i
const NOT_A_NAME = new Set(
  `tired bored sad happy fine good great ok okay here back sorry done ready curious confused lost
   hungry thirsty angry stressed anxious alone lonely busy free new old sure right wrong dumb`
    .split(/\s+/)
    .filter(Boolean),
)
const PERSONAL_STATEMENT =
  /^\s*(?:i(?:'m| am|'ve| have| just| had| got| feel| felt| can'?t| cannot| don'?t| do not| love| hate| like| wish| hope)\b|my\b|our\b|we\b)/i

/** Titles are headings; mid-sentence they should read like ordinary nouns. */
function lowerTitle(title: string) {
  return title.replace(/^[A-Z](?=[a-z])/, (letter) => letter.toLowerCase())
}

function firstSentences(text: string, count: number) {
  const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text]
  return sentences.slice(0, count).join(' ').trim()
}

function simplerReply(subject?: string) {
  if (!subject)
    return 'Let me start over. Ask me the question once more, or point at the part that felt off, and I will take it from there.'
  const match = search(subject)
  if (match.score >= MATCH_THRESHOLD && hasOverlap(match))
    return `Let me try that again, shorter.\n\n${relax(firstSentences(match.doc.answer, 1))}\n\nWhich part is still fuzzy? I can unpack just that piece.`
  return 'I am not sure which part I lost you on. Say it in your own words and I will try a different angle rather than repeating myself.'
}

function clockReply() {
  const now = new Date()
  return `My clock says **${now.toISOString().slice(11, 16)} UTC** on **${now.toISOString().slice(0, 10)}**. I have no timezone setting, so that is UTC rather than your local time.`
}

function learningReply(raw: string) {
  const topic = raw.replace(/^\s*(?:about|how to|how|to)\s+/i, '').trim()
  const match = search(topic)
  if (match.score >= MATCH_THRESHOLD && hasOverlap(match))
    return `Good choice — ${lowerTitle(match.doc.title)} is worth the time. ${relax(firstSentences(match.doc.answer, 2))}\n\nWhat part are you working on? I can go deeper, or you can say "more" for the longer version.`
  return `Nice, ${topic} is a decent thing to spend time on. I should be honest: my knowledge is a small fixed set, so I may not have much on it. Tell me what you are stuck on and I will say plainly whether I can help.`
}

function helpReply(raw: string) {
  const topic = raw.replace(/^\s*(?:with|on|about)\s+/i, '').trim()
  const match = search(topic)
  if (match.score >= MATCH_THRESHOLD && hasOverlap(match))
    return `Let's start with the basics, then narrow it down.\n\n${relax(firstSentences(match.doc.answer, 3))}\n\nTell me which part is failing and I will try to be more specific.`
  return `Tell me what is going wrong with ${topic} and I will help if it is in my knowledge base. If it is not, I will say so rather than making something up.`
}

function opinionReply(raw: string) {
  const topic = raw.trim()
  const match = search(topic)
  if (match.score >= MATCH_THRESHOLD && hasOverlap(match))
    return `I do not really have opinions, so I will not fake one. I can tell you what I know about ${lowerTitle(match.doc.title)}, though.\n\n${relax(match.doc.answer)}`
  return `I do not have opinions or preferences, honestly. I can explain ${topic} if it is in my knowledge base, but I would only be reciting, not judging. Want to try a topic I might know?`
}

function criticismReply(text: string) {
  if (!CRITICISM.test(text)) return null
  const replies = [
    'Fair criticism, and I would rather hear it than not. I am a small retrieval model: when a question sits outside my fixed training set, I miss or fall back. Try asking in different words, or name a topic from my list, and I will have a better shot.',
    'That is fair. My matching is literal, so small wording changes can land on a different entry. Send the question again another way and I will try again.',
    'Fair hit. I do not generate answers, I retrieve them, so I am only as good as the entries I was fitted on. Give me the question once more and I will tell you honestly whether it is in there.',
  ]
  return replies[hash(text) % replies.length]
}

/** Social replies that keep a conversation going instead of failing to a lookup. */
function converse(text: string, previousUser?: string): string | null {
  const trimmed = text.trim()

  const criticism = criticismReply(trimmed)
  if (criticism) return criticism

  const opinion = trimmed.match(OPINION)
  if (opinion) return opinionReply(opinion[1])

  const learn = trimmed.match(LEARNING) ?? trimmed.match(WANT_TO_LEARN)
  if (learn) return learningReply(learn[1])

  const help = trimmed.match(NEED_HELP)
  if (help?.[1]) return helpReply(help[1])

  for (const feeling of FEELINGS)
    if (feeling.pattern.test(trimmed))
      return feeling.replies[hash(trimmed) % feeling.replies.length]

  const confused = trimmed.match(CONFUSED)
  if (confused) {
    const subject = confused[1]?.trim()
    const usable =
      subject && !/^(?:this|that|it|you|your answer|the answer|what you mean)$/i.test(subject)
    return simplerReply(usable ? subject : previousUser)
  }

  if (ASK_CLOCK.test(trimmed)) return clockReply()
  if (ASK_WEATHER.test(trimmed))
    return 'I cannot check. I have no internet access at runtime, so I genuinely do not know the weather anywhere. A real forecast app will beat me at that.'
  if (ASK_MEMORY.test(trimmed))
    return 'Not between sessions. I do not store anything: each request carries the conversation so far, and nothing is saved on the server. So tell me again and I will work with what is in front of me.'
  if (ASK_PRESENCE.test(trimmed)) return 'I am here. What is on your mind?'
  if (ASK_CHAT.test(trimmed))
    return 'Happy to. I am better at topics than gossip, so pick one: I can explain something from science or code, suggest something to make, or listen while you describe your day.'
  if (QUESTION_INVITE.test(trimmed))
    return 'Of course, go ahead. If it is outside my knowledge base I will tell you rather than guess.'
  if (AND_YOU.test(trimmed)) return 'I do not have a day to report, honestly. What about you — how is yours going?'

  const name = trimmed.match(SAY_NAME)
  if (name && !NOT_A_NAME.has(name[1].toLowerCase())) {
    const proper = name[1].replace(/\b\w/g, (letter) => letter.toUpperCase())
    return `Nice to meet you, ${proper}. I will not remember it once this page is gone, but I will use it while we talk. What would you like to get into?`
  }

  if (ACK_ONLY.test(trimmed)) {
    const replies = [
      'Alright. What else is on your mind?',
      'Got it. Ask me anything, or tell me what kind of day you are having.',
      'Cool. I am here if you want to dig into something.',
      'Fair enough. Want a topic, a summary, or just some company for a minute?',
    ]
    return replies[hash(trimmed) % replies.length]
  }

  // "i want to know about X" is a lookup, not a personal statement.
  if (
    PERSONAL_STATEMENT.test(trimmed) &&
    !trimmed.includes('?') &&
    stripLead(trimmed) === trimmed
  )
    return 'That sounds worth talking about, though I should be honest that I mostly know topics rather than people. Tell me a bit more, or we can switch to something I can actually help with.'

  return null
}

/* --- Conversation --- */

/* --- Basic manners --- */

const PLEASE_ONLY = /^\s*(?:please|pls|plz|kindly)\s*[?.!]*$/i
const WELCOME = /^\s*(?:you'?re welcome|you are welcome|no problem|no worries|anytime|np)\b/i
const APOLOGY_FULL = /^\s*(?:sorry|apologies|i apologi[sz]e|my bad|excuse me|pardon me)\b/i
const COMPLIMENT = [
  /^\s*(?:good bot|nice work|well done|great job|you rock|you'?re the best)\b/i,
  /^\s*(?:that'?s|that was|this is)\s+(?:great|awesome|amazing|perfect|brilliant|really helpful|so helpful|lovely)\s*[!. ]*$/i,
  /^\s*i (?:love|like) you\b/i,
  /^\s*(?:thank you|thanks)[, ]+(?:that'?s|that was)\s+(?:great|helpful|perfect)/i,
]

function politeness(text: string): string | null {
  const trimmed = text.trim()
  const words = tokenize(trimmed).length
  if (PLEASE_ONLY.test(trimmed)) return 'Of course. What would you like me to explain?'
  if (WELCOME.test(trimmed) && words <= 4) return 'Thanks for saying so. What else can I help with?'
  if (words <= 8 && COMPLIMENT.some((pattern) => pattern.test(trimmed)))
    return 'That is kind of you. I am only a small trained model, but I am glad that landed well.'
  if (
    APOLOGY_FULL.test(trimmed) &&
    words <= 3 &&
    !/\b(?:what|how|why|who|when|where|which|explain|tell)\b/i.test(trimmed)
  )
    return 'No need to apologise at all. Ask me whatever you like.'
  return null
}

/** "please, what is gravity?" and "sorry to bother you, ..." still get answered. */
function stripManners(text: string) {
  return text
    .trim()
    .replace(/^\s*(?:please|pls|plz|kindly)[\s,.:!-]+/i, '')
    .replace(
      /^\s*(?:sorry|apologies|my bad)(?:\s+(?:to bother you|to ask again|for the (?:basic|silly|dumb) question))?[\s,.:!-]+/i,
      '',
    )
    .replace(/^\s*(?:thanks|thank you)(?:\s+(?:for|that))?[\s,.:!-]+(?=\S)/i, '')
    .replace(/^\s*(?:hi|hello|hey)(?: there| lumen)?[\s,.:!-]+(?=\S)/i, '')
    .replace(/\s*,?\s*(?:please|pls|plz)\s*[?.!]*$/i, '')
    .trim()
}

function fallback() {
  return `Sorry, I do not have anything reliable on that. My knowledge is a small fixed set fitted inside this project, so I would rather say so than invent an answer.\n\nThings I can help with:\n${FALLBACK_TOPICS.map((line) => `- ${line}`).join('\n')}\n\nFor school work, try **"notes on integers for class 7"**, **"formulas for class 10 maths"** or **"test me on photosynthesis"**. I can also summarise text you paste, do arithmetic and suggest starting ideas.`
}

function aboutReply(self: boolean) {
  if (self)
    return `I'm Lumen, a study assistant for Classes 1–12 built around the NCERT/CBSE curriculum, with ICSE, state board and international syllabuses in mind too. There's no external AI service behind me: a small model and a curriculum map are fitted inside this project and run in the Worker.\n\nI can teach a topic at your class level, make notes, build a revision sheet, list formulas, set a short test and check an answer against the key. I'm honest about the limits: I can't browse, I don't remember between sessions, and I won't invent textbook content I don't have.`
  return `I'm a small study assistant trained inside this project, with no external AI service and no runtime network calls. Here is what I can do:\n\n- Teach topics across Classes 1–12 in maths, science, social science, commerce, computer science and languages, adapting the depth to the class you name.\n- Make notes, a revision sheet or a full lesson on a named chapter.\n- List formulas with what each symbol means, for a topic, subject or class.\n- Set a short test, give practice questions, and check your written answer against the key points.\n- Do arithmetic, summarise text you paste and suggest starting ideas.\n\nI default to NCERT/CBSE and will say so when something sits outside the material I was fitted on, rather than guessing.`
}

function greeting() {
  return `Hi, I'm Lumen, a study assistant for Classes 1–12 with an NCERT/CBSE orientation. I can explain a topic, make notes, build a revision sheet, list formulas, set you a short test and check an answer against the key. I'm a small model trained here in this project rather than a general assistant, so I'll say plainly when something sits outside what I know. What are we studying?`
}

function thanks(text: string) {
  const replies = [
    'You are welcome. Glad that helped.',
    'Anytime. That is what I am here for.',
    'No problem at all. Ask away if something else comes up.',
  ]
  return replies[hash(text) % replies.length]
}

function detail(doc: Doc) {
  if (doc.more) return doc.more
  const related = doc.related ? byId.get(doc.related) : undefined
  if (related) return `Here is a related idea, **${related.title}**:\n\n${related.answer}`
  return `That is about all I have on ${doc.title}. Ask me about another topic, or paste some text and I can summarise it.`
}

const MORE_PHRASE = /^\s*(tell me more|more|go on|go deeper|elaborate|explain (?:further|more)|what else|keep going|say more)\b/i

// Statistically learned intents are still checked against the shape of the text,
// so a confidently wrong label cannot hijack a real question.
const GREETING_TEXT =
  /^\s*(hi|hii|heyy?|hello|hullo|helloo|yo|sup|hiya|howdy|gm|ta|greetings|morning|evening|afternoon|good (?:morning|evening|afternoon|day)|hiya|hey (?:there|bot|you)?|hi (?:there|bot)|hello (?:there|again)|anyone (?:there|here)|what'?s up|how are you)\b/i
const THANKS_TEXT =
  /(?:\bthanks?\b|\bthank you\b|\bthx\b|\bty\b|\bcheers\b|appreciate it|much appreciated|that helps|that was helpful|perfect thanks)/i
const IDEA_TEXT =
  /(?:\bideas?\b|brainstorm|suggest|something to (?:build|make|try)|what should i (?:build|make|try)|what could i (?:build|make)|\bout of ideas\b|invent)/i

const ABOUT =
  /^\s*(what can you do|what are you good at|how can you help(?: me)?|what topics (?:do you know|can you (?:help with|answer))|who are you|what are you|are you (?:chatgpt|gpt|claude|an ai)|who trained you|how (?:were|are) you (?:made|trained)|help)\s*[?.!]*\s*$/i
const SELF =
  /^\s*((?:who|what) are you|are you (?:chatgpt|gpt|claude|an ai)|how (?:were|are) you (?:made|trained)|who trained you)\s*[?.!]*\s*$/i

export function reply(messages: ChatMessage[]): string {
  const last = messages.at(-1)
  if (!last || last.role !== 'user') return greeting()
  const text = last.content
  const previousUser = [...messages]
    .reverse()
    .slice(1)
    .find((message) => message.role === 'user')?.content

  const courtesy = politeness(text)
  if (courtesy) return courtesy

  const question = stripManners(text) || text
  const smallTalk = chatter(text) ?? (question === text ? null : chatter(question))
  if (smallTalk) return relax(smallTalk)

  // Conversational replies are written plainly here and relaxed the same way as
  // stored answers, so they use contractions instead of reading like a manual.
  const conversation = converse(question, previousUser)
  if (conversation) return relax(conversation)

  if (ABOUT.test(text)) return aboutReply(SELF.test(text))

  const capitalAnswer = capital(question)
  if (capitalAnswer) return capitalAnswer

  const calculation = arithmetic(question)
  if (calculation) return calculation

  if (/^\s*(summarize|summarise|tl;?dr|summary|condense)\b/i.test(question))
    return summarize(question)

  // Teaching requests (notes, revision, formulas, tests, class-level lessons)
  // are answered from the curriculum map. Anything it cannot place falls back
  // to the ordinary knowledge lookup below.
  const teaching = educationReply(question, previousUser)
  if (teaching) return teaching

  const intentLabel = classify(question)
  const words = tokenize(question)

  const BARE_GREETING =
    /^\s*(?:hi|hii|heyy?|hello|hullo|helloo|yo|sup|hiya|howdy|gm|greetings|morning|evening|afternoon|good (?:morning|evening|afternoon|day)|what'?s up)(?:[\s,!.]*(?:there|lumen|bot|again|all|you))?[\s!.?]*$/i
  if (BARE_GREETING.test(text)) return greeting()
  if (THANKS_TEXT.test(text) && tokenize(text).length <= 4) return thanks(text)
  if (intentLabel === 'summarize' && question.length > 120) return summarize(question)
  if (intentLabel === 'idea' && IDEA_TEXT.test(question)) return ideas(question)

  // "why?" asks for the reason behind the previous answer, not the summary again.
  if (WHY_ONLY.test(question) && previousUser) {
    const { doc } = search(previousUser)
    const deeper = doc.more || doc.related ? detail(doc) : doc.answer
    return speak(deeper, `${doc.id} why ${question}`, question)
  }

  if (MORE_PHRASE.test(question) || (intentLabel === 'more' && words.length <= 4)) {
    if (!previousUser)
      return 'I am not sure what to expand on yet. Ask me something first, like *what is a black hole?*, please.'
    const { doc } = search(previousUser)
    return speak(detail(doc), `${doc.id} more`, previousUser)
  }

  // "big o" is a notation, not a question about the Big Bang.
  if (/\bbig[ -]?o\b/i.test(question)) {
    const doc = byId.get('big-o')
    if (doc) return speak(doc.answer, `big-o ${question}`, question)
  }

  const lookup = stripLead(question) || question
  let match = search(lookup)

  // Short follow-ups inherit context from the previous turn.
  if (words.length <= 2 && previousUser) {
    const combined = search(`${previousUser} ${question}`)
    if (combined.score > match.score) match = combined
    if (!words.length) return detail(match.doc)
  }

  if (match.score >= MATCH_THRESHOLD && hasOverlap(match))
    return speak(match.doc.answer, `${match.doc.id} ${question}`, question)
  return fallback()
}

export const modelInfo = {
  documents: model.documents,
  terms: model.terms,
  trainedAt: model.trainedAt,
  intents: model.intentModel.labels,
}
