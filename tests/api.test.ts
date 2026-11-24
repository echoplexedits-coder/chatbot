import { afterEach, expect, test } from 'bun:test'
import { app } from '../src/worker/index'
import { reply } from '../src/worker/model'
import { TOPICS } from '../src/worker/education'

const env = {} as never
const post = (body: string, ip = '10.0.0.1') =>
  app.request(
    '/api/chat',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'cf-connecting-ip': ip },
      body,
    },
    env,
  )

const realFetch = globalThis.fetch
afterEach(() => {
  globalThis.fetch = realFetch
})

test('the model was trained: documents, terms and intents are loaded', async () => {
  const body = (await (await app.request('/api/health', {}, env)).json()) as {
    ok: boolean
    model: { documents: number; terms: number; trainedAt: string }
  }
  expect(body.ok).toBe(true)
  expect(body.model.documents).toBeGreaterThan(120)
  expect(body.model.terms).toBeGreaterThan(500)
  expect(Number.isNaN(Date.parse(body.model.trainedAt))).toBe(false)
})

test('answers questions from its trained knowledge', () => {
  expect(reply([{ role: 'user', content: 'what is photosynthesis' }])).toContain(
    'chlorophyll',
  )
  expect(reply([{ role: 'user', content: 'how does HTTPS differ from HTTP' }])).toContain(
    'TLS',
  )
  expect(reply([{ role: 'user', content: 'what is big o notation' }])).toContain(
    'O(n log n)',
  )
})

test('answers casual and misspelled phrasings', () => {
  const cases: [string, RegExp][] = [
    ['waht is photosynthsis', /chlorophyll/i],
    ['why is the sky blu', /rayleigh/i],
    ['wat is a blak hole', /event horizon/i],
    ['wat is an api', /endpoint|contract/i],
    ['waht is recusion', /base case/i],
    ['how do i make a buget', /fixed costs|needs/i],
    ['relly cant sleep at night', /wake time|caffeine/i],
    ['how do i stop procastinating', /first step|phone/i],
  ]
  for (const [question, expected] of cases)
    expect(reply([{ role: 'user', content: question }])).toMatch(expected)
})

test('answers the expanded topic set', () => {
  expect(reply([{ role: 'user', content: 'how do earthquakes happen' }])).toMatch(/fault/i)
  expect(reply([{ role: 'user', content: 'waht is inflation' }])).toMatch(/price level/i)
  expect(reply([{ role: 'user', content: 'how do solar panels work' }])).toMatch(
    /photovoltaic/i,
  )
  expect(reply([{ role: 'user', content: 'correlation vs causation' }])).toMatch(
    /confounder/i,
  )
  expect(reply([{ role: 'user', content: 'how do i center a div' }])).toMatch(/flex/i)
})

test('answers the new knowledge entries', () => {
  const cases: [string, RegExp][] = [
    ['how do antibiotics work', /penicillin|bacteria/i],
    ['how big is the universe', /13\.8 billion/i],
    ['how do stars form', /fusion|hydrogen/i],
    ['what is supply and demand', /price/i],
    ['how does insurance work', /premium|risk/i],
    ['what is a hash table', /key/i],
    ['what is dns', /DNS|phone book/i],
    ['what is web accessibility', /screen reader|alt text/i],
  ]
  for (const [question, expected] of cases)
    expect(reply([{ role: 'user', content: question }])).toMatch(expected)
})

test('answers general knowledge facts', () => {
  const cases: [string, RegExp][] = [
    ['what is the capital of australia', /canberra/i],
    ['capitol of france', /paris/i],
    ['how many bones do we have', /206/],
    ['when did world war 2 end', /1945/],
    ['how many planets are there', /eight/i],
    ['how do i convert celsius to fahrenheit', /9\/5|1\.8/],
    ['why do onions make you cry', /sulf|sulph/i],
    ['how many hearts does an octopus have', /three hearts/i],
  ]
  for (const [question, expected] of cases)
    expect(reply([{ role: 'user', content: question }])).toMatch(expected)

  expect(
    reply([{ role: 'user', content: 'what is the capital of mongolia' }]),
  ).toMatch(/do not have the capital of Mongolia/)
})

test('speaks in conversational sentences without a stock preamble', () => {
  const answer = reply([{ role: 'user', content: 'what is dna' }])
  expect(answer).toMatch(/It's a double helix/)
  expect(answer).not.toMatch(/^(?:Good question|Certainly|Absolutely|As an AI)/i)

  const how = reply([{ role: 'user', content: 'how do solar panels work' }])
  expect(how).toMatch(/photovoltaic/i)
  expect(how).not.toMatch(/^Good question/i)

  // A short factual question gets answered on the first word.
  expect(
    reply([{ role: 'user', content: 'what is the capital of france' }]).startsWith(
      'The capital of France',
    ),
  ).toBe(true)
})

test('re-explains instead of repeating when the user is confused', () => {
  const messages = [
    { role: 'user' as const, content: 'what is a black hole' },
    { role: 'assistant' as const, content: 'A black hole is...' },
    { role: 'user' as const, content: "i don't understand" },
  ]
  const answer = reply(messages)
  expect(answer).toMatch(/black hole|horizon/i)
  expect(answer).toMatch(/shorter|again|fuzzy/i)
  expect(reply([{ role: 'user', content: "i don't understand recursion" }])).toMatch(
    /calling itself|recursion/i,
  )
  expect(reply([{ role: 'user', content: 'eli5' }])).toMatch(/start over|question/i)
})

test('has basic manners', () => {
  expect(reply([{ role: 'user', content: 'please' }])).toMatch(/Of course/)
  expect(reply([{ role: 'user', content: 'sorry' }])).toMatch(/No need to apologise/)
  expect(reply([{ role: 'user', content: 'good bot' }])).toMatch(/kind of you/)
  expect(reply([{ role: 'user', content: 'no problem' }])).toMatch(/Thanks for saying so/)
  expect(reply([{ role: 'user', content: 'thank you' }])).toMatch(
    /welcome|Anytime|No problem/,
  )
})

test('stays polite without dropping the actual question', () => {
  expect(reply([{ role: 'user', content: 'please, what is gravity?' }])).toMatch(/mass/)
  expect(
    reply([{ role: 'user', content: 'sorry to bother you, what is dna?' }]),
  ).toMatch(/double helix/)
  expect(reply([{ role: 'user', content: 'could you explain photosynthesis please' }])).toMatch(
    /chlorophyll/i,
  )
  expect(reply([{ role: 'user', content: 'hi, what is a black hole?' }])).toMatch(
    /event horizon/i,
  )
})

test('handles small talk honestly', () => {
  expect(reply([{ role: 'user', content: 'how are you?' }])).toMatch(/doing well/i)
  expect(reply([{ role: 'user', content: "what's your name?" }])).toMatch(/Lumen/)
  expect(reply([{ role: 'user', content: 'tell me a joke' }])).toMatch(/canned/i)
  expect(reply([{ role: 'user', content: 'can you speak?' }])).toMatch(/no voice output/i)
  expect(reply([{ role: 'user', content: 'how old are you?' }])).toMatch(/don't have an age/i)
  expect(reply([{ role: 'user', content: 'do you sleep?' }])).toMatch(/not alive/i)
  expect(reply([{ role: 'user', content: 'where do you live?' }])).toMatch(/Worker/i)
})

test('attempts basic conversation instead of falling back', () => {
  const cases: [string, RegExp][] = [
    ['i am bored', /bored/i],
    ['i am exhausted', /tired/i],
    ['i had a long day', /long day|decompress|low bar/i],
    ['i am learning about black holes', /event horizon|black hole/i],
    ['im stuck on recursion', /base case/i],
    ['you are dumb', /fair/i],
    ['ok', /mind|anything|dig|company/i],
    ['my name is sam', /Sam/],
    ["what's the weather", /no internet/i],
    ['what time is it', /UTC/],
    ['do you remember me', /between sessions/i],
    ["let's chat", /topics than gossip|listen|suggest/i],
    ['i just adopted a puppy', /worth talking about/i],
  ]
  for (const [message, expected] of cases)
    expect(reply([{ role: 'user', content: message }])).toMatch(expected)
})

test('reacts to feelings and keeps the conversation going', () => {
  for (const message of ['i am tired', 'i am sad', 'i am stressed', 'i am excited']) {
    const answer = reply([{ role: 'user', content: message }])
    expect(answer).not.toContain('do not have anything reliable')
    expect(answer.length).toBeGreaterThan(40)
    expect(answer).toContain('?')
  }
  expect(reply([{ role: 'user', content: 'i am sad' }])).not.toMatch(/as an ai/i)
})

test('has opinions only about what it was trained on', () => {
  const answer = reply([{ role: 'user', content: 'what do you think about black holes' }])
  expect(answer).toMatch(/don't really have opinions/i)
  expect(answer).toMatch(/event horizon/i)
  expect(reply([{ role: 'user', content: 'do you like music?' }])).toMatch(
    /don't have opinions/i,
  )
})

test('ignores conversational lead-ins before looking up a topic', () => {
  expect(reply([{ role: 'user', content: 'can you tell me about gravity' }])).toMatch(/mass/)
  expect(reply([{ role: 'user', content: 'i want to know about photosynthesis' }])).toMatch(
    /chlorophyll/i,
  )
  expect(reply([{ role: 'user', content: 'explain dns to me' }])).toMatch(/DNS|phone book/i)
})

test('handles follow-ups using the previous turn', () => {
  const messages = [
    { role: 'user' as const, content: 'what is a black hole' },
    { role: 'assistant' as const, content: 'A black hole is a region...' },
    { role: 'user' as const, content: 'tell me more' },
  ]
  const answer = reply(messages)
  expect(answer.length).toBeGreaterThan(60)
  expect(answer.toLowerCase()).toMatch(/singularity|spacetime|horizon/)

  const followUp = reply([
    { role: 'user' as const, content: 'what is photosynthesis' },
    { role: 'assistant' as const, content: 'Photosynthesis is...' },
    { role: 'user' as const, content: 'why?' },
  ])
  expect(followUp.toLowerCase()).toMatch(/chlorophyll|calvin|thylakoid|carbon/)
})

test('does arithmetic and percentages locally', () => {
  expect(reply([{ role: 'user', content: 'what is 12 * 7' }])).toContain('**84**')
  expect(reply([{ role: 'user', content: '15% of 80' }])).toContain('**12**')
  expect(reply([{ role: 'user', content: '(3 + 4) * 5' }])).toContain('**35**')
  expect(reply([{ role: 'user', content: 'sqrt(144)' }])).toContain('12')
})

test('summarises pasted text extractively', () => {
  const text =
    'The Moon causes tides by pulling on the oceans. Water nearest the Moon bulges toward it. ' +
    'The Earth also moves away from the water on the far side, creating a second bulge. ' +
    'As the Earth rotates through these bulges, coasts see two high tides each day. ' +
    'The Sun adds roughly half the Moon tidal force. When the Sun and Moon align, the range is larger.'
  const answer = reply([{ role: 'user', content: `summarize: ${text}` }])
  expect(answer).toContain('**Summary**')
  expect(answer.length).toBeLessThan(text.length)
})

test('generates ideas for a topic', () => {
  const answer = reply([{ role: 'user', content: 'give me an idea for a side project' }])
  expect(answer).toContain('side project')
  expect(answer.split('\n- ').length).toBeGreaterThanOrEqual(3)
})

test('describes itself and its capabilities', () => {
  expect(reply([{ role: 'user', content: 'what can you do?' }])).toContain(
    'no external AI service',
  )
  expect(reply([{ role: 'user', content: 'who are you?' }])).toContain('Lumen')
})

test('admits when something is outside its knowledge', () => {
  const answer = reply([
    { role: 'user', content: 'who won the 1998 world cup final in football' },
  ])
  expect(answer).toContain('do not have anything reliable')
  expect(answer).toContain('small fixed set')
})

test('never calls the network', async () => {
  let called = false
  globalThis.fetch = (async () => {
    called = true
    return new Response('{}')
  }) as unknown as typeof fetch
  expect(reply([{ role: 'user', content: 'what is gravity' }])).toContain('mass')
  await post(
    JSON.stringify({ messages: [{ role: 'user', content: 'what is gravity' }] }),
  )
  expect(called).toBe(false)
})

test('chat validates the conversation and streams the local answer', async () => {
  for (const body of [
    '{}',
    'null',
    'not json',
    '{"messages":[]}',
    '{"messages":[{"role":"system","content":"hi"}]}',
    '{"messages":[{"role":"assistant","content":"hi"}]}',
    '{"messages":[{"role":"user","content":"  "}]}',
    '{"messages":[{"role":"user","content":"hi","extra":1}]}',
    JSON.stringify({ messages: [{ role: 'user', content: 'x'.repeat(2001) }] }),
  ])
    expect((await post(body, '10.0.0.2')).status).toBe(400)

  const response = await post(
    JSON.stringify({ messages: [{ role: 'user', content: 'what is recursion' }] }),
    '10.0.0.3',
  )
  expect(response.status).toBe(200)
  expect(response.headers.get('content-type')).toContain('text/event-stream')
  expect(response.headers.get('cache-control')).toBe('no-store')
  const text = await response.text()
  expect(text).toContain('"delta"')
  expect(text).toContain('"done":true')
  expect(text).toContain('base case')
})

test('chat rate limits one client', async () => {
  const body = JSON.stringify({ messages: [{ role: 'user', content: 'hello' }] })
  for (let index = 0; index < 20; index++)
    expect((await post(body, '10.9.9.9')).status).toBe(200)
  const limited = await post(body, '10.9.9.9')
  expect(limited.status).toBe(429)
  expect(await limited.json()).toMatchObject({ error: { code: 'rate_limited' } })
})

test('unknown API routes stay JSON', async () => {
  const missing = await app.request('/api/missing', {}, env)
  expect(missing.status).toBe(404)
  const spec = (await (await app.request('/openapi.json', {}, env)).json()) as {
    paths: Record<string, unknown>
  }
  expect(Object.keys(spec.paths).sort()).toEqual(['/api/chat', '/api/health', '/api/me'])
})

test('requires a session once accounts are configured', async () => {
  const authEnv = {
    FREEBUFF_CLIENT_ID: 'test-client',
    APP_ORIGIN: 'https://lumen.test',
  } as never

  const me = (await (await app.request('/api/me', {}, authEnv)).json()) as {
    signInAvailable: boolean
    user: unknown
  }
  expect(me.signInAvailable).toBe(true)
  expect(me.user).toBeNull()

  const chat = await app.request(
    '/api/chat',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'cf-connecting-ip': '10.0.0.4' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hello' }] }),
    },
    authEnv,
  )
  expect(chat.status).toBe(401)
  expect(await chat.json()).toMatchObject({ error: { code: 'sign_in_required' } })

  // Health stays public so the page can report readiness before signing in.
  expect((await app.request('/api/health', {}, authEnv)).status).toBe(200)
})

test('teaches a curriculum topic with its key ideas', () => {
  const answer = reply([{ role: 'user', content: 'teach me integers for class 7' }])
  expect(answer).toContain('Integers')
  expect(answer).toMatch(/number line/i)
  expect(answer).toContain('Key ideas')
})

test('makes notes and a revision sheet', () => {
  const notes = reply([{ role: 'user', content: 'make notes on photosynthesis' }])
  expect(notes).toContain('Notes — Photosynthesis')
  expect(notes).toMatch(/chlorophyll/i)
  const revision = reply([{ role: 'user', content: 'revision sheet on the French Revolution' }])
  expect(revision).toContain('Revision sheet')
  expect(revision).toContain('French Revolution')
})

test('lists formulas for a topic and for a whole class', () => {
  const topic = reply([{ role: 'user', content: 'formulas for quadratic equations' }])
  expect(topic).toContain('b² − 4ac')
  const sheet = reply([{ role: 'user', content: 'give me class 10 maths formulas' }])
  expect(sheet).toContain('Formula sheet')
  expect(sheet.toLowerCase()).toContain('quadratic')
})

test('sets a test and shows the key when asked', () => {
  const quiz = reply([{ role: 'user', content: 'test me on class 10 quadratic equations' }])
  expect(quiz).toContain('Q1')
  expect(quiz).toMatch(/answers/i)
  const key = reply([
    { role: 'user', content: 'test me on class 10 quadratic equations' },
    { role: 'assistant', content: quiz },
    { role: 'user', content: 'answers' },
  ])
  expect(key).toContain('Answer key')
})

test('checks a written answer against the key points', () => {
  const checked = reply([
    {
      role: 'user',
      content:
        'check my answer on quadratic equations: the roots come from the quadratic formula with the discriminant',
    },
  ])
  expect(checked).toContain('key ideas')
  expect(checked).toContain('model answer')
})

test('asks for a topic when only a class and subject are given', () => {
  const answer = reply([{ role: 'user', content: 'make notes on class 9 science' }])
  expect(answer).toContain('Class 9 Science')
  expect(answer).toMatch(/topic name/i)
})

test('the curriculum map has unique ids and covers every class band', () => {
  const ids = TOPICS.map((topic) => topic.id)
  expect(new Set(ids).size).toBe(ids.length)
  expect(TOPICS.length).toBeGreaterThan(450)
  for (const level of [1, 4, 6, 8, 10, 12]) {
    expect(TOPICS.some((topic) => topic.classes.includes(level))).toBe(true)
  }
})

test('senior chapters added for full coverage are reachable in teaching modes', () => {
  expect(
    reply([{ role: 'user', content: 'teach me inverse trigonometric functions for class 12' }]),
  ).toContain('Inverse trigonometric functions')
  expect(reply([{ role: 'user', content: 'notes on the solid state class 12' }])).toContain(
    'The solid state',
  )
  expect(reply([{ role: 'user', content: 'explain kinetic theory class 11' }])).toContain(
    'Kinetic theory',
  )
  expect(reply([{ role: 'user', content: 'notes on the cold war class 12' }])).toContain(
    'Cold War',
  )
  expect(reply([{ role: 'user', content: 'teach me disaster management class 9' }])).toContain(
    'Disaster management',
  )
})
