import { createFreebuffAuth, type FreebuffAuth, type FreebuffUser } from '@freebuff/auth'
import { createRoute, OpenAPIHono, z } from '@hono/zod-openapi'
import { bodyLimit } from 'hono/body-limit'
import { modelInfo, reply, type ChatMessage } from './model'

const MAX_MESSAGE_CHARS = 2000
const MAX_MESSAGES = 24
const MAX_TOTAL_CHARS = 24_000
const RATE_LIMIT = 20
const RATE_WINDOW_MS = 10 * 60 * 1000

type Bindings = {
  ASSETS: Fetcher
  FREEBUFF_CLIENT_ID?: string
  FREEBUFF_ISSUER?: string
  APP_ORIGIN?: string
}

type Variables = { user: FreebuffUser | null }

/**
 * One auth instance per isolate, so discovery and signing keys are cached.
 * The client ID and origin are public configuration injected at deploy time;
 * without them (plain `bun run dev`) the app runs without accounts.
 */
let auth: FreebuffAuth | undefined
function getAuth(env: Bindings) {
  if (!auth && env.FREEBUFF_CLIENT_ID && env.APP_ORIGIN)
    auth = createFreebuffAuth({
      clientId: env.FREEBUFF_CLIENT_ID,
      issuer: env.FREEBUFF_ISSUER,
      baseUrl: env.APP_ORIGIN,
      // Loopback only: lets `wrangler dev` exercise the real flow. Deploys use https.
      allowInsecureLocalhost: /^http:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(
        env.APP_ORIGIN,
      ),
    })
  return auth
}

const messageSchema = z
  .object({
    role: z.enum(['user', 'assistant']),
    content: z.string().trim().min(1).max(MAX_MESSAGE_CHARS),
  })
  .strict()

const chatSchema = z
  .object({ messages: z.array(messageSchema).min(1).max(MAX_MESSAGES) })
  .strict()

const errorSchema = z.object({
  error: z.object({ code: z.string(), message: z.string() }),
})
const json = <T extends z.ZodType>(schema: T) => ({
  'application/json': { schema },
})

export const app = new OpenAPIHono<{ Bindings: Bindings; Variables: Variables }>({
  defaultHook: (result, c) => {
    if (!result.success)
      return c.json(
        {
          error: {
            code: 'invalid_input',
            message: `Check your message: each one can be up to ${MAX_MESSAGE_CHARS.toLocaleString('en-US')} characters.`,
          },
        },
        400,
      )
  },
})

/** Small per-instance limiter so one client cannot hammer the model. */
const hits = new Map<string, number[]>()

function limited(key: string) {
  const now = Date.now()
  const recent = (hits.get(key) ?? []).filter((time) => now - time < RATE_WINDOW_MS)
  if (recent.length >= RATE_LIMIT) {
    hits.set(key, recent)
    return true
  }
  recent.push(now)
  hits.set(key, recent)
  if (hits.size > 5000)
    for (const [k, times] of hits)
      if (!times.some((time) => now - time < RATE_WINDOW_MS)) hits.delete(k)
  return false
}

// Authorization lives on the server: a hidden composer is not access control.
app.use('/api/chat', async (c, next) => {
  const session = getAuth(c.env)
  if (!session) {
    c.set('user', null)
    return next()
  }
  const user = await session.getUser(c.req.raw)
  if (!user)
    return c.json(
      {
        error: {
          code: 'sign_in_required',
          message: 'Sign in to talk to Lumen.',
        },
      },
      401,
    )
  c.set('user', user)
  await next()
})

app.use('/api/chat', async (c, next) => {
  if (c.req.method === 'POST') {
    try {
      JSON.parse(await c.req.raw.clone().text())
    } catch {
      return c.json(
        {
          error: {
            code: 'invalid_input',
            message: 'Send a JSON body with a messages array.',
          },
        },
        400,
      )
    }
  }
  await next()
})

app.use(
  '/api/*',
  bodyLimit({
    maxSize: 64 * 1024,
    onError: (c) =>
      c.json(
        {
          error: {
            code: 'body_too_large',
            message: 'That conversation is too long. Clear it and try again.',
          },
        },
        413,
      ),
  }),
)

app.openapi(
  createRoute({
    method: 'get',
    path: '/api/health',
    summary: 'Report the locally trained model that is loaded',
    responses: {
      200: {
        description: 'The service is running.',
        content: json(
          z.object({
            ok: z.boolean(),
            model: z.object({
              documents: z.number(),
              terms: z.number(),
              trainedAt: z.string(),
            }),
          }),
        ),
      },
    },
  }),
  (c) => c.json({ ok: true, model: modelInfo }, 200),
)

app.openapi(
  createRoute({
    method: 'get',
    path: '/api/me',
    summary: 'Report the signed-in visitor, for display only',
    responses: {
      200: {
        description: 'The current session, or null when signed out.',
        content: json(
          z.object({
            signInAvailable: z.boolean(),
            user: z
              .object({
                sub: z.string(),
                name: z.string().optional(),
                email: z.string().optional(),
                picture: z.string().optional(),
              })
              .nullable(),
          }),
        ),
      },
    },
  }),
  async (c) => {
    const session = getAuth(c.env)
    const user = session ? await session.getUser(c.req.raw) : null
    return c.json(
      {
        signInAvailable: Boolean(session),
        user: user
          ? {
              sub: user.sub,
              name: user.name,
              email: user.email,
              picture: user.picture,
            }
          : null,
      },
      200,
    )
  },
)

app.openapi(
  createRoute({
    method: 'post',
    path: '/api/chat',
    summary: 'Answer locally with the trained model, streamed as events',
    request: { body: { required: true, content: json(chatSchema) } },
    responses: {
      200: {
        description: 'Server-sent events with reply deltas.',
        content: { 'text/event-stream': { schema: z.string() } },
      },
      400: { description: 'Invalid conversation.', content: json(errorSchema) },
      413: { description: 'Body too large.', content: json(errorSchema) },
      429: { description: 'Too many requests.', content: json(errorSchema) },
      401: { description: 'Sign in required.', content: json(errorSchema) },
    },
  }),
  (c) => {
    const ip = c.req.header('cf-connecting-ip') ?? 'unknown'
    if (limited(ip))
      return c.json(
        {
          error: {
            code: 'rate_limited',
            message: 'You are sending messages too quickly. Try again in a minute.',
          },
        },
        429,
      )

    const { messages } = c.req.valid('json')
    if (messages.at(-1)?.role !== 'user')
      return c.json(
        {
          error: {
            code: 'invalid_input',
            message: 'The last message must be from the user.',
          },
        },
        400,
      )
    const total = messages.reduce((sum, message) => sum + message.content.length, 0)
    if (total > MAX_TOTAL_CHARS)
      return c.json(
        {
          error: {
            code: 'conversation_too_long',
            message: 'This conversation is quite long. Clear it to start fresh.',
          },
        },
        400,
      )

    const answer = reply(messages as ChatMessage[])

    // Stream the locally computed answer so the interface can render it incrementally.
    const chunks: string[] = []
    let cursor = 0
    while (cursor < answer.length) {
      const nextSpace = answer.indexOf(' ', cursor + 24)
      const end = nextSpace === -1 ? answer.length : nextSpace + 1
      chunks.push(answer.slice(cursor, end))
      cursor = end
    }

    const encoder = new TextEncoder()
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks)
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ delta: chunk })}\n\n`),
          )
        controller.enqueue(encoder.encode('data: {"done":true}\n\n'))
        controller.close()
      },
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
        Connection: 'keep-alive',
      },
    })
  },
)

app.doc('/openapi.json', {
  openapi: '3.0.0',
  info: { title: 'Lumen local model API', version: '2.0.0' },
})

app.notFound((c) => {
  if (c.req.path === '/api' || c.req.path.startsWith('/api/'))
    return c.json(
      { error: { code: 'not_found', message: 'Endpoint not found.' } },
      404,
    )
  return c.env.ASSETS.fetch(c.req.raw)
})

app.onError((_error, c) =>
  c.json(
    { error: { code: 'internal_error', message: 'Something went wrong.' } },
    500,
  ),
)

let wrapped: ((
  request: Request,
  env: Bindings,
  ctx: ExecutionContext,
) => Promise<Response>) | undefined

export default {
  async fetch(request: Request, env: Bindings, ctx: ExecutionContext): Promise<Response> {
    const session = getAuth(env)
    if (!session) return await app.fetch(request, env, ctx)
    // Serves /auth/freebuff/* (sign in, callback, sign out, session) and keeps
    // sessions fresh, before the API and the static-asset fallback.
    wrapped ??= session.wrap<[Bindings, ExecutionContext]>(async (req, e, c) =>
      app.fetch(req, e, c),
    )
    return wrapped(request, env, ctx)
  },
} satisfies { fetch: (request: Request, env: Bindings, ctx: ExecutionContext) => Promise<Response> }
