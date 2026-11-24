import './style.css'

type Role = 'user' | 'assistant'
type Message = { role: Role; content: string }

const MAX_INPUT = 2000
const MAX_CONTEXT_MESSAGES = 20

const thread = document.querySelector<HTMLElement>('#thread')!
const conversation = document.querySelector<HTMLElement>('#conversation')!
const intro = document.querySelector<HTMLElement>('#intro')!
const suggestions = document.querySelector<HTMLElement>('#suggestions')!
const form = document.querySelector<HTMLFormElement>('#composer')!
const input = document.querySelector<HTMLTextAreaElement>('#input')!
const sendButton = document.querySelector<HTMLButtonElement>('#send')!
const clearButton = document.querySelector<HTMLButtonElement>('#clear')!
const themeButton = document.querySelector<HTMLButtonElement>('#theme')!
const statusEl = document.querySelector<HTMLElement>('#status')!
const accountEl = document.querySelector<HTMLElement>('#account')!
const signinCard = document.querySelector<HTMLElement>('#signin')!
const composerMeta = document.querySelector<HTMLElement>('#composer-meta')!
const statusText = document.querySelector<HTMLElement>('#status-text')!
const count = document.querySelector<HTMLElement>('#count')!

let history: Message[] = []
let controller: AbortController | null = null
let busy = false

/* Account */

type Viewer = { sub: string; name?: string; email?: string; picture?: string }
let viewer: Viewer | null = null
let signedIn = true

function renderAccount() {
  accountEl.replaceChildren()
  if (!signedIn) return
  if (!viewer) {
    const signIn = document.createElement('a')
    signIn.className = 'account-signin'
    signIn.href = '/auth/freebuff/signin?returnTo=/'
    signIn.textContent = 'Sign in'
    accountEl.append(signIn)
    return
  }
  const chip = document.createElement('div')
  chip.className = 'account-chip'
  const label = viewer.name ?? viewer.email ?? 'Signed in'
  if (viewer.picture) {
    const avatar = document.createElement('img')
    avatar.src = viewer.picture
    avatar.alt = ''
    avatar.width = 22
    avatar.height = 22
    avatar.referrerPolicy = 'no-referrer'
    chip.append(avatar)
  }
  const name = document.createElement('span')
  name.className = 'account-name'
  name.textContent = label
  chip.append(name)
  accountEl.append(chip)

  const form = document.createElement('form')
  form.method = 'post'
  form.action = '/auth/freebuff/signout'
  const out = document.createElement('button')
  out.type = 'submit'
  out.className = 'text-button'
  out.textContent = 'Sign out'
  form.append(out)
  accountEl.append(form)
}

function applyAuth(next: { signedIn: boolean; user: Viewer | null }) {
  signedIn = next.signedIn
  viewer = next.user
  document.querySelector<HTMLFormElement>('#composer')!.hidden = !signedIn
  composerMeta.hidden = !signedIn
  signinCard.hidden = signedIn
  renderAccount()
  syncControls()
  if (signedIn) input.focus()
}

async function loadSession() {
  try {
    const response = await fetch('/api/me', { headers: { Accept: 'application/json' } })
    if (response.status !== 200) {
      applyAuth({ signedIn: false, user: null })
      return
    }
    const body = (await response.json()) as {
      signInAvailable?: boolean
      user?: Viewer | null
    }
    applyAuth({
      signedIn: body.signInAvailable === false || Boolean(body.user),
      user: body.user ?? null,
    })
  } catch {
    // Keep the page usable offline; the server still checks the session.
    applyAuth({ signedIn: true, user: null })
  }
}

/* Theme */

const savedTheme = localStorage.getItem('lumen-theme')
const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches
document.documentElement.dataset.theme =
  savedTheme ?? (systemDark ? 'dark' : 'light')

themeButton.addEventListener('click', () => {
  const next =
    document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'
  document.documentElement.dataset.theme = next
  localStorage.setItem('lumen-theme', next)
})

/* Markdown: small, safe renderer for assistant replies */

const escapeHtml = (value: string) =>
  value.replace(
    /[&<>"']/g,
    (char) =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      })[char]!,
  )

function inline(raw: string) {
  const codes: string[] = []
  let text = escapeHtml(raw)
  text = text.replace(/`([^`]+)`/g, (_match, code: string) => {
    codes.push(`<code>${code}</code>`)
    return `\uE000${codes.length - 1}\uE001`
  })
  text = text.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_match, label: string, url: string) =>
      `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`,
  )
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  text = text.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
  return text.replace(/\uE000(\d+)\uE001/g, (_m, i: string) => codes[+i] ?? '')
}

function renderMarkdown(source: string) {
  const blocks: string[] = []
  const text = source.replace(
    /```([\w+-]*)\n?([\s\S]*?)(?:```|$)/g,
    (_match, lang: string, code: string) => {
      blocks.push(
        `<pre><code${lang ? ` class="language-${escapeHtml(lang)}"` : ''}>${escapeHtml(
          code.replace(/\n$/, ''),
        )}</code></pre>`,
      )
      return `\uE000${blocks.length - 1}\uE001\n\n`
    },
  )

  const output: string[] = []
  let paragraph: string[] = []
  let list: { ordered: boolean; items: string[] } | null = null
  let quote: string[] = []

  const flushParagraph = () => {
    if (!paragraph.length) return
    output.push(`<p>${paragraph.map(inline).join('<br />')}</p>`)
    paragraph = []
  }
  const flushList = () => {
    if (!list) return
    const tag = list.ordered ? 'ol' : 'ul'
    output.push(
      `<${tag}>${list.items.map((item) => `<li>${inline(item)}</li>`).join('')}</${tag}>`,
    )
    list = null
  }
  const flushQuote = () => {
    if (!quote.length) return
    output.push(`<blockquote>${quote.map(inline).join('<br />')}</blockquote>`)
    quote = []
  }
  const flushAll = () => {
    flushParagraph()
    flushList()
    flushQuote()
  }

  for (const line of text.split('\n')) {
    const trimmed = line.trim()

    if (!trimmed) {
      flushAll()
      continue
    }
    const placeholder = trimmed.match(/^\uE000(\d+)\uE001$/)
    if (placeholder) {
      flushAll()
      output.push(blocks[+placeholder[1]] ?? '')
      continue
    }
    if (/^(-{3,}|\*{3,})$/.test(trimmed)) {
      flushAll()
      output.push('<hr />')
      continue
    }
    const heading = trimmed.match(/^#{1,4}\s+(.*)$/)
    if (heading) {
      flushAll()
      output.push(`<h3>${inline(heading[1])}</h3>`)
      continue
    }
    const quoted = trimmed.match(/^>\s?(.*)$/)
    if (quoted) {
      flushParagraph()
      flushList()
      quote.push(quoted[1])
      continue
    }
    const bullet = trimmed.match(/^[-*+]\s+(.*)$/)
    const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/)
    if (bullet || numbered) {
      flushParagraph()
      flushQuote()
      const ordered = Boolean(numbered)
      if (!list || list.ordered !== ordered) {
        flushList()
        list = { ordered, items: [] }
      }
      list.items.push((bullet ?? numbered)![1])
      continue
    }
    if (list) {
      // Continuation of the current list item.
      list.items[list.items.length - 1] += ` ${trimmed}`
      continue
    }
    if (quote.length) flushQuote()
    paragraph.push(trimmed)
  }
  flushAll()
  return output.join('')
}

/* UI helpers */

function setStatus(state: 'ready' | 'thinking' | 'offline') {
  statusEl.dataset.state = state
  statusText.textContent =
    state === 'thinking' ? 'Thinking…' : state === 'offline' ? 'Offline' : 'Ready'
}

function syncControls() {
  const hasText = input.value.trim().length > 0
  if (busy) {
    sendButton.dataset.mode = 'stop'
    sendButton.disabled = false
    sendButton.setAttribute('aria-label', 'Stop response')
  } else {
    sendButton.dataset.mode = 'send'
    sendButton.disabled = !hasText
    sendButton.setAttribute('aria-label', 'Send message')
  }
  count.hidden = input.value.length <= 1600
  count.textContent = `${input.value.length} / ${MAX_INPUT}`
  clearButton.hidden = history.length === 0 && !busy
}

function resizeInput() {
  input.style.height = 'auto'
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`
}

function nearBottom() {
  return (
    conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight <
    120
  )
}

function scrollToBottom() {
  conversation.scrollTop = conversation.scrollHeight
}

function showIntro(show: boolean) {
  intro.hidden = !show
}

function addUserMessage(content: string) {
  showIntro(false)
  const message = document.createElement('div')
  message.className = 'message user'
  const bubble = document.createElement('div')
  bubble.className = 'bubble'
  bubble.textContent = content
  message.append(bubble)
  thread.append(message)
}

function addAssistantShell() {
  showIntro(false)
  const message = document.createElement('div')
  message.className = 'message assistant'
  const bubble = document.createElement('div')
  bubble.className = 'bubble'
  const typing = document.createElement('div')
  typing.className = 'typing'
  typing.setAttribute('aria-label', 'Assistant is typing')
  typing.innerHTML = '<span></span><span></span><span></span>'
  bubble.append(typing)
  const tools = document.createElement('div')
  tools.className = 'message-tools'
  tools.hidden = true
  const copy = document.createElement('button')
  copy.className = 'tool'
  copy.type = 'button'
  copy.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2.5" stroke="currentColor" stroke-width="1.6"/><path d="M15 6.5V5.5A1.5 1.5 0 0 0 13.5 4h-8A1.5 1.5 0 0 0 4 5.5v8A1.5 1.5 0 0 0 5.5 15h1" stroke="currentColor" stroke-width="1.6"/></svg><span>Copy</span>'
  copy.addEventListener('click', async () => {
    const text = message.dataset.text ?? ''
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      copy.querySelector('span')!.textContent = 'Copied'
    } catch {
      copy.querySelector('span')!.textContent = 'Press Ctrl+C'
    }
    window.setTimeout(() => {
      copy.querySelector('span')!.textContent = 'Copy'
    }, 1600)
  })
  tools.append(copy)
  message.append(bubble, tools)
  thread.append(message)
  return { message, bubble, tools }
}

function addErrorBox(text: string, canRetry: boolean) {
  const box = document.createElement('div')
  box.className = 'error-box'
  const label = document.createElement('strong')
  label.textContent = 'Something went wrong.'
  const detail = document.createElement('span')
  detail.textContent = text
  box.append(label, detail)
  if (canRetry) {
    const retry = document.createElement('button')
    retry.className = 'retry'
    retry.type = 'button'
    retry.textContent = 'Retry'
    retry.addEventListener('click', () => {
      box.remove()
      void request()
    })
    box.append(retry)
  }
  thread.append(box)
}

/* Conversation */

async function request() {
  if (busy) return
  const payload = history.slice(-MAX_CONTEXT_MESSAGES)
  if (!payload.length || payload.at(-1)?.role !== 'user') return

  busy = true
  controller = new AbortController()
  setStatus('thinking')
  syncControls()

  const { message, bubble, tools } = addAssistantShell()
  let text = ''
  let frame = 0
  let failed = false

  const paint = () => {
    frame = 0
    bubble.innerHTML = `${renderMarkdown(text)}<span class="cursor" aria-hidden="true"></span>`
    bubble.classList.add('cursor')
  }

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: payload }),
      signal: controller.signal,
    })

    if (response.status === 401) applyAuth({ signedIn: false, user: null })
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as {
        error?: { message?: string }
      } | null
      const messageText =
        body?.error?.message ?? 'The assistant could not be reached.'
      throw new Error(messageText)
    }
    if (!response.body) throw new Error('The reply did not arrive.')

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let streamError: string | null = null

    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''
      for (const chunk of frames) {
        const line = chunk.trim()
        if (!line.startsWith('data:')) continue
        let event: {
          delta?: string
          done?: boolean
          error?: { message?: string }
        }
        try {
          event = JSON.parse(line.slice(5).trim())
        } catch {
          continue
        }
        if (event.delta) {
          text += event.delta
          if (!frame) frame = requestAnimationFrame(paint)
        } else if (event.error) {
          streamError = event.error.message ?? 'The reply was cut short.'
        }
      }
    }
    if (frame) cancelAnimationFrame(frame)
    if (streamError) throw new Error(streamError)
    if (!text.trim()) throw new Error('The assistant returned an empty reply.')
  } catch (error) {
    const aborted = error instanceof DOMException && error.name === 'AbortError'
    if (frame) cancelAnimationFrame(frame)
    if (aborted) {
      if (text.trim()) {
        history.push({ role: 'assistant', content: text })
        bubble.innerHTML = renderMarkdown(text)
        message.dataset.text = text
        tools.hidden = false
      } else {
        message.remove()
      }
    } else {
      failed = true
      message.remove()
      addErrorBox(
        error instanceof Error ? error.message : 'Please try again.',
        true,
      )
    }
  } finally {
    busy = false
    controller = null
    setStatus('ready')
    syncControls()
    if (failed) scrollToBottom()
  }

  if (failed || !text.trim()) return

  history.push({ role: 'assistant', content: text })
  bubble.classList.remove('cursor')
  bubble.innerHTML = renderMarkdown(text)
  message.dataset.text = text
  tools.hidden = false
  syncControls()
  scrollToBottom()
}

function send(content: string) {
  const text = content.trim().slice(0, MAX_INPUT)
  if (!text || busy) return
  history.push({ role: 'user', content: text })
  addUserMessage(text)
  input.value = ''
  resizeInput()
  syncControls()
  scrollToBottom()
  void request()
}

form.addEventListener('submit', (event) => {
  event.preventDefault()
  if (busy) {
    controller?.abort()
    return
  }
  send(input.value)
})

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    if (!busy) send(input.value)
  }
})

input.addEventListener('input', () => {
  resizeInput()
  syncControls()
})

suggestions.addEventListener('click', (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>(
    '.suggestion',
  )
  if (button) send(button.dataset.prompt ?? '')
})

clearButton.addEventListener('click', () => {
  controller?.abort()
  history = []
  thread.replaceChildren()
  showIntro(true)
  syncControls()
  input.focus()
})

async function checkHealth() {
  try {
    const response = await fetch('/api/health')
    if (!response.ok) throw new Error('unhealthy')
  } catch {
    setStatus('offline')
  }
}

resizeInput()
syncControls()
void loadSession()
void checkHealth()
