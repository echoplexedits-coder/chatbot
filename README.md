# Lumen — a K–12 study assistant with a locally trained model

No API keys, no provider accounts, no network calls at runtime. Everything is
fitted inside this repository and runs in the Worker: a curriculum map of 581
topics and 534 formulas across Classes 1–12, a TF-IDF retrieval index over a
230-entry knowledge base (81 deep topics plus 149 general-knowledge facts), and
a softmax intent classifier trained by gradient descent. Lessons, notes,
revision sheets, formula lists, tests and answer checks are all composed
locally, and the browser never talks to anything except this Worker.

```sh
bun install
bun run train     # fits the model and writes src/model/model.json
bun run dev       # http://localhost:5173
bun test
bun run typecheck
bun run build
```

## Education engine

The reference frame is the Indian NCERT/CBSE curriculum, with ICSE, state boards
and international syllabuses acknowledged rather than assumed identical. The
map in `data/curriculum/` covers Classes 1–12 across mathematics, physics,
chemistry, biology, science and EVS, history, geography, civics and political
science, economics, accountancy, business studies, computer science, English,
Hindi, sociology and psychology. Each NCERT chapter band is represented: primary and middle-school
basics, the Class 9–10 science, social science and mathematics course, the
Class 11–12 science stream (including organic chemistry, plant and human
physiology, genetics and biotechnology), the senior humanities and commerce
streams, and the Class 11–12 computer science, English and Hindi syllabuses.
Cross-cutting topics such as climate change, waste management, disaster
management, road safety, financial literacy, the two world wars and human
rights sit alongside the chapters they support.

It understands what kind of help is being asked for:

| Ask | Get |
| --- | --- |
| `teach me integers for class 7` | A lesson: definition, key ideas, formulas, worked example, common mistakes, a question to try |
| `make notes on photosynthesis` | Structured notes with a definition, key points, example and a recall check |
| `revision sheet on the French Revolution` | A compact sheet: definition, remember-list, watch-outs, one-line recall |
| `formulas for class 10 maths` | A formula sheet for that subject and level, with symbols |
| `test me on class 10 quadratic equations` | Five questions; then `answers` shows the key |
| `important questions on trigonometry` | Exam-style practice questions with mark hints |
| `explain the human eye in detail` | The full lesson plus related topics |
| `check my answer on photosynthesis: …` | An honest comparison against the stored key points |

Naming a class changes the depth: Classes 1–2 get the summary and two points,
Classes 3–5 get three, and older classes get the full entry. If a topic is
taught later than the class asked for, the answer says so. A chapter that is not
in the map, or a request that names only a class and subject, gets an honest
prompt for the topic plus the topics that are available at that level.

These are compact teaching summaries, not scanned textbook pages. The engine
does not invent chapter names or claim a question will appear in an exam, and it
will not mark a free-text answer as a teacher would: it compares your wording
with the key points and says what is missing.

## What it can do

- **Teach and revise** any topic in the curriculum map, at the class level you
  name, in the formats in the table above.
- **Answer** questions covered by `data/knowledge.json` (science, technology,
  programming, maths, everyday life) and `data/facts.json` (capitals, history
  dates, unit conversions, measurements, oceans, planets, common everyday
  science). It retrieves the best matching entry and returns it.
- **Speak** in sentences: contractions (`it is` → `it's`) and plainer wording,
  with answers starting on the point rather than behind a stock opener. A
  short lead-in appears only on long explanations, and a closing offer only
  occasionally, when there is genuinely more to read. Lists, formulas and code
  keep their formatting.
- **Basic manners**: answers please, sorry, thank you, you're welcome, good bot
  and similar courtesies politely, and strips lead-in politeness so "sorry to
  bother you, what is DNA?" still gets the real answer.
- **Conversation**: reacts to how you say you are feeling (tired, bored, sad,
  stressed, happy, angry, lonely), long days, being stuck on something,
  learning a topic, opinions, criticism, plain acknowledgements ("ok", "cool"),
  names, "are you there", and requests to chat. "I don't understand" or
  "explain it simpler" re-explains the previous topic in one shorter sentence
  instead of repeating the same paragraph, and "why?" reaches for the deeper
  note behind the previous answer. It answers honestly about having no moods,
  opinions or memory between sessions and asks a question back so the exchange
  can continue, instead of dropping straight to the knowledge list. How are
  you, your name, your age, where you run, a joke, goodbye and whether it can
  speak are covered the same honest way.
- **Conversational lead-ins**: "can you tell me about X", "I want to know about
  X" and "explain X to me" are treated as lookups for X, so the framing words
  do not drown the topic.
- **Follow-ups**: "tell me more", "why?", "explain further" expand the previous
  topic instead of starting over.
- **Arithmetic** locally: `12 * 7`, `(3+4)*5`, `sqrt(144)`, `15% of 80`.
- **Summarise** pasted text extractively (top sentences by term frequency).
- **Ideas** for a topic, from constraint-based templates.
- **Admit ignorance**: anything outside the knowledge base gets an honest
  "I do not have anything reliable on that" plus the list of real topics.

It is not a large language model. It cannot write open-ended prose, translate,
write code, or answer general world questions beyond its fixed entries, and it
does not learn from your conversation. It also cannot grade an essay, and the
curriculum entries are concise revision material rather than a replacement for
the textbook. That is the honest trade for zero running cost and full privacy.

## Training

`data/knowledge.json` holds the deep entries and `data/facts.json` the short
general-knowledge ones; both share a shape: a title, keywords, example
questions, optional `casual` phrasings, an answer, an optional deeper `more`
paragraph and an optional `related` entry. `data/casual.json` maps a topic id to
further casual and misspelled phrasings (`waht is photosynthsis`), so the curated
examples stay readable. `data/intents.json` holds labelled examples for
greetings, thanks, maths, summaries, ideas and follow-ups.

`scripts/train.ts` then:

1. tokenises and stems everything with `src/shared/text.ts` (shared with the
   runtime, so features match exactly),
2. fits IDF weights and L2-normalised TF-IDF vectors for every entry,
3. builds a classifier vocabulary from the corpus plus the intent examples,
4. trains a 7-class softmax regression by SGD with balanced class weights and
   L2 regularisation,
5. writes `src/model/model.json` (~278 KB),
6. evaluates on held-out phrasings: every fourth phrasing per topic is removed
   before fitting, so the reported numbers measure generalisation rather than
   memorisation.

Retrieval scores are ranked by cosine similarity weighted by how much of the
question the entry covers, so one shared word cannot win on its own, and an
entry is only answered when it shares enough content terms and clears a score
threshold. Some facts that must not be guessed are structured instead: capitals
come from a table, and an unknown country gets an honest refusal.

Current run: **230 documents from 1,628 phrasings, 1,535 retrieval features, 7
intents, ~278 KB**. Held-out retrieval top-1 **82.0%** (201/245), 78.0% of
held-out questions answered end to end, 11/12 off-topic probes refused by
retrieval alone (the remaining one, an unknown capital, is refused by the
capital table), held-out intent accuracy **94.4%**, in-sample **97.1%**. The
larger, more overlapping corpus is harder than the earlier 172-entry one, and
some misses land on a closely related entry rather than nothing. Rerun
`bun run train` after editing any data file; the app picks up the new model on
the next build or dev reload. `MODEL_DIAGNOSTICS=1 bun run train` also prints
per-question scores.

Adding knowledge is just another JSON entry: add example questions a user would
actually type, and the retrieval index learns to match them.

## Endpoint

`POST /api/chat` accepts `{ messages: [{ role, content }] }` (max 24 messages,
2,000 characters each), validates it, applies a limit of 20 requests per 10
minutes per client, then streams the locally computed answer as server-sent
events (`{delta}` … `{done}`). `GET /api/health` reports the loaded model
(document count, features, training time).

Everything runs in the same Worker that serves the page, so there is no
database and no secrets.

## Sign in

The deployed site uses **Sign in with Freebuff** (`@freebuff/auth`). The Worker
serves `/auth/freebuff/{signin,callback,signout,session}` before the API and the
asset fallback, and `/api/chat` refuses requests without a session
(`401 sign_in_required`) — the hidden composer in the interface is only a
courtesy, the server is what enforces it. `GET /api/me` returns the visitor for
display, and `GET /api/health` stays public so the page can report readiness
before signing in.

There is no client secret: `FREEBUFF_CLIENT_ID`, `FREEBUFF_ISSUER` and
`APP_ORIGIN` are public configuration, injected by `sites_deploy` for the
deployed site. Running `bun run dev` without them serves the app with accounts
disabled, which is what the tests and local use rely on. Sign out is a `POST`
to `/auth/freebuff/signout`.

## Files

| Path | Purpose |
| --- | --- |
| `data/knowledge.json` | The deep topic entries |
| `data/facts.json` | General-knowledge facts |
| `data/casual.json` | Casual and misspelled phrasings per topic |
| `data/intents.json` | Labelled intent examples |
| `data/curriculum/*.json` | Classes 1–12 curriculum map: topics, classes, formulas (four class-band files plus fourteen per-stream extension files) |
| `scripts/train.ts` | Training script (retrieval index + classifier) |
| `src/model/model.json` | Trained weights, generated |
| `src/worker/model.ts` | Inference: retrieval, classifier, maths, summaries, ideas |
| `src/worker/education.ts` | Teaching modes over the curriculum map |
| `src/worker/index.ts` | Validation, sign-in gate, rate limit, streaming endpoint |
| `src/chat.ts`, `src/style.css`, `index.html` | The interface |

## Publish

Build, then deploy with `sites_deploy` using `deploy.json`.
