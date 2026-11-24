const STOPWORDS = new Set(
  `a an the and or but if then than of at by for with about into over under from to in on out up down
  is are was were be been being am do does did doing have has had having
  i you he she it we they me him her them us my your his its our their this that these those
  what which who whom whose when where why how not no nor so too very just also only even
  can could should would will shall may might must s t don now here there as
  get got make made like want need know think please tell give show help`
    .split(/\s+/)
    .filter(Boolean),
)

/** Crude suffix stripping so "plants" and "plant" match. */
export function stem(word: string) {
  if (word.length > 4 && word.endsWith('ies')) return `${word.slice(0, -3)}y`
  // "adaptation" -> "adapt", so it matches "adapt".
  if (word.length > 7 && word.endsWith('ation')) return word.slice(0, -5)
  if (word.length > 4 && word.endsWith('ses')) return word.slice(0, -2)
  if (word.length > 3 && word.endsWith('s') && !word.endsWith('ss'))
    return word.slice(0, -1)
  if (word.length > 5 && word.endsWith('ing')) return word.slice(0, -3)
  if (word.length > 4 && word.endsWith('ed')) return word.slice(0, -2)
  if (word.length > 5 && word.endsWith('ly')) return word.slice(0, -2)
  return word
}

/** Shared by the training script and the runtime so both see the same features. */
export function tokenize(text: string) {
  return text
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^\p{L}\p{N}\s+*/^%=.-]/gu, ' ')
    .split(/\s+/)
    .map((word) => word.replace(/^[.-]+|[.-]+$/g, ''))
    .filter((word) => word.length > 1 && !STOPWORDS.has(word))
    .map(stem)
    .filter((word) => word.length > 1 && !STOPWORDS.has(word))
}
