import type { Language } from "./types";

/** Language editions the UI offers. The backend accepts any MediaWiki code. */
export const LANGUAGES: Language[] = [
  { code: "en", label: "English" },
  { code: "es", label: "Español" },
  { code: "fr", label: "Français" },
  { code: "de", label: "Deutsch" },
  { code: "hi", label: "हिन्दी" },
  { code: "mr", label: "मराठी" },
  { code: "ja", label: "日本語" },
];
