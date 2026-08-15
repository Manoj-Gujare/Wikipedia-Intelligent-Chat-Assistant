"use client";

const SUGGESTIONS = [
  "What is the event horizon of a black hole?",
  "How does photosynthesis work?",
  "Who was Alan Turing and why does he matter?",
  "Tell me about Mercury",
];

export function EmptyState({
  sharedChunks,
  personalArticles,
  onPick,
}: {
  sharedChunks: number;
  personalArticles: number;
  onPick: (suggestion: string) => void;
}) {
  return (
    <div className="empty">
      <h2>Ask anything covered by Wikipedia</h2>
      <p>
        Every answer is built from indexed article text and cites the exact
        sections it used.
      </p>
      <div className="suggestions">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            className="suggestion"
            onClick={() => onPick(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
      <p className="index-note">
        {sharedChunks.toLocaleString()} shared passages
        {personalArticles > 0
          ? ` · ${personalArticles.toLocaleString()} of your own articles`
          : " · add your own topics from the sidebar"}
      </p>
    </div>
  );
}
