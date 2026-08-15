import type { DisambiguationOption } from "@/lib/types";

/**
 * Shown when the question maps to a Wikipedia disambiguation page. Picking an
 * option sends it back as the next turn, so the conversation resolves the
 * ambiguity instead of guessing.
 */
export function DisambiguationList({
  options,
  onPick,
  disabled,
}: {
  options: DisambiguationOption[];
  onPick: (title: string) => void;
  disabled: boolean;
}) {
  if (options.length === 0) return null;

  return (
    <section className="disambiguation">
      <div className="disambiguation-options">
        {options.map((option) => (
          <div key={option.url} className="disambiguation-option">
            <button
              className="disambiguation-pick"
              onClick={() => onPick(option.title)}
              disabled={disabled}
            >
              {option.title}
            </button>
            <a
              className="disambiguation-link"
              href={option.url}
              target="_blank"
              rel="noopener noreferrer"
              title={`Open ${option.title} on Wikipedia`}
            >
              ↗
            </a>
          </div>
        ))}
      </div>
    </section>
  );
}
