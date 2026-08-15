import type { ReactNode } from "react";
import type { Source } from "@/lib/types";

/**
 * Renders the model's answer.
 *
 * Deliberately a small hand-rolled renderer rather than a markdown library:
 * the model emits a narrow subset (paragraphs, bullets, bold, inline code) plus
 * `[n]` citation markers that need to become links into the source list. Output
 * is React elements, never raw HTML, so model text can never inject markup.
 */
export function AnswerText({
  text,
  sources,
}: {
  text: string;
  sources: Source[];
}) {
  const blocks: ReactNode[] = [];
  const lines = text.split("\n");
  let listItems: ReactNode[] = [];
  let paragraph: string[] = [];

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul className="answer-list" key={`ul-${blocks.length}`}>
        {listItems}
      </ul>,
    );
    listItems = [];
  };

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    blocks.push(
      <p key={`p-${blocks.length}`}>{inline(paragraph.join(" "), sources)}</p>,
    );
    paragraph = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (line === "") {
      flushParagraph();
      flushList();
      continue;
    }

    const bullet = line.match(/^[-*•]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      listItems.push(
        <li key={`li-${listItems.length}`}>{inline(bullet[1], sources)}</li>,
      );
      continue;
    }

    const numbered = line.match(/^\d+\.\s+(.*)$/);
    if (numbered) {
      flushParagraph();
      listItems.push(
        <li key={`li-${listItems.length}`}>{inline(numbered[1], sources)}</li>,
      );
      continue;
    }

    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push(
        <h4 className="answer-heading" key={`h-${blocks.length}`}>
          {inline(heading[1], sources)}
        </h4>,
      );
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();

  return <div className="answer">{blocks}</div>;
}

// Matches, in one pass: **bold**, *italic*, `code`, and [1] / [1][2] citations.
const INLINE = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\[\d{1,2}\])/g;

function inline(text: string, sources: Source[]): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null;

    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }

    const citation = part.match(/^\[(\d{1,2})\]$/);
    if (citation) {
      const index = Number(citation[1]);
      const source = sources.find((s) => s.index === index);
      // While streaming, sources aren't known yet — show the marker inert.
      if (!source) return <sup className="citation citation-pending" key={i}>{index}</sup>;
      return (
        <a
          key={i}
          className="citation"
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          title={`${source.title} — ${source.section}`}
        >
          <sup>{index}</sup>
        </a>
      );
    }

    return <span key={i}>{part}</span>;
  });
}
