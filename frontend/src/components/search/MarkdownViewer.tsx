import React from 'react';
import { ShieldAlert, Info, AlertTriangle } from 'lucide-react';

interface MarkdownViewerProps {
  content?: string | null;
  className?: string;
}

/**
 * Normalizes backslash-escaped markdown markers without corrupting
 * legitimate backslashes, such as Windows paths (e.g. C:\Users, D:\GovernmentOrders)
 * or LaTeX math expressions (e.g. $\rightarrow$).
 */
export function normalizeMarkdown(text: string): string {
  if (!text) return '';
  return text
    // Normalize escaped headings at start of line: \## -> ##, \### -> ###
    .replace(/^\\(#{1,6}\s)/gm, '$1')
    // Normalize escaped bold/italic: \**text\** -> **text**, \*text\* -> *text*
    .replace(/\\(\*{1,2})(.*?)\\(\*{1,2})/g, '$1$2$3')
    // Normalize isolated escaped asterisks attached to words: \** -> **
    .replace(/(^|[^\\])\\(\*{1,2})/g, '$1$2')
    // Normalize escaped blockquotes at start of line: \> -> >
    .replace(/^\\(>\s*)/gm, '$1')
    // Normalize escaped inline code: \`code\` -> `code`
    .replace(/\\(`)(.*?)\\(`)/g, '$1$2$3');
}

/**
 * Parses inline formatting: **bold**, `code`, *italic*, and LaTeX arrows ($\rightarrow$).
 * Strictly avoids dangerouslySetInnerHTML; returns pure React elements.
 */
function renderInline(text: string): React.ReactNode[] {
  // Regex splitting on: **bold**, `code`, *italic*, $\rightarrow$
  const tokenRegex = /(\*\*[^*]+?\*\*|`[^`]+?`|\*[^*]+?\*|\$\\rightarrow\$|->)/g;
  const parts = text.split(tokenRegex);

  return parts.map((part, idx) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
      return (
        <strong key={idx} className="font-semibold text-slate-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
      return (
        <code
          key={idx}
          className="px-1.5 py-0.5 text-xs font-mono font-medium bg-slate-100 text-slate-800 rounded border border-slate-200"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length >= 2) {
      return (
        <em key={idx} className="italic text-slate-600">
          {part.slice(1, -1)}
        </em>
      );
    }
    if (part === '$\\rightarrow$' || part === '->') {
      return (
        <span key={idx} className="text-slate-400 font-bold mx-1">
          →
        </span>
      );
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

export const MarkdownViewer: React.FC<MarkdownViewerProps> = ({ content, className = '' }) => {
  if (!content) return null;

  const normalized = normalizeMarkdown(content);
  const rawLines = normalized.split('\n');

  // Group into blocks: headings, lists, callout/notice blocks, paragraphs
  type Block =
    | { type: 'h2'; text: string }
    | { type: 'h3'; text: string }
    | { type: 'h4'; text: string }
    | { type: 'callout'; lines: string[]; variant: 'important' | 'warning' | 'info' }
    | { type: 'list'; items: { text: string; indent: number }[] }
    | { type: 'paragraph'; text: string };

  const blocks: Block[] = [];
  let currentList: { text: string; indent: number }[] | null = null;
  let currentCallout: { lines: string[]; variant: 'important' | 'warning' | 'info' } | null = null;

  const flushList = () => {
    if (currentList && currentList.length > 0) {
      blocks.push({ type: 'list', items: currentList });
      currentList = null;
    }
  };

  const flushCallout = () => {
    if (currentCallout && currentCallout.lines.length > 0) {
      blocks.push({
        type: 'callout',
        lines: currentCallout.lines,
        variant: currentCallout.variant,
      });
      currentCallout = null;
    }
  };

  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i];
    const trimmed = line.trim();

    // Check for blockquote / callout
    if (trimmed.startsWith('>')) {
      flushList();
      const contentLine = trimmed.replace(/^>\s*/, '');
      const lower = contentLine.toLowerCase();

      let variant: 'important' | 'warning' | 'info' = 'info';
      if (lower.includes('[!important]') || lower.includes('official notice')) {
        variant = 'important';
      } else if (lower.includes('[!warning]') || lower.includes('warning') || lower.includes('incomplete')) {
        variant = 'warning';
      }

      // Filter out raw "[!IMPORTANT]" or "[!WARNING]" markers from display text
      const cleanLine = contentLine
        .replace(/\[!IMPORTANT\]/i, '')
        .replace(/\[!WARNING\]/i, '')
        .trim();

      if (!currentCallout) {
        currentCallout = { lines: cleanLine ? [cleanLine] : [], variant };
      } else {
        if (cleanLine) currentCallout.lines.push(cleanLine);
      }
      continue;
    } else {
      flushCallout();
    }

    // Check for empty line
    if (!trimmed) {
      flushList();
      continue;
    }

    // Check for headings
    if (trimmed.startsWith('#### ')) {
      flushList();
      blocks.push({ type: 'h4', text: trimmed.slice(5) });
      continue;
    }
    if (trimmed.startsWith('### ')) {
      flushList();
      blocks.push({ type: 'h3', text: trimmed.slice(4) });
      continue;
    }
    if (trimmed.startsWith('## ')) {
      flushList();
      blocks.push({ type: 'h2', text: trimmed.slice(3) });
      continue;
    }

    // Check for list items
    const listMatch = line.match(/^(\s*)-\s+(.*)$/);
    if (listMatch) {
      const indent = listMatch[1].length;
      const text = listMatch[2];
      if (!currentList) currentList = [];
      currentList.push({ text, indent });
      continue;
    } else {
      flushList();
    }

    // Fallback: standard paragraph
    blocks.push({ type: 'paragraph', text: trimmed });
  }

  flushList();
  flushCallout();

  return (
    <div className={`space-y-3 ${className}`}>
      {blocks.map((block, bIdx) => {
        switch (block.type) {
          case 'h2':
            return (
              <h2 key={bIdx} className="text-base sm:text-lg font-bold text-slate-900 tracking-tight mt-4 mb-2">
                {renderInline(block.text)}
              </h2>
            );
          case 'h3':
            return (
              <h3 key={bIdx} className="text-sm sm:text-base font-semibold text-slate-900 mt-3 mb-1.5">
                {renderInline(block.text)}
              </h3>
            );
          case 'h4':
            return (
              <h4 key={bIdx} className="text-xs sm:text-sm font-semibold text-slate-800 mt-2 mb-1">
                {renderInline(block.text)}
              </h4>
            );
          case 'list':
            return (
              <ul key={bIdx} className="space-y-1.5 my-2">
                {block.items.map((item, iIdx) => (
                  <li
                    key={iIdx}
                    className={`text-sm text-slate-700 leading-relaxed flex items-start ${
                      item.indent > 0 ? 'ml-6 text-slate-600' : 'ml-1'
                    }`}
                  >
                    <span className="text-emerald-500 mr-2 shrink-0 select-none">•</span>
                    <span>{renderInline(item.text)}</span>
                  </li>
                ))}
              </ul>
            );
          case 'callout':
            return (
              <div
                key={bIdx}
                className={`my-3 p-4 rounded-xl border flex items-start space-x-3 text-xs sm:text-sm leading-relaxed ${
                  block.variant === 'warning'
                    ? 'bg-amber-50/80 border-amber-200 text-amber-900'
                    : block.variant === 'important'
                    ? 'bg-emerald-50/70 border-emerald-200 text-emerald-950'
                    : 'bg-slate-50 border-slate-200 text-slate-800'
                }`}
              >
                {block.variant === 'warning' ? (
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                ) : block.variant === 'important' ? (
                  <ShieldAlert className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                ) : (
                  <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                )}
                <div className="space-y-1">
                  {block.lines.map((l, lIdx) => (
                    <p key={lIdx}>{renderInline(l)}</p>
                  ))}
                </div>
              </div>
            );
          case 'paragraph':
            return (
              <p key={bIdx} className="text-sm text-slate-700 leading-relaxed my-1.5">
                {renderInline(block.text)}
              </p>
            );
          default:
            return null;
        }
      })}
    </div>
  );
};
