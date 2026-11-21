import type React from 'react';
import { Search, X } from 'lucide-react';

interface SearchBarProps {
  query: string;
  onChange: (value: string) => void;
  onSearch: (searchQuery?: string) => void;
  isLoading: boolean;
  onCancel?: () => void;
}

const EXAMPLE_QUERIES = [
  'Finance Department',
  'G.O.(P) No. 45/2024/Fin',
  'dearness relief pension',
  'Which later order modified G.O. 114?',
];

export const SearchBar: React.FC<SearchBarProps> = ({
  query,
  onChange,
  onSearch,
  isLoading,
  onCancel,
}) => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isLoading && query.trim()) {
      onSearch();
    }
  };

  const handleChipClick = (example: string) => {
    onChange(example);
    onSearch(example);
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="relative flex items-center">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-400">
            <Search className="w-5 h-5" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => onChange(e.target.value)}
            disabled={isLoading}
            placeholder="Search by topic, question, department name, or exact GO number…"
            className="w-full pl-11 pr-10 py-3.5 text-sm sm:text-base bg-white border border-slate-300 rounded-xl shadow-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 disabled:bg-slate-50 disabled:text-slate-500 transition-colors"
          />
          {query && !isLoading && (
            <button
              type="button"
              onClick={() => onChange('')}
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600"
              aria-label="Clear query"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        <div className="ml-3 flex items-center space-x-2">
          {isLoading ? (
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-3.5 text-sm font-medium text-rose-700 bg-rose-50 border border-rose-200 rounded-xl hover:bg-rose-100 focus:outline-none focus:ring-2 focus:ring-rose-500 transition-colors"
            >
              Cancel
            </button>
          ) : (
            <button
              type="submit"
              disabled={!query.trim()}
              className="px-6 py-3.5 text-sm sm:text-base font-semibold text-white bg-emerald-600 rounded-xl shadow-xs hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
            >
              Search
            </button>
          )}
        </div>
      </form>

      {/* Example Query Chips */}
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span className="font-medium text-slate-600">Examples:</span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => handleChipClick(example)}
            disabled={isLoading}
            className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
};
