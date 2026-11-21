import { useState } from 'react';
import {
  Shield,
  Search,
  AlertTriangle,
  Loader2,
  FileQuestion,
  Info,
} from 'lucide-react';
import type { QueryResponse } from './types/api';
import { submitQuery } from './api/documents';
import { ApiError } from './api/client';
import { SearchBar } from './components/search/SearchBar';
import { AgentSummaryBanner } from './components/search/AgentSummaryBanner';
import { ResultCard } from './components/search/ResultCard';
import { DocumentDetailModal } from './components/search/DocumentDetailModal';

export default function App() {
  const [query, setQuery] = useState<string>('');
  const [activeQuery, setActiveQuery] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [selectedGoId, setSelectedGoId] = useState<number | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const handleSearch = async (overrideQuery?: string) => {
    const q = (overrideQuery ?? query).trim();
    if (!q || isLoading) return;

    if (abortController) {
      abortController.abort();
    }

    const controller = new AbortController();
    setAbortController(controller);
    setIsLoading(true);
    setError(null);
    setActiveQuery(q);

    try {
      const res = await submitQuery({ query: q }, controller.signal);
      setResponse(res);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError(
          'Search request cancelled in browser. Note: cancelling in the browser frees your interface but does not stop ongoing server-side workflow execution.',
        );
      } else {
        const message =
          err instanceof ApiError
            ? err.detail
            : err instanceof Error
            ? err.message
            : 'Workflow execution encountered an unexpected error.';
        setError(message);
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
    }
  };

  const handleCancel = () => {
    if (abortController) {
      abortController.abort();
      setIsLoading(false);
      setAbortController(null);
      setError(
        'Search request cancelled in browser. Note: cancelling in the browser frees your interface but does not stop ongoing server-side workflow execution.',
      );
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 shadow-xs sticky top-0 z-40">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-emerald-600 text-white p-2 rounded-xl shadow-xs">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900">
                Kerala Government Orders
              </h1>
              <p className="text-xs text-slate-500">
                Search, Relationship Investigation & Evidence-Based Discovery
              </p>
            </div>
          </div>

          <div className="hidden sm:flex items-center space-x-2 text-xs font-semibold text-emerald-800 bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-200">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Agentic Search Workflow</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8">
        {/* Search Hero Section */}
        <div className="text-center max-w-3xl mx-auto mb-8">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Government Orders Investigation & Discovery
          </h2>
          <p className="mt-2 text-sm sm:text-base text-slate-600">
            Enter a topic, natural-language inquiry, department name, or canonical Government Order number to explore indexed orders and verified documentary relationships.
          </p>
        </div>

        {/* Prominent Search Bar */}
        <div className="mb-10">
          <SearchBar
            query={query}
            onChange={setQuery}
            onSearch={handleSearch}
            isLoading={isLoading}
            onCancel={handleCancel}
          />
        </div>

        {/* Loading State Banner */}
        {isLoading && (
          <div className="w-full max-w-4xl mx-auto mb-8 p-6 bg-white border border-emerald-200 rounded-2xl shadow-xs text-center flex flex-col items-center justify-center space-y-3">
            <Loader2 className="w-8 h-8 text-emerald-600 animate-spin" />
            <p className="text-sm sm:text-base font-semibold text-slate-800">
              Analysing your request and searching indexed Government Order records. This may take a few minutes.
            </p>
            <p className="text-xs text-slate-500 max-w-md">
              The LangGraph Supervisor Agent is classifying your inquiry, coordinating search retrieval, and cross-examining documentary evidence.
            </p>
            <button
              type="button"
              onClick={handleCancel}
              className="mt-2 px-4 py-1.5 text-xs font-semibold text-rose-700 bg-rose-50 border border-rose-200 rounded-lg hover:bg-rose-100 transition-colors"
            >
              Cancel search
            </button>
          </div>
        )}

        {/* Error Alert Banner */}
        {error && !isLoading && (
          <div className="w-full max-w-4xl mx-auto mb-8 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 flex items-start space-x-3 text-sm">
            <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold block mb-0.5">Search Notification</span>
              <p>{error}</p>
            </div>
            {activeQuery && (
              <button
                type="button"
                onClick={() => handleSearch(activeQuery)}
                className="px-3 py-1 font-semibold text-xs bg-white border border-rose-300 rounded-lg text-rose-800 hover:bg-rose-100 shrink-0 transition-colors"
              >
                Retry
              </button>
            )}
          </div>
        )}

        {/* AI Multi-Agent Investigation Synthesis Banner */}
        {!isLoading && response && (
          <AgentSummaryBanner response={response} />
        )}

        {/* Search Results List */}
        {!isLoading && response && (
          <div className="w-full max-w-4xl mx-auto space-y-4">
            <div className="flex items-center justify-between px-1">
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                Indexed Government Orders ({response.search_results.length})
              </h3>
              {activeQuery && (
                <span className="text-xs text-slate-500">
                  Results for <span className="font-semibold text-slate-700">"{activeQuery}"</span>
                </span>
              )}
            </div>

            {/* Empty State */}
            {response.search_results.length === 0 ? (
              <div className="text-center py-16 bg-white border border-slate-200 rounded-2xl p-8 shadow-xs">
                <FileQuestion className="w-12 h-12 text-slate-300 mx-auto mb-3" />
                <h4 className="text-base font-bold text-slate-800 mb-1">
                  No matching Government Orders found
                </h4>
                <p className="text-sm text-slate-500 max-w-md mx-auto">
                  No orders matching this inquiry were identified in indexed records. Try searching by department name (e.g., "Finance Department"), specific topics ("pension revision"), or canonical GO numbers.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {response.search_results.map((result, idx) => (
                  <ResultCard
                    key={result.go_id ?? `${result.go_number}-${idx}`}
                    result={result}
                    onViewDetails={(goId) => setSelectedGoId(goId)}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Initial Welcome Helper (when no query has been submitted yet) */}
        {!isLoading && !response && !error && (
          <div className="w-full max-w-4xl mx-auto bg-white border border-slate-200 rounded-2xl p-8 text-center shadow-xs">
            <div className="w-12 h-12 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-4">
              <Search className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-2">
              Explore Kerala Government Orders with Multi-Agent AI
            </h3>
            <p className="text-sm text-slate-600 max-w-xl mx-auto leading-relaxed">
              Every citizen search passes through the LangGraph agentic workflow. The Supervisor Agent interprets your query, coordinates PostgreSQL search and vector retrieval, and corroborates documentary references.
            </p>
            <div className="mt-6 flex flex-wrap justify-center gap-3 text-xs text-slate-500">
              <span className="inline-flex items-center px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200">
                <Info className="w-3.5 h-3.5 mr-1.5 text-slate-400" />
                Keyword & Topic Search
              </span>
              <span className="inline-flex items-center px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200">
                <Info className="w-3.5 h-3.5 mr-1.5 text-slate-400" />
                Exact GO Number Lookups
              </span>
              <span className="inline-flex items-center px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200">
                <Info className="w-3.5 h-3.5 mr-1.5 text-slate-400" />
                Documentary Relationship Graphs
              </span>
            </div>
          </div>
        )}
      </main>

      {/* Document Detail Modal */}
      {selectedGoId !== null && (
        <DocumentDetailModal
          goId={selectedGoId}
          onClose={() => setSelectedGoId(null)}
        />
      )}

      {/* Persistent Legal Disclaimer Footer */}
      <footer className="bg-white border-t border-slate-200 px-6 py-4 mt-auto">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-xs text-slate-500 leading-relaxed">
            <span className="font-semibold text-slate-700">Notice:</span> Legal effect and
            current enforceability are not independently verified by this system.
          </p>
        </div>
      </footer>
    </div>
  );
}
