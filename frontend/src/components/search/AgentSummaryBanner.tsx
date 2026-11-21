import type React from 'react';
import { Sparkles, ShieldAlert, CheckCircle2, Clock } from 'lucide-react';
import type { QueryResponse } from '../../types/api';
import { MarkdownViewer } from './MarkdownViewer';

interface AgentSummaryBannerProps {
  response: QueryResponse | null;
}

export const AgentSummaryBanner: React.FC<AgentSummaryBannerProps> = ({ response }) => {
  if (!response || !response.final_answer) {
    return null;
  }

  const isVerified = response.verification_status === 'VERIFIED';
  const confidence = response.confidence?.toUpperCase();

  return (
    <div className="w-full max-w-4xl mx-auto mb-8 bg-white border border-slate-200 rounded-2xl shadow-xs overflow-hidden">
      {/* Banner Header */}
      <div className="bg-gradient-to-r from-emerald-50 via-teal-50 to-slate-50 px-6 py-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 bg-emerald-600 text-white rounded-lg shadow-xs">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight">
              AI Multi-Agent Investigation Synthesis
            </h3>
            <p className="text-xs text-slate-500">
              Corroborated across PostgreSQL records and vector semantic search
            </p>
          </div>
        </div>

        {/* Status Pills */}
        <div className="flex items-center space-x-2">
          {response.verification_status && (
            <span
              className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border ${
                isVerified
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                  : 'bg-amber-50 text-amber-800 border-amber-200'
              }`}
            >
              {isVerified ? (
                <CheckCircle2 className="w-3.5 h-3.5 mr-1 text-emerald-600" />
              ) : (
                <Clock className="w-3.5 h-3.5 mr-1 text-amber-600" />
              )}
              {response.verification_status}
            </span>
          )}

          {confidence && (
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
              Confidence: {confidence}
            </span>
          )}
        </div>
      </div>

      {/* Answer Body with Formatted Markdown Rendering */}
      <div className="p-6 text-sm sm:text-base text-slate-800 leading-relaxed">
        <MarkdownViewer content={response.final_answer} />
      </div>

      {/* Mandatory Legal Caveat */}
      <div className="bg-slate-50 px-6 py-3 border-t border-slate-200 flex items-center space-x-2 text-xs text-slate-600">
        <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
        <span>
          <span className="font-semibold text-slate-700">Notice:</span> Legal effect and
          current enforceability are not independently verified by this system.
        </span>
      </div>
    </div>
  );
};
