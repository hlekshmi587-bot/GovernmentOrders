import { useEffect, useState } from 'react';
import type React from 'react';
import {
  X,
  FileText,
  ExternalLink,
  Download,
  Calendar,
  Building2,
  ShieldAlert,
  Loader2,
  AlertTriangle,
  ArrowDownLeft,
  ArrowUpRight,
} from 'lucide-react';
import type { DocumentDetailResponse, DocumentRelationshipsResponse } from '../../types/api';
import { getDocumentDetail, getDocumentRelationships, getPdfUrl } from '../../api/documents';

interface DocumentDetailModalProps {
  goId: number | null;
  onClose: () => void;
}

export const DocumentDetailModal: React.FC<DocumentDetailModalProps> = ({ goId, onClose }) => {
  const [detail, setDetail] = useState<DocumentDetailResponse | null>(null);
  const [relationships, setRelationships] = useState<DocumentRelationshipsResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (goId === null) return;

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    Promise.all([getDocumentDetail(goId), getDocumentRelationships(goId)])
      .then(([det, rels]) => {
        if (isMounted) {
          setDetail(det);
          setRelationships(rels);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          console.error('Failed to load document details:', err);
          setError('Failed to load complete document records. Please try again.');
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [goId]);

  if (goId === null) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6"
      role="dialog"
      aria-modal="true"
    >
      <div className="relative bg-white rounded-2xl shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden border border-slate-200">
        {/* Header */}
        <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight">
              {detail ? detail.go_number : 'Loading Document Records…'}
            </h2>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500 mt-1">
              <span className="inline-flex items-center">
                <Calendar className="w-3.5 h-3.5 mr-1 text-slate-400" />
                {detail?.issue_date || 'Date unrecorded'}
              </span>
              <span className="inline-flex items-center">
                <Building2 className="w-3.5 h-3.5 mr-1 text-slate-400" />
                {detail?.department || 'Department unrecorded'}
              </span>
              {detail?.source_filename && (
                <span className="inline-flex items-center">
                  <FileText className="w-3.5 h-3.5 mr-1 text-slate-400" />
                  {detail.source_filename}
                </span>
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-200 transition-colors"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm text-slate-800">
          {isLoading && (
            <div className="py-16 text-center text-slate-500 flex flex-col items-center justify-center">
              <Loader2 className="w-8 h-8 animate-spin text-emerald-600 mb-3" />
              <p className="text-sm font-medium">Retrieving document and relationship records…</p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 flex items-center space-x-3">
              <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!isLoading && !error && detail && (
            <>
              {/* PDF Actions Bar */}
              {detail.has_pdf && detail.pdf_url && (
                <div className="flex flex-wrap items-center justify-between gap-3 p-4 bg-emerald-50/60 border border-emerald-200 rounded-xl">
                  <div className="flex items-center space-x-2 text-xs text-emerald-800">
                    <FileText className="w-4 h-4 text-emerald-600" />
                    <span className="font-medium">Original PDF document archived on disk.</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <a
                      href={getPdfUrl(detail.id, false)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center px-3 py-1.5 text-xs font-semibold text-emerald-800 bg-white border border-emerald-300 rounded-lg hover:bg-emerald-50 transition-colors"
                    >
                      <span>View PDF</span>
                      <ExternalLink className="w-3 h-3 ml-1.5" />
                    </a>
                    <a
                      href={getPdfUrl(detail.id, true)}
                      className="inline-flex items-center px-3 py-1.5 text-xs font-semibold text-emerald-800 bg-white border border-emerald-300 rounded-lg hover:bg-emerald-50 transition-colors"
                    >
                      <Download className="w-3 h-3 mr-1.5" />
                      <span>Download</span>
                    </a>
                  </div>
                </div>
              )}

              {/* Extracted Text Preview Section */}
              <div className="space-y-2">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                  Extracted text preview
                </h3>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 max-h-48 overflow-y-auto text-xs sm:text-sm font-mono text-slate-700 whitespace-pre-wrap leading-relaxed">
                  {detail.extracted_text_preview || 'No extracted text preview available.'}
                </div>
              </div>

              {/* Read References Section */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                    "Read" References ({detail.references.length})
                  </h3>
                </div>

                {detail.references.length === 0 ? (
                  <p className="text-xs text-slate-500 italic bg-slate-50 p-3 rounded-lg border border-slate-100">
                    No cited references recorded in indexed preamble.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {detail.references.map((ref, idx) => (
                      <div
                        key={ref.id ?? idx}
                        className="p-3 bg-white border border-slate-200 rounded-lg text-xs space-y-1"
                      >
                        <div className="flex items-center justify-between font-semibold text-slate-900">
                          <span>
                            {ref.ordinal}. {ref.referenced_go_number}
                          </span>
                          <span className="text-slate-500 font-normal">
                            {ref.reference_date ? `Dated: ${ref.reference_date}` : 'Date unrecorded'}
                            {ref.page_number ? ` (Page ${ref.page_number})` : ''}
                          </span>
                        </div>
                        {ref.evidence_text && (
                          <p className="text-slate-600 italic">"{ref.evidence_text}"</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Incoming Relationships Section */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <ArrowDownLeft className="w-4 h-4 text-emerald-600" />
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                    Incoming Relationships ({relationships?.incoming_relationships.length || 0})
                  </h3>
                </div>
                <p className="text-xs text-slate-500">
                  Later Government Orders that reference, amend, supersede, or cancel this order.
                </p>

                {(!relationships?.incoming_relationships ||
                  relationships.incoming_relationships.length === 0) ? (
                  <p className="text-xs text-slate-500 italic bg-slate-50 p-3 rounded-lg border border-slate-100">
                    No incoming relationships recorded in indexed records.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {relationships.incoming_relationships.map((rel) => (
                      <div
                        key={rel.id}
                        className="p-3 bg-white border border-slate-200 rounded-lg text-xs space-y-1.5"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="font-semibold text-slate-900">
                            Acting Order: {rel.source_go_number}
                          </span>
                          <div className="flex items-center space-x-1.5">
                            <span className="px-2 py-0.5 rounded-full font-bold bg-slate-100 text-slate-800 border border-slate-200">
                              {rel.relationship_type}
                            </span>
                            <span
                              className={`px-2 py-0.5 rounded-full font-medium border ${
                                rel.verification_status === 'VERIFIED'
                                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                  : 'bg-amber-50 text-amber-800 border-amber-200'
                              }`}
                            >
                              {rel.verification_status}
                            </span>
                          </div>
                        </div>
                        {rel.matched_phrase && (
                          <p className="text-slate-700">
                            <span className="font-medium text-slate-500">Matched phrase:</span> "
                            {rel.matched_phrase}"
                          </p>
                        )}
                        {rel.evidence_text && (
                          <p className="text-slate-600 italic">"{rel.evidence_text}"</p>
                        )}
                        <div className="text-slate-400 text-[11px] flex items-center space-x-3 pt-1 border-t border-slate-100">
                          <span>Page: {rel.page_number ?? '—'}</span>
                          <span>Extraction Confidence: {rel.confidence ?? '—'}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Outgoing Relationships Section */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <ArrowUpRight className="w-4 h-4 text-blue-600" />
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                    Outgoing Relationships ({relationships?.outgoing_relationships.length || 0})
                  </h3>
                </div>
                <p className="text-xs text-slate-500">
                  Earlier Government Orders cited, amended, or superseded by this order.
                </p>

                {(!relationships?.outgoing_relationships ||
                  relationships.outgoing_relationships.length === 0) ? (
                  <p className="text-xs text-slate-500 italic bg-slate-50 p-3 rounded-lg border border-slate-100">
                    No outgoing relationships recorded in indexed records.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {relationships.outgoing_relationships.map((rel) => (
                      <div
                        key={rel.id}
                        className="p-3 bg-white border border-slate-200 rounded-lg text-xs space-y-1.5"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="font-semibold text-slate-900">
                            Target Order: {rel.target_go_number || 'Unspecified / Statutory Rule'}
                          </span>
                          <div className="flex items-center space-x-1.5">
                            <span className="px-2 py-0.5 rounded-full font-bold bg-slate-100 text-slate-800 border border-slate-200">
                              {rel.relationship_type}
                            </span>
                            <span className="px-2 py-0.5 rounded-full font-medium bg-slate-100 text-slate-700 border border-slate-200">
                              {rel.verification_status}
                            </span>
                          </div>
                        </div>
                        {rel.evidence_text && (
                          <p className="text-slate-600 italic">"{rel.evidence_text}"</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Modal Footer with Mandatory Disclaimer */}
        <div className="bg-slate-50 px-6 py-3 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-600 shrink-0">
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
            <span>Legal effect and current enforceability are not independently verified by this system.</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-100 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
