import { useEffect, useState } from 'react';
import { FileText, ExternalLink, Calendar, Building2 } from 'lucide-react';
import type { RelationshipItem, SearchResultItem } from '../../types/api';
import { getDocumentRelationships, getPdfUrl } from '../../api/documents';

interface ResultCardProps {
  result: SearchResultItem;
  onViewDetails: (goId: number) => void;
}

export interface StatusBadgeInfo {
  label: string;
  tone: 'blue' | 'amber' | 'rose' | 'neutral';
  tooltip: string;
}

/**
 * Derives recorded status strictly from incoming relationships using approved conservative rules.
 */
export function computeRecordedStatus(
  incoming: RelationshipItem[] | null,
  isLoading: boolean,
  isError: boolean,
): StatusBadgeInfo {
  if (isLoading) {
    return {
      label: 'Checking recorded status…',
      tone: 'neutral',
      tooltip: 'Querying incoming relationships in indexed records.',
    };
  }

  if (isError) {
    return {
      label: 'Status requires further verification',
      tone: 'neutral',
      tooltip: 'Relationship lookup could not be completed for this order.',
    };
  }

  if (!incoming || incoming.length === 0) {
    return {
      label: 'Status not established in indexed records',
      tone: 'neutral',
      tooltip: 'No confirmed incoming amendment, supersession, or cancellation records found.',
    };
  }

  // Inspect ONLY incoming records
  const incomingOnly = incoming.filter((r) => r.direction === 'INCOMING');
  if (incomingOnly.length === 0) {
    return {
      label: 'Status not established in indexed records',
      tone: 'neutral',
      tooltip: 'No incoming relationship records found.',
    };
  }

  // If ANY incoming relationship has a non-VERIFIED status (including REJECTED, PENDING, UNRECORDED)
  // or an unsupported relationship type, display "Status requires further verification".
  const hasUnverifiedOrUnsupported = incomingOnly.some((r) => {
    const type = r.relationship_type.toUpperCase();
    const isSupportedType = ['AMENDS', 'MODIFIES', 'SUPERSEDES', 'CANCELS', 'REVOKES'].includes(type);
    return r.verification_status !== 'VERIFIED' || !isSupportedType;
  });

  if (hasUnverifiedOrUnsupported) {
    return {
      label: 'Status requires further verification',
      tone: 'neutral',
      tooltip: 'One or more incoming signals lack verified confirmation, have rejected evidence, or have an unsupported type.',
    };
  }

  // At this point, ALL incoming relationships are VERIFIED and belong to supported types!
  // Normalize into three distinct categories:
  // - Amendment: AMENDS, MODIFIES
  // - Cancellation: CANCELS, REVOKES
  // - Supersession: SUPERSEDES
  const verifiedTypes = incomingOnly.map((r) => r.relationship_type.toUpperCase());
  const isCancelsOrRevokes = (t: string) => ['CANCELS', 'REVOKES'].includes(t);
  const isSupersedes = (t: string) => t === 'SUPERSEDES';
  const isAmendsOrModifies = (t: string) => ['AMENDS', 'MODIFIES'].includes(t);

  const hasCancellation = verifiedTypes.some(isCancelsOrRevokes);
  const hasSupersession = verifiedTypes.some(isSupersedes);
  const hasAmendment = verifiedTypes.some(isAmendsOrModifies);

  // Count how many distinct normalized categories are present
  const categoriesCount =
    (hasCancellation ? 1 : 0) + (hasSupersession ? 1 : 0) + (hasAmendment ? 1 : 0);

  // If normalized categories conflict (e.g. Amendment + Cancellation), return "Status requires further verification"
  if (categoriesCount > 1) {
    return {
      label: 'Status requires further verification',
      tone: 'neutral',
      tooltip: 'Conflicting verified incoming categories exist whose chronological sequence requires verification.',
    };
  }

  // Exclusively Cancellation / Revocation
  if (hasCancellation) {
    return {
      label: 'Cancellation recorded — verification required',
      tone: 'rose',
      tooltip: 'A later indexed order contains a verified cancellation or revocation signal.',
    };
  }

  // Exclusively Supersession
  if (hasSupersession) {
    return {
      label: 'Superseded in indexed records',
      tone: 'amber',
      tooltip: 'A later indexed order contains a verified supersession signal.',
    };
  }

  // Exclusively Amendment / Modification
  if (hasAmendment) {
    return {
      label: 'Amended in indexed records',
      tone: 'blue',
      tooltip: 'A later indexed order contains a verified amendment or modification signal.',
    };
  }

  return {
    label: 'Status not established in indexed records',
    tone: 'neutral',
    tooltip: 'Evidence insufficient to establish a confirmed status.',
  };
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, onViewDetails }) => {
  const [incomingRels, setIncomingRels] = useState<RelationshipItem[] | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState<boolean>(result.go_id !== null);
  const [isStatusError, setIsStatusError] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;

    if (result.go_id === null) {
      setIsLoadingStatus(false);
      return;
    }

    setIsLoadingStatus(true);
    setIsStatusError(false);

    getDocumentRelationships(result.go_id)
      .then((res) => {
        if (isMounted) {
          setIncomingRels(res.incoming_relationships || []);
          setIsLoadingStatus(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          console.warn(`Could not load relationships for GO ID ${result.go_id}:`, err);
          setIsStatusError(true);
          setIsLoadingStatus(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [result.go_id]);

  const statusInfo = computeRecordedStatus(incomingRels, isLoadingStatus, isStatusError);

  const getToneClasses = (tone: StatusBadgeInfo['tone']) => {
    switch (tone) {
      case 'rose':
        return 'bg-rose-50 text-rose-800 border-rose-200';
      case 'amber':
        return 'bg-amber-50 text-amber-800 border-amber-200';
      case 'blue':
        return 'bg-blue-50 text-blue-800 border-blue-200';
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs hover:border-slate-300 transition-colors">
      {/* Header Row: GO Number & Status Badge */}
      <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
        <div>
          <button
            type="button"
            onClick={() => result.go_id !== null && onViewDetails(result.go_id)}
            disabled={result.go_id === null}
            className="text-base sm:text-lg font-bold text-slate-900 hover:text-emerald-700 text-left transition-colors disabled:cursor-default disabled:hover:text-slate-900"
          >
            {result.go_number}
          </button>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500 mt-1">
            <span className="inline-flex items-center">
              <Calendar className="w-3.5 h-3.5 mr-1 text-slate-400" />
              {result.issue_date || 'Date unrecorded'}
            </span>
            <span className="inline-flex items-center">
              <Building2 className="w-3.5 h-3.5 mr-1 text-slate-400" />
              {result.department || 'Department unrecorded'}
            </span>
          </div>
        </div>

        {/* Status Badge */}
        <span
          title={statusInfo.tooltip}
          className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${getToneClasses(
            statusInfo.tone,
          )}`}
        >
          {statusInfo.label}
        </span>
      </div>

      {/* Extracted Text Snippet */}
      <div className="text-sm text-slate-700 mb-4 bg-slate-50 p-3 rounded-lg border border-slate-100">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">
          Extracted excerpt:
        </span>
        <p className="line-clamp-3 italic text-slate-600">
          "{result.text_snippet || 'No excerpt preview available.'}"
        </p>
      </div>

      {/* Footer Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100 text-xs">
        <div className="flex items-center space-x-2">
          {result.go_id !== null ? (
            <button
              type="button"
              onClick={() => onViewDetails(result.go_id!)}
              className="px-3 py-1.5 font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors"
            >
              View Document Details
            </button>
          ) : (
            <span className="text-slate-400 italic">Details not indexed</span>
          )}
        </div>

        <div className="flex items-center space-x-3">
          {result.has_pdf && result.go_id !== null ? (
            <a
              href={getPdfUrl(result.go_id, false)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center font-medium text-slate-600 hover:text-emerald-700 transition-colors"
            >
              <FileText className="w-4 h-4 mr-1 text-emerald-600" />
              <span>View PDF</span>
              <ExternalLink className="w-3 h-3 ml-1 text-slate-400" />
            </a>
          ) : (
            <span className="text-slate-400 flex items-center">
              <FileText className="w-3.5 h-3.5 mr-1 text-slate-300" />
              PDF not on disk
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
