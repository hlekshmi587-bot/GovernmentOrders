"""
Response Agent Implementation for Kerala Government Orders System.

Responsible for synthesizing strictly grounded, evidence-backed final responses.
Categorizes findings into distinct sections:
  1. Verified Textual Findings (only claims/relationships backed by VerificationAgent)
  2. Unverified Relationship Candidates (candidate mentions only)
  3. Database Discrepancies
  4. Missing Documents in the Currently Indexed Database
  5. Incomplete Investigation / Qualified Findings Notice

Enforces structural invariants:
  - legal_effect_verified is structurally Literal[False] = False
  - Mandatory disclaimer on automated textual analysis vs official gazette status
  - Never fabricates page numbers, operative sections, or source orders
"""

import re
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResponseAgentInput(BaseModel):
    """
    Validated input contract for ResponseAgent.
    Enforces structural invariants and separates verified findings from candidates.
    """
    model_config = ConfigDict(extra="forbid")

    user_query: str
    intent: Optional[str] = None
    focal_go: Optional[str] = None
    verification_status: Optional[str] = None
    verification_confidence_score: Optional[float] = None  # None if verification not performed
    confidence: Optional[str] = None                      # Categorical retrieval confidence (e.g. "HIGH", "MEDIUM", "LOW")
    evidence_sufficient: bool = False
    
    # STRUCTURAL INVARIANT: Must strictly be False
    legal_effect_verified: Literal[False] = False

    verified_findings: List[Dict[str, Any]] = Field(default_factory=list)
    partially_verified_supported: List[Dict[str, Any]] = Field(default_factory=list)
    partially_verified_unsupported: List[Dict[str, Any]] = Field(default_factory=list)
    unverified_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    discrepancies: List[Dict[str, Any]] = Field(default_factory=list)
    missing_indexed_orders: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    order_recorded_statuses: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    document_analysis: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    is_incomplete_investigation: bool = False
    incomplete_reason: Optional[str] = None

    @field_validator("legal_effect_verified")
    @classmethod
    def reject_true_legal_effect(cls, v: Any) -> Literal[False]:
        if v is not False:
            raise ValueError(
                "legal_effect_verified MUST remain False: automated textual analysis "
                "does not verify legal enforceability or official gazette publication."
            )
        return False


class ResponseAgent:
    """
    Specialized agent synthesizing structured, evidence-grounded responses.
    """

    MANDATORY_DISCLAIMER = (
        "> [!IMPORTANT]\n"
        "> **Official Notice:** This automated report is limited strictly to textual analysis "
        "of documents available in the indexed database. It does not certify official gazette "
        "publication, judicial standing, or legal enforceability. Legal effect remains unverified."
    )

    @classmethod
    def compute_order_recorded_status(
        cls,
        relationships: List[Any],
        lookup_success: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """
        Derives the canonical recorded status for an order strictly from incoming relationships
        using normalized categories:
          - Amendment category: AMENDS, MODIFIES -> "Amended in indexed records"
          - Cancellation category: CANCELS, REVOKES -> "Cancellation recorded — verification required"
          - Supersession category: SUPERSEDES -> "Superseded in indexed records"
          - Lookup failure -> "Status requires further verification"
          - Zero records (lookup_success=True) -> "Status not established in indexed records"
          - Conflicting normalized categories -> "Status requires further verification"
          - Any REJECTED, PENDING, or UNRECORDED -> "Status requires further verification"

        Returns:
            Tuple[str, Optional[str]]: (recorded_status_string, related_go_number_or_none)
        """
        if not lookup_success:
            return "Status requires further verification", None

        if not relationships:
            return "Status not established in indexed records", None

        # Filter strictly incoming relationships
        incoming_only = []
        for r in relationships:
            direction = (
                getattr(r, "direction", None)
                or (r.get("direction") if isinstance(r, dict) else "")
            )
            if str(direction).upper() == "INCOMING":
                incoming_only.append(r)

        if not incoming_only:
            return "Status not established in indexed records", None

        # Check if ANY relationship is not VERIFIED or has unsupported type
        supported_types = {"AMENDS", "MODIFIES", "SUPERSEDES", "CANCELS", "REVOKES"}
        for r in incoming_only:
            ver_status = (
                getattr(r, "verification_status", None)
                or (r.get("verification_status") if isinstance(r, dict) else "")
            )
            rel_type = (
                getattr(r, "relationship_type", None)
                or (r.get("relationship_type") if isinstance(r, dict) else "")
            )
            rel_type_upper = str(rel_type).upper()
            if ver_status != "VERIFIED" or rel_type_upper not in supported_types:
                return "Status requires further verification", None

        # Normalize into categories
        has_amendment = False
        has_cancellation = False
        has_supersession = False
        related_go = None

        for r in incoming_only:
            rel_type_upper = str(
                getattr(r, "relationship_type", None)
                or (r.get("relationship_type") if isinstance(r, dict) else "")
            ).upper()
            src_go = (
                getattr(r, "source_go_number", None)
                or (r.get("source_go_number") if isinstance(r, dict) else None)
            )
            if src_go and not related_go:
                related_go = str(src_go)

            if rel_type_upper in ("AMENDS", "MODIFIES"):
                has_amendment = True
            elif rel_type_upper in ("CANCELS", "REVOKES"):
                has_cancellation = True
            elif rel_type_upper == "SUPERSEDES":
                has_supersession = True

        categories_count = (
            (1 if has_cancellation else 0)
            + (1 if has_supersession else 0)
            + (1 if has_amendment else 0)
        )

        if categories_count > 1:
            return "Status requires further verification", None

        if has_cancellation:
            return "Cancellation recorded — verification required", related_go
        if has_supersession:
            return "Superseded in indexed records", related_go
        if has_amendment:
            return "Amended in indexed records", related_go

        return "Status not established in indexed records", None

    @classmethod
    def select_relevant_orders(
        cls,
        query: str,
        search_results: List[Dict[str, Any]],
        max_orders: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Identifies the top 1-2 genuinely relevant Government Orders based strictly on
        document text and subject information.

        Negative invariant:
          Department names, GO numbers, and matched_sources (bm25, dense_vector)
          must not independently establish topical relevance without textual evidence.
        """
        if not search_results:
            return []

        # Common stop words
        stopwords = {
            "a", "an", "the", "in", "on", "of", "for", "to", "and", "or", "is",
            "are", "was", "were", "by", "with", "at", "from", "as", "about",
            "order", "orders", "government", "kerala", "regarding", "dated",
            "what", "when", "which", "who", "whom", "how", "why", "can", "could",
            "show", "find", "get", "give", "tell", "list", "me", "all", "any"
        }

        # Tokenize query terms
        query_clean = re.sub(r"[^\w\s]", " ", query.lower())
        tokens = [t for t in query_clean.split() if t and t not in stopwords and len(t) >= 2]
        if not tokens:
            return search_results[:max_orders]

        # Generic stemming rules (plural 's'/'es', past 'ed', gerund 'ing', agent 'er'/'ers')
        def get_stem_variants(token: str) -> List[str]:
            # Maintain sorted order with full token and natural inflections first, prefix last
            variants = [token]
            if token.endswith("ers") and len(token) > 5:
                variants.extend([token[:-1], token[:-3]])
            elif token.endswith("er") and len(token) > 4:
                variants.extend([token + "s", token[:-2]])
            elif token.endswith("ies") and len(token) > 4:
                variants.append(token[:-3] + "y")
            elif token.endswith("ing") and len(token) > 5:
                variants.extend([token[:-3], token[:-3] + "e"])
            elif token.endswith("ed") and len(token) > 4:
                variants.extend([token[:-1], token[:-2]])
            elif token.endswith("es") and len(token) > 4:
                variants.append(token[:-2])
            elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                variants.append(token[:-1])
            elif len(token) >= 4:
                variants.extend([token + "s", token + "es", token + "er", token + "ers"])

            # Prefix variation for words >= 5 chars
            if len(token) >= 5:
                variants.append(token[:5])
            elif len(token) >= 4:
                variants.append(token[:4])

            # Deduplicate preserving order
            seen = set()
            ordered = []
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    ordered.append(v)
            return ordered

        query_clusters = [get_stem_variants(tok) for tok in tokens]

        scored_candidates = []
        for cand in search_results:
            subject = str(cand.get("subject") or cand.get("title") or "").lower()
            snippet = str(cand.get("text_snippet") or cand.get("snippet") or cand.get("text") or "").lower()
            doc_text = f"{subject} {snippet}"

            # Document text must provide the substantive evidence
            cluster_matches = 0
            matching_terms = []
            for cluster in query_clusters:
                matched_in_text = False
                for w in cluster:
                    pattern = rf"\b{re.escape(w)}" if len(w) >= 4 else rf"\b{re.escape(w)}\b"
                    if re.search(pattern, doc_text):
                        matched_in_text = True
                        matching_terms.append(w)
                        break
                if matched_in_text:
                    cluster_matches += 1

            # Calculate textual score
            # Priority: subject match + snippet match
            subject_hits = sum(1 for cluster in query_clusters if any(re.search(rf"\b{re.escape(w)}", subject) for w in cluster))
            snippet_hits = sum(1 for cluster in query_clusters if any(re.search(rf"\b{re.escape(w)}", snippet) for w in cluster))

            score = (cluster_matches * 10.0) + (subject_hits * 5.0) + (snippet_hits * 3.0)

            # Preserve RRF / relevance as fine tie-breaker only when textual evidence exists
            base_score = float(cand.get("relevance_score") or cand.get("rrf_score") or 0.0)
            final_score = score + min(base_score, 1.0)

            scored_candidates.append({
                "candidate": cand,
                "textual_score": score,
                "cluster_matches": cluster_matches,
                "matching_terms": matching_terms,
                "final_score": final_score,
            })

        # Sort by final score descending
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # Select candidates that have substantive textual evidence (at least 1 cluster match)
        selected = []
        for sc in scored_candidates:
            if sc["cluster_matches"] > 0 or not selected:
                cand_copy = dict(sc["candidate"])
                cand_copy["_matched_terms"] = sc["matching_terms"]
                cand_copy["_textual_score"] = sc["textual_score"]
                selected.append(cand_copy)
                if len(selected) >= max_orders:
                    break

        return selected if selected else search_results[:max_orders]

    @staticmethod
    def _are_go_matching(go1: Optional[str], go2: Optional[str]) -> bool:
        """Check structural equivalence between two GO numbers."""
        if not go1 or not go2:
            return False
        c1 = re.sub(r"\s+", "", str(go1).strip().upper())
        c2 = re.sub(r"\s+", "", str(go2).strip().upper())
        p = re.compile(r"^(?:G\.O\.?(?:\([A-Za-z]+\))?(?:NO\.?)?)?(\d+)/(\d{4})(?:/([A-Za-z0-9&.-]+))?$", re.IGNORECASE)
        m1 = p.match(c1)
        m2 = p.match(c2)
        if not m1 or not m2:
            return c1 == c2
        return (m1.group(1) == m2.group(1)) and (m1.group(2) == m2.group(2)) and ((m1.group(3) or "") == (m2.group(3) or ""))

    def _extract_relationship_source_target(self, item: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract source and target Government Order numbers from any relationship representation
        (dict or Pydantic/dataclass object), supporting all canonical and legacy field names.
        """
        if not isinstance(item, dict):
            src = (
                getattr(item, "source_go_number", None)
                or getattr(item, "source_order_number", None)
                or getattr(item, "source_go", None)
                or getattr(item, "cited_go_number", None)
                or getattr(item, "cited_go", None)
            )
            tgt = (
                getattr(item, "target_go_number", None)
                or getattr(item, "target_order_number", None)
                or getattr(item, "target_go", None)
                or getattr(item, "referenced_go_number", None)
                or getattr(item, "referenced_order_number", None)
            )
            if not tgt and getattr(item, "referenced_go_numbers", None):
                refs = item.referenced_go_numbers
                if isinstance(refs, (list, tuple)) and refs:
                    tgt = refs[0]
            return (str(src).strip() if src else None), (str(tgt).strip() if tgt else None)

        src = (
            item.get("source_go_number")
            or item.get("source_order_number")
            or item.get("source_go")
            or item.get("cited_go_number")
            or item.get("cited_go")
        )
        tgt = (
            item.get("target_go_number")
            or item.get("target_order_number")
            or item.get("target_go")
            or item.get("referenced_go_number")
            or item.get("referenced_order_number")
        )
        if not tgt and item.get("referenced_go_numbers"):
            refs = item.get("referenced_go_numbers")
            if isinstance(refs, (list, tuple)) and refs:
                tgt = refs[0]

        return (str(src).strip() if src else None), (str(tgt).strip() if tgt else None)

    def generate_response(self, input_data: ResponseAgentInput) -> str:
        """
        Synthesize markdown response strictly according to verified findings and status.
        """
        lines: List[str] = []
        lines.append(f"## Government Orders Analysis Report\n")
        lines.append(f"**Query:** \"{input_data.user_query}\"\n")

        # Status Banner
        if input_data.verification_status == "VERIFIED":
            score_str = f" (Confidence Score: {input_data.verification_confidence_score:.2f})" if input_data.verification_confidence_score is not None else ""
            lines.append(f"**Verification Status:** `VERIFIED`{score_str}\n")
        elif input_data.verification_status == "PARTIALLY_VERIFIED":
            score_str = f" (Confidence Score: {input_data.verification_confidence_score:.2f})" if input_data.verification_confidence_score is not None else ""
            lines.append(f"**Verification Status:** `PARTIALLY_VERIFIED (INCOMPLETE CORROBORATION)`{score_str}\n")
        elif input_data.verification_status == "INSUFFICIENT_EVIDENCE":
            lines.append("**Verification Status:** `INSUFFICIENT_EVIDENCE (INCONCLUSIVE)`\n")
        elif input_data.verification_status == "CONFLICTING_EVIDENCE":
            lines.append("**Verification Status:** `CONFLICTING_EVIDENCE (REQUIRES HUMAN REVIEW)`\n")
        elif input_data.verification_status == "NOT_FOUND":
            lines.append("**Status:** `NOT FOUND IN INDEXED DATABASE`\n")
        else:
            if input_data.confidence:
                lines.append(f"**Retrieval Precision:** `{input_data.confidence}` (Verification not performed)\n")

        # Incomplete Investigation Alert
        if input_data.is_incomplete_investigation:
            reason = input_data.incomplete_reason or "Maximum retries or generation limits reached before conclusive evidence could be gathered."
            lines.append(
                f"> [!WARNING]\n"
                f"> **Incomplete Investigation:** {reason} "
                f"The findings below represent partial retrieval only.\n"
            )

        # Government Order Details (when document_analysis has valid order records)
        valid_docs = [
            doc for doc in input_data.document_analysis
            if isinstance(doc, dict) and doc.get("status") != "NOT_FOUND" and (doc.get("go_number") or doc.get("order_data") or doc.get("data"))
        ]

        is_incoming = (
            (input_data.intent and "INCOMING" in str(input_data.intent).upper())
            or any(k in input_data.user_query.lower() for k in ["incoming", "which later", "later order", "subsequent order", "orders that modify", "orders that reference"])
        )

        user_focal_go = input_data.focal_go
        if not user_focal_go and is_incoming and input_data.user_query:
            m = re.search(
                r"G\.?\s*O\.?\s*(?:\([A-Za-z]+\))?\s*(?:No\.?)?\s*\d+(?:/\d{4}/[A-Za-z&.-]+)?|"
                r"സ\s*\.?\s*ഉ\s*\.?\s*(?:\([^)]*\))?\s*(?:നം|നമ്പർ)?\s*\.?\s*\d+(?:/\d{4}/[\w&.-]+)?|"
                r"\b\d+/\d{4}/[A-Za-z&.-]+",
                input_data.user_query,
                re.IGNORECASE,
            )
            if m:
                user_focal_go = m.group(0).rstrip(".?, ")
        if not user_focal_go and valid_docs:
            user_focal_go = valid_docs[0].get("go_number") or (valid_docs[0].get("order_data", {}).get("go_number") if isinstance(valid_docs[0].get("order_data"), dict) else None)

        unverified_candidates = list(input_data.unverified_candidates)
        discrepancies = list(input_data.discrepancies)
        verified_findings = list(input_data.verified_findings)
        partially_verified_supported = list(input_data.partially_verified_supported)

        if is_incoming and user_focal_go:
            filtered_vf = []
            for item in verified_findings:
                src, tgt = self._extract_relationship_source_target(item)
                if src and self._are_go_matching(src, user_focal_go):
                    continue
                if not tgt or not self._are_go_matching(tgt, user_focal_go):
                    continue
                filtered_vf.append(item)
            verified_findings = filtered_vf

            filtered_pvs = []
            for item in partially_verified_supported:
                src, tgt = self._extract_relationship_source_target(item)
                if src and self._are_go_matching(src, user_focal_go):
                    continue
                if not tgt or not self._are_go_matching(tgt, user_focal_go):
                    continue
                filtered_pvs.append(item)
            partially_verified_supported = filtered_pvs

            filtered_cands = []
            for cand in unverified_candidates:
                src, tgt = self._extract_relationship_source_target(cand)
                if src and self._are_go_matching(src, user_focal_go):
                    continue
                if not tgt or not self._are_go_matching(tgt, user_focal_go):
                    continue
                filtered_cands.append(cand)
            unverified_candidates = filtered_cands

            filtered_discs = []
            for disc in discrepancies:
                src, tgt = self._extract_relationship_source_target(disc)
                if src and self._are_go_matching(src, user_focal_go):
                    continue
                if not tgt or not self._are_go_matching(tgt, user_focal_go):
                    continue
                filtered_discs.append(disc)
            discrepancies = filtered_discs

        if valid_docs:
            heading = "### 1. Indexed Database Records\n" if not input_data.verification_status and not input_data.search_results else "### Indexed Database Records (Government Order Details)\n"
            lines.append(heading)
            for doc in valid_docs:
                data = doc.get("order_data") or doc.get("data") or doc
                go_num = data.get("go_number", "Unknown")
                dept = data.get("department", "Not specified")
                date = data.get("issue_date", "Not specified")
                preview = data.get("text_preview", "")
                lines.append(f"- **Order:** `{go_num}` | **Department:** {dept} | **Date:** {date}")
                if preview:
                    lines.append(f"  - *Preview:* \"{preview[:300].strip()}...\"")
            lines.append("")

        has_relationships = bool(verified_findings or partially_verified_supported or unverified_candidates)
        is_relational = bool(
            (input_data.intent and "RELATIONSHIP" in str(input_data.intent).upper())
            or is_incoming
            or any(k in input_data.user_query.lower() for k in ["incoming", "relationship", "subsequent", "who modified"])
        )

        # 1. No Relationships Notice (When relational query returns zero verified/supported/candidate relationships)
        if is_relational and not has_relationships:
            if is_incoming or any(k in input_data.user_query.lower() for k in ["incoming", "later", "subsequent", "who modified"]):
                lines.append("### 1. Incoming Relationships\n")
                lines.append(
                    "No incoming relationships (subsequent Government Orders modifying, superseding, "
                    "or referencing this order) were found in the indexed database.\n"
                )
            else:
                lines.append("### 1. Relationships\n")
                lines.append("No relationship records were found in the indexed database for this query.\n")

            if input_data.verification_status == "PARTIALLY_VERIFIED" and input_data.partially_verified_unsupported:
                lines.append("#### Unsupported / Unresolved Claims\n")
                for item in input_data.partially_verified_unsupported:
                    claim_text = item.get("claim_text") or item.get("reasoning") or str(item)
                    lines.append(f"- **Claim:** {claim_text}")
                    if item.get("reasoning"):
                        lines.append(f"  - *Note:* {item['reasoning']}")
                lines.append("")

        # 1. Verified Textual Findings (VERIFIED only, with verified findings)
        elif input_data.verification_status == "VERIFIED" and verified_findings:
            lines.append("### 1. Verified Textual Findings\n")
            seen_verified: set = set()
            for item in verified_findings:
                src, tgt = self._extract_relationship_source_target(item)
                rtype = str(item.get("relationship_type") or item.get("candidate_relationship_type") or "").strip().upper()
                sig = (src or "", tgt or "", rtype)
                if sig != ("", "", ""):
                    if sig in seen_verified:
                        continue
                    seen_verified.add(sig)
                lines.extend(self._format_verified_item(item))

        # 1. Partially Verified Findings (PARTIALLY_VERIFIED)
        elif input_data.verification_status == "PARTIALLY_VERIFIED":
            lines.append("### 1. Partially Verified Textual Findings\n")
            if partially_verified_supported:
                lines.append("#### A. Supported Textual Findings\n")
                seen_supported: set = set()
                for item in partially_verified_supported:
                    src, tgt = self._extract_relationship_source_target(item)
                    rtype = str(item.get("relationship_type") or item.get("candidate_relationship_type") or "").strip().upper()
                    sig = (src or "", tgt or "", rtype)
                    if sig != ("", "", ""):
                        if sig in seen_supported:
                            continue
                        seen_supported.add(sig)
                    lines.extend(self._format_verified_item(item))
            if input_data.partially_verified_unsupported:
                lines.append("#### B. Unsupported / Unresolved Claims\n")
                for item in input_data.partially_verified_unsupported:
                    claim_text = item.get("claim_text") or item.get("reasoning") or str(item)
                    lines.append(f"- **Claim:** {claim_text}")
                    if item.get("reasoning"):
                        lines.append(f"  - *Note:* {item['reasoning']}")
                lines.append("")

        # 1. Direct Search Retrieval (When VerificationAgent was not run and no document records exist)
        elif not input_data.verification_status and input_data.search_results and not valid_docs:
            count = len(input_data.search_results)
            depts = sorted(list({str(r.get("department")) for r in input_data.search_results if r.get("department")}))
            dept_str = f" across {len(depts)} department(s) ({', '.join(depts[:3])}{'...' if len(depts) > 3 else ''})" if depts else ""
            lines.append("### Retrieved Search Results\n")
            lines.append(
                f"Retrieved {count} Government Order(s) from the indexed database{dept_str} matching the search inquiry.\n"
            )

            # Prioritize top 1-2 genuinely relevant Government Orders
            top_orders = self.select_relevant_orders(
                query=input_data.user_query,
                search_results=input_data.search_results,
                max_orders=2,
            )

            for idx, res in enumerate(top_orders):
                order_id = res.get("id") or res.get("go_id")
                go_num = res.get("go_number", "Unknown")
                dept = res.get("department", "Not specified")
                date = res.get("issue_date", "Not specified")
                subject = res.get("subject") or res.get("title") or "Not specified"
                snippet = res.get("text_snippet") or ""

                # Compute or retrieve recorded status
                status_info = (
                    input_data.order_recorded_statuses.get(order_id)
                    if order_id is not None
                    else None
                )
                if status_info:
                    rec_status = status_info.get("status")
                    related_go = status_info.get("related_go")
                else:
                    rec_status, related_go = "Status not established in indexed records", None

                # Generate grounded relevance explanation
                matched_terms = res.get("_matched_terms", [])
                if matched_terms:
                    terms_str = ", ".join(f'"{t}"' for t in matched_terms[:3])
                    rel_explanation = f"Matches inquiry terms ({terms_str}) in document text."
                elif snippet:
                    rel_explanation = f"Operative text relates to inquiry: \"{snippet[:160].strip()}...\""
                elif subject != "Not specified":
                    rel_explanation = f"Subject relates to inquiry: \"{subject.strip()}\""
                else:
                    rel_explanation = "Retrieved by indexed administrative search matching inquiry keywords."

                heading = "#### Relevant Government Order" if idx == 0 else "#### Related Significant Order"
                lines.append(f"{heading}")
                lines.append(f"- **Government Order:** `{go_num}` ({date})")
                lines.append(f"- **Department:** {dept}")
                lines.append(f"- **Subject:** {subject}")
                lines.append(f"- **Recorded Status:** {rec_status}")
                if related_go:
                    lines.append(f"- **Related Order:** `{related_go}`")
                lines.append(f"- **Relevance:** {rel_explanation}\n")

            lines.append("*Additional matching Government Orders are listed in the search results below.*\n")

        # 2. Unverified Relationship Candidates
        if unverified_candidates:
            lines.append("### 2. Unverified Relationship Candidates (Candidate Mentions Only)\n")
            lines.append("*The following relationships were extracted from text patterns but have NOT been verified against authoritative database records:*\n")
            for cand in unverified_candidates:
                src = cand.get("source_go_number") or cand.get("source_order_number") or "Unknown"
                tgt = cand.get("target_order_number") or cand.get("target_go_number") or (cand.get("referenced_go_numbers", ["Unknown"])[0] if cand.get("referenced_go_numbers") else "Unknown")
                rel_type = cand.get("relationship_type", "REFERENCES")
                phrase = cand.get("matched_phrase") or cand.get("evidence_text") or ""
                lines.append(f"- **Source Order:** `{src}` $\\rightarrow$ **Action:** `{rel_type}` $\\rightarrow$ **Referenced:** `{tgt}`")
                if phrase:
                    lines.append(f"  - *Extracted Phrase:* \"{phrase.strip()}\"")
            lines.append("")

        # 3. Database Discrepancies
        if discrepancies:
            lines.append("### 3. Database Discrepancies\n")
            for disc in discrepancies:
                desc = disc.get("description") or disc.get("notes") or str(disc)
                ref_go = disc.get("referenced_go_number") or disc.get("target_go_number")
                prefix = f"**Order `{ref_go}`:** " if ref_go else ""
                lines.append(f"- {prefix}{desc}")
            lines.append("")

        # 4. Missing Documents in Indexed Database
        if input_data.missing_indexed_orders or input_data.verification_status == "NOT_FOUND":
            lines.append("### 4. Documents Missing From Currently Indexed Database\n")
            orders = list(input_data.missing_indexed_orders)
            if not orders and input_data.verification_status == "NOT_FOUND":
                orders = [input_data.user_query]
            for order in orders:
                lines.append(
                    f"- Government Order `{order}` was not found in the currently indexed database. "
                    "This indicates that the document has not been ingested into the local repository; "
                    "it does not establish that the order was never officially issued or published."
                )
            lines.append("")

        # Mandatory Disclaimer
        lines.append(self.MANDATORY_DISCLAIMER)
        return "\n".join(lines)

    def _format_verified_item(self, item: Dict[str, Any]) -> List[str]:
        """
        Format a single verified item without fabricating missing citations.
        """
        out = []
        src, tgt = self._extract_relationship_source_target(item)
        rel_type = item.get("relationship_type") or item.get("candidate_relationship_type") or item.get("database_relationship_type")
        claim_text = item.get("claim_text")
        evidence_text = item.get("evidence_text")
        directness = item.get("evidence_directness")
        page_num = item.get("page_number")
        page_verified = item.get("page_number_verified", False)

        if src and tgt and rel_type:
            out.append(f"- **Verified Relationship:** `{src}` $\\rightarrow$ `{rel_type}` $\\rightarrow$ `{tgt}`")
        elif claim_text:
            out.append(f"- **Verified Claim:** {claim_text}")
        else:
            out.append(f"- **Verified Item:** {src or 'Document'}")

        # Citation details (strictly non-fabricated)
        citations = []
        if directness and str(directness) not in ("EvidenceDirectness.NONE", "NONE", "UNCLASSIFIED"):
            dir_str = str(directness).replace("EvidenceDirectness.", "")
            citations.append(f"Section: `{dir_str}`")

        if page_num is not None:
            if page_verified:
                citations.append(f"Page: {page_num} (Verified)")
            else:
                citations.append(f"Page: {page_num} (Unverified)")

        source_doc = item.get("source_filename")
        if source_doc:
            citations.append(f"Document: `{source_doc}`")

        claim_id = item.get("claim_id")
        if claim_id:
            citations.append(f"Ref: {claim_id}")

        notes = item.get("notes")
        if notes:
            citations.append(f"Note: {notes}")

        if citations:
            out.append(f"  - *Citation:* {', '.join(citations)}")

        if evidence_text:
            out.append(f"  - *Operative Excerpt:* \"{evidence_text.strip()}\"")

        return out
