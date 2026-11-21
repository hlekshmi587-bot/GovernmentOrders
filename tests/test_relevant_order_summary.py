"""
Unit and regression tests for the relevant Government Order summary,
recorded status resolution, SQLAlchemy session recovery, and category normalization.
"""

from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from app.agents.specialized.response_agent import ResponseAgent, ResponseAgentInput
from app.api.schemas import RelationshipItemSchema
from app.database.models import VerificationRecordDB
from app.services.relationship_resolution import (
    resolve_incoming_relationships_for_order,
    resolve_relationship_verification_status,
)


class TestVerificationRecordResolution(unittest.TestCase):
    """Test resolution of verification status according to effective timestamps and IDs."""

    def test_newer_rejected_overrides_older_verified(self):
        """A newer REJECTED record must override an older VERIFIED record for the same relationship."""
        t1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)

        rec_verified = VerificationRecordDB(
            id=10,
            relationship_id=100,
            verification_status="VERIFIED",
            verified_at=t1,
            created_at=t1,
        )
        rec_rejected = VerificationRecordDB(
            id=11,
            relationship_id=100,
            verification_status="REJECTED",
            verified_at=t2,
            created_at=t2,
        )

        status = resolve_relationship_verification_status([rec_verified, rec_rejected])
        self.assertEqual(status, "REJECTED")

    def test_newer_verified_overrides_older_rejected(self):
        """A newer VERIFIED record overrides an older REJECTED record."""
        t1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)

        rec_rejected = VerificationRecordDB(
            id=10,
            relationship_id=100,
            verification_status="REJECTED",
            verified_at=t1,
            created_at=t1,
        )
        rec_verified = VerificationRecordDB(
            id=11,
            relationship_id=100,
            verification_status="VERIFIED",
            verified_at=t2,
            created_at=t2,
        )

        status = resolve_relationship_verification_status([rec_rejected, rec_verified])
        self.assertEqual(status, "VERIFIED")

    def test_tie_breaking_by_id_descending(self):
        """When timestamps are identical, deterministic tie-breaker is rec.id descending."""
        t = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        rec_lower_id = VerificationRecordDB(
            id=5,
            relationship_id=100,
            verification_status="VERIFIED",
            verified_at=t,
            created_at=t,
        )
        rec_higher_id = VerificationRecordDB(
            id=9,
            relationship_id=100,
            verification_status="REJECTED",
            verified_at=t,
            created_at=t,
        )

        status = resolve_relationship_verification_status([rec_lower_id, rec_higher_id])
        self.assertEqual(status, "REJECTED")

    def test_empty_records_returns_unrecorded(self):
        """If no records exist, return UNRECORDED."""
        self.assertEqual(resolve_relationship_verification_status([]), "UNRECORDED")


class TestRecordedStatusMapping(unittest.TestCase):
    """Test ResponseAgent.compute_order_recorded_status normalized categories and decisions."""

    def _make_rel(self, rel_type: str, status: str = "VERIFIED", src_go: str = "GO(P)No.99/2026/FIN"):
        return RelationshipItemSchema(
            id=1,
            direction="INCOMING",
            source_go_id=10,
            source_go_number=src_go,
            target_go_number="GO(P)No.50/2025/FIN",
            relationship_type=rel_type,
            matched_phrase="sample phrase",
            evidence_text="sample evidence",
            page_number=1,
            confidence="HIGH",
            extraction_method="RULE_BASED",
            review_required=False,
            verification_status=status,
            legal_effect_verified=False,
        )

    def test_rejected_returns_status_requires_further_verification(self):
        """Rejected evidence must NOT return empty status; it must return 'Status requires further verification'."""
        rel = self._make_rel("AMENDS", status="REJECTED")
        status, related_go = ResponseAgent.compute_order_recorded_status([rel], lookup_success=True)
        self.assertEqual(status, "Status requires further verification")
        self.assertIsNone(related_go)

    def test_empty_records_returns_status_not_established(self):
        """A successful lookup with 0 records returns 'Status not established in indexed records'."""
        status, related_go = ResponseAgent.compute_order_recorded_status([], lookup_success=True)
        self.assertEqual(status, "Status not established in indexed records")
        self.assertIsNone(related_go)

    def test_lookup_failure_returns_status_requires_further_verification(self):
        """Lookup failure (lookup_success=False) returns 'Status requires further verification'."""
        status, related_go = ResponseAgent.compute_order_recorded_status([], lookup_success=False)
        self.assertEqual(status, "Status requires further verification")
        self.assertIsNone(related_go)

    def test_amends_and_modifies_are_same_category(self):
        """AMENDS plus MODIFIES belong to the Amendment category and do not conflict."""
        rel1 = self._make_rel("AMENDS", status="VERIFIED", src_go="GO(P)No.101/2026/FIN")
        rel2 = self._make_rel("MODIFIES", status="VERIFIED", src_go="GO(P)No.102/2026/FIN")
        status, related_go = ResponseAgent.compute_order_recorded_status([rel1, rel2], lookup_success=True)
        self.assertEqual(status, "Amended in indexed records")
        self.assertEqual(related_go, "GO(P)No.101/2026/FIN")

    def test_cancels_and_revokes_are_same_category(self):
        """CANCELS plus REVOKES belong to the Cancellation category and do not conflict."""
        rel1 = self._make_rel("CANCELS", status="VERIFIED", src_go="GO(P)No.101/2026/FIN")
        rel2 = self._make_rel("REVOKES", status="VERIFIED", src_go="GO(P)No.102/2026/FIN")
        status, related_go = ResponseAgent.compute_order_recorded_status([rel1, rel2], lookup_success=True)
        self.assertEqual(status, "Cancellation recorded — verification required")
        self.assertEqual(related_go, "GO(P)No.101/2026/FIN")

    def test_supersedes_category(self):
        """SUPERSEDES returns 'Superseded in indexed records'."""
        rel = self._make_rel("SUPERSEDES", status="VERIFIED", src_go="GO(P)No.103/2026/FIN")
        status, related_go = ResponseAgent.compute_order_recorded_status([rel], lookup_success=True)
        self.assertEqual(status, "Superseded in indexed records")
        self.assertEqual(related_go, "GO(P)No.103/2026/FIN")

    def test_conflicting_categories_return_status_requires_further_verification(self):
        """Different categories (AMENDS + CANCELS) conflict and return 'Status requires further verification'."""
        rel1 = self._make_rel("AMENDS", status="VERIFIED")
        rel2 = self._make_rel("CANCELS", status="VERIFIED")
        status, related_go = ResponseAgent.compute_order_recorded_status([rel1, rel2], lookup_success=True)
        self.assertEqual(status, "Status requires further verification")
        self.assertIsNone(related_go)

    def test_pending_or_unrecorded_returns_status_requires_further_verification(self):
        """Pending or unrecorded status returns 'Status requires further verification'."""
        rel_pending = self._make_rel("AMENDS", status="PENDING")
        status, related_go = ResponseAgent.compute_order_recorded_status([rel_pending], lookup_success=True)
        self.assertEqual(status, "Status requires further verification")
        self.assertIsNone(related_go)


class TestRelevanceSelection(unittest.TestCase):
    """Test text-grounded and paraphrase-aware candidate selection in ResponseAgent."""

    def test_selects_orders_grounded_in_snippet_and_subject(self):
        """Selects order with genuine text match over order matching only department or GO number."""
        query = "dearness relief pension"
        results = [
            {
                "id": 1,
                "go_number": "GO(P)No.1/2025/FIN",
                "department": "Finance Department",
                "subject": "Motor vehicle tax exemption rules",
                "text_snippet": "Inspection of transport vehicles under finance department rules.",
                "relevance_score": 0.85,
            },
            {
                "id": 2,
                "go_number": "GO(P)No.2/2025/FIN",
                "department": "Finance Department",
                "subject": "Revision of Dearness Relief to State Pensioners",
                "text_snippet": "Sanctioning revised rates of dearness relief to service pensioners with arrears.",
                "relevance_score": 0.80,
            },
        ]
        selected = ResponseAgent.select_relevant_orders(query, results, max_orders=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], 2)
        matched = selected[0].get("_matched_terms", [])
        self.assertTrue(any("pension" in m or "pensioner" in m for m in matched))

    def test_handles_paraphrase_and_morphological_variants(self):
        """Matches morphological variations such as pensioners and relieving."""
        query = "pension relief"
        results = [
            {
                "id": 3,
                "go_number": "GO(P)No.3/2025/FIN",
                "department": "Finance",
                "subject": "Benefits for retired state employees",
                "text_snippet": "Payment of relief arrears to pensioners across all district treasuries.",
            }
        ]
        selected = ResponseAgent.select_relevant_orders(query, results, max_orders=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], 3)

    def test_generic_topic_selection_across_unrelated_domains(self):
        """
        Demonstrates that selection logic is generic and works for topics, departments,
        electricity, education, salaries, and policy questions without hardcoded domain lists.
        """
        # 1. Electricity / Power Sector
        elec_query = "solar rooftop subsidy tariff"
        elec_results = [
            {
                "id": 101,
                "go_number": "GO(P)No.101/2025/PD",
                "department": "Power Department",
                "subject": "Incentives and tariff rules for solar rooftop installations",
                "text_snippet": "Implementation of subsidy assistance for domestic rooftop solar consumers.",
            },
            {
                "id": 102,
                "go_number": "GO(P)No.102/2025/PD",
                "department": "Power Department",
                "subject": "Hydroelectric dam reservoir safety inspection guidelines",
                "text_snippet": "Annual monsoon dam monitoring protocol and structural inspection.",
            },
        ]
        elec_sel = ResponseAgent.select_relevant_orders(elec_query, elec_results, max_orders=1)
        self.assertEqual(elec_sel[0]["id"], 101)

        # 2. Education / Teachers
        edu_query = "higher secondary teacher transfer norms"
        edu_results = [
            {
                "id": 201,
                "go_number": "GO(P)No.201/2025/GED",
                "department": "General Education Department",
                "subject": "General transfer guidelines for higher secondary school teachers",
                "text_snippet": "Criteria and online application process for teacher postings and transfers.",
            },
            {
                "id": 202,
                "go_number": "GO(P)No.202/2025/GED",
                "department": "General Education Department",
                "subject": "Free mid-day meal scheme allocation for primary schools",
                "text_snippet": "Funds released for cooking cost and grains for government primary schools.",
            },
        ]
        edu_sel = ResponseAgent.select_relevant_orders(edu_query, edu_results, max_orders=1)
        self.assertEqual(edu_sel[0]["id"], 201)

        # 3. Natural Language Question
        nl_query = "What are the rules for maternity leave surrender benefits?"
        nl_results = [
            {
                "id": 301,
                "go_number": "GO(P)No.301/2025/HED",
                "department": "Higher Education",
                "subject": "Hostel fee revision in engineering colleges",
                "text_snippet": "Revised mess charges and room rent for government hostel inmates.",
            },
            {
                "id": 302,
                "go_number": "GO(P)No.302/2025/P&ARD",
                "department": "Personnel and Administrative Reforms",
                "subject": "Sanction of maternity leave and surrender of earned leave",
                "text_snippet": "Orders issued clarifying leave surrender and maternity benefits for state employees.",
            },
        ]
        nl_sel = ResponseAgent.select_relevant_orders(nl_query, nl_results, max_orders=1)
        self.assertEqual(nl_sel[0]["id"], 302)


class TestSqlAlchemySessionRecovery(unittest.TestCase):
    """Test session rollback and per-order recovery."""

    def test_rollback_on_first_order_allows_second_order_to_succeed(self):
        """
        When Order 1's query fails and triggers rollback, Order 2's query proceeds
        and succeeds, isolating the failure.
        """
        mock_db = MagicMock(spec=Session)

        # Mock result object for the second call
        mock_result = MagicMock()
        mock_result.all.return_value = []

        # First call raises an exception, second call returns result object
        mock_db.execute.side_effect = [
            Exception("Database connection transient failure"),
            mock_result,
        ]

        # Call Order 1 (fails)
        rels1, success1 = resolve_incoming_relationships_for_order(
            order_id=1,
            go_number="GO(P)No.1/2025/FIN",
            db=mock_db,
        )
        self.assertFalse(success1)
        self.assertEqual(rels1, [])

        # Session recovers via rollback
        mock_db.rollback()

        # Call Order 2 (succeeds)
        rels2, success2 = resolve_incoming_relationships_for_order(
            order_id=2,
            go_number="GO(P)No.2/2025/FIN",
            db=mock_db,
        )
        self.assertTrue(success2)
        self.assertEqual(rels2, [])


class TestMandatoryDisclaimerAndInvariants(unittest.TestCase):
    """Test preservation of official disclaimer and legal invariants."""

    def test_mandatory_disclaimer_verbatim_present(self):
        """The mandatory disclaimer must be present verbatim in synthesized output."""
        agent = ResponseAgent()
        resp_input = ResponseAgentInput(
            user_query="dearness relief pension",
            intent="KEYWORD_SEARCH",
            verification_status=None,
            confidence="HIGH",
            search_results=[
                {
                    "id": 10,
                    "go_number": "GO(P)No.10/2025/FIN",
                    "department": "Finance Department",
                    "issue_date": "2025-03-01",
                    "subject": "Revision of Dearness Relief to Pensioners",
                    "text_snippet": "Sanctioning revised rates of dearness relief to pensioners.",
                }
            ],
            order_recorded_statuses={
                10: {
                    "status": "Amended in indexed records",
                    "related_go": "GO(P)No.50/2025/FIN",
                }
            },
            legal_effect_verified=False,
        )

        final_response = agent.generate_response(resp_input)
        self.assertIn(ResponseAgent.MANDATORY_DISCLAIMER, final_response)
        self.assertIn("Amended in indexed records", final_response)
        self.assertIn("GO(P)No.50/2025/FIN", final_response)
        self.assertIn("Relevant Government Order", final_response)


if __name__ == "__main__":
    unittest.main()
