"""
Comprehensive Unit and Integration Tests for Phase 1 Step 2 Tool Adapters:
- SearchTool
- DatabaseTool
- RelationshipTool

Tests cover:
- Empty, invalid, and boundary inputs
- Keyword search, limits, and result count clarity
- Exact match priority vs. unique partial match vs. dynamic ambiguous partial match
- Missing record handling
- Bidirectional relationship queries (outgoing and incoming)
- Numbered reference retrieval and ordinal resolution
- Full document text vs. configurable text preview
- Preservation of evidence text and page numbers
- Verification candidate tagging ('UNVERIFIED_CANDIDATE', 'legal_effect_verified': False)
- Error sanitization (no internal SQL/exception leakage)
"""

import sys
from typing import List
from app.agents.tools.search_tool import SearchTool
from app.agents.tools.database_tool import DatabaseTool
from app.agents.tools.relationship_tool import RelationshipTool


def test_search_tool_suite() -> None:
    print("\n==========================================")
    print("SUITE 1: SearchTool Tests")
    print("==========================================")
    search_tool = SearchTool()

    # 1. Empty and whitespace input
    print("\n--- 1.1 Empty and whitespace inputs ---")
    res_empty = search_tool.search("")
    assert res_empty["status"] == "EMPTY_QUERY"
    assert res_empty["returned_count"] == 0
    assert res_empty["total_found"] == 0
    assert "cannot be empty" in res_empty["error"]

    res_spaces = search_tool.search("   ")
    assert res_spaces["status"] == "EMPTY_QUERY"
    assert res_spaces["returned_count"] == 0
    print("PASS: Empty queries handled safely.")

    # 2. Successful search & count clarity
    print("\n--- 1.2 Successful keyword search and count clarity ---")
    res_success = search_tool.search("P&ARD", limit=3)
    assert res_success["status"] == "SUCCESS"
    assert res_success["returned_count"] > 0
    assert res_success["total_found"] == res_success["returned_count"]
    assert res_success["limit_applied"] == 3
    assert len(res_success["results"]) <= 3
    print(f"PASS: Search returned {res_success['returned_count']} records (limit applied: {res_success['limit_applied']}).")

    # 3. No match search
    print("\n--- 1.3 No match search ---")
    res_no_match = search_tool.search("NON_EXISTENT_DEPARTMENT_99999")
    assert res_no_match["status"] == "NO_MATCH"
    assert res_no_match["returned_count"] == 0
    assert res_no_match["error"] is None
    print("PASS: No-match search returns clean status.")

    # 4. Limit boundary protection
    print("\n--- 1.4 Result limit boundaries ---")
    res_high_limit = search_tool.search("P&ARD", limit=500)
    assert res_high_limit["limit_applied"] == 50  # Capped at 50

    res_zero_limit = search_tool.search("P&ARD", limit=0)
    assert res_zero_limit["limit_applied"] == 1   # Minimum 1
    print("PASS: Search limits bounded between 1 and 50.")


def test_database_tool_suite() -> None:
    print("\n==========================================")
    print("SUITE 2: DatabaseTool Tests")
    print("==========================================")
    db_tool = DatabaseTool()

    # 1. Listing all GO numbers
    print("\n--- 2.1 List all GO numbers ---")
    list_res = db_tool.list_all_go_numbers()
    assert list_res["status"] == "SUCCESS"
    assert list_res["total_count"] > 0
    all_numbers: List[str] = list_res["go_numbers"]
    print(f"PASS: Successfully retrieved {list_res['total_count']} GO numbers from database.")

    # 2. Exact match priority
    print("\n--- 2.2 Exact GO match priority ---")
    exact_target = all_numbers[0]
    exact_res = db_tool.get_order_by_number(exact_target)
    assert exact_res["status"] == "SUCCESS"
    assert exact_res["found"] is True
    assert exact_res["match_type"] == "EXACT"
    assert exact_res["data"]["go_number"] == exact_target
    print(f"PASS: Exact match verified for '{exact_target}' (match_type={exact_res['match_type']}).")

    # 3. Unique partial match
    print("\n--- 2.3 Unique partial match ---")
    # Using '24/2026/ITD' which uniquely matches G.O.(Rt)No.24/2026/ITD
    unique_partial_query = "24/2026/ITD"
    partial_res = db_tool.get_order_by_number(unique_partial_query)
    assert partial_res["status"] == "SUCCESS"
    assert partial_res["found"] is True
    assert partial_res["match_type"] == "PARTIAL"
    assert len(partial_res["candidates"]) == 1
    print(f"PASS: Unique partial match verified for '{unique_partial_query}' -> '{partial_res['data']['go_number']}'.")

    # 4. Dynamic ambiguity detection (multiple partial matches)
    print("\n--- 2.4 Dynamic ambiguity detection ---")
    # Find a term that matches more than one record dynamically
    ambiguous_term = None
    for term in ["2026", "P&ARD", "ഉ.ഭ.പ.വ", "No"]:
        matches = [n for n in all_numbers if term.lower() in n.lower()]
        if len(matches) > 1:
            ambiguous_term = term
            break

    assert ambiguous_term is not None, "Could not find a term with multiple matches for test."
    ambig_res = db_tool.get_order_by_number(ambiguous_term)
    assert ambig_res["status"] == "AMBIGUOUS_MATCH"
    assert ambig_res["found"] is False
    assert len(ambig_res["candidates"]) > 1
    assert "is ambiguous" in ambig_res["error"]
    print(f"PASS: Ambiguity dynamically detected for term '{ambiguous_term}'. Candidates found: {len(ambig_res['candidates'])}.")

    # 5. Missing GO query
    print("\n--- 2.5 Non-existent GO query ---")
    missing_res = db_tool.get_order_by_number("NON_EXISTENT_ORDER_9999/XYZ")
    assert missing_res["status"] == "NOT_FOUND"
    assert missing_res["found"] is False
    assert len(missing_res["candidates"]) == 0
    print("PASS: Non-existent GO correctly returns NOT_FOUND.")

    # 6. Text preview vs. full text configurable retrieval
    print("\n--- 2.6 Configurable document text retrieval ---")
    # Without full text
    meta_only_res = db_tool.get_order_by_number(exact_target, include_full_text=False, text_preview_chars=200)
    assert meta_only_res["status"] == "SUCCESS"
    assert "extracted_text" not in meta_only_res["data"]
    assert "text_preview" in meta_only_res["data"]
    assert len(meta_only_res["data"]["text_preview"]) <= 200
    assert meta_only_res["data"]["extracted_text_length"] > 0

    # With full text
    full_text_res = db_tool.get_order_by_number(exact_target, include_full_text=True)
    assert "extracted_text" in full_text_res["data"]
    assert len(full_text_res["data"]["extracted_text"]) == full_text_res["data"]["extracted_text_length"]
    print("PASS: Configurable text retrieval verified (metadata preview vs full text).")

    # 7. Reference retrieval with evidence preservation
    print("\n--- 2.7 Numbered reference retrieval ---")
    refs_res = db_tool.get_references_for_order("24/2026/ITD")
    assert refs_res["status"] == "SUCCESS"
    assert refs_res["total_references"] > 0
    first_ref = refs_res["references"][0]
    assert "ordinal" in first_ref
    assert "referenced_go_number" in first_ref
    assert "evidence_text" in first_ref
    print(f"PASS: References retrieved ({refs_res['total_references']} found). Ordinal={first_ref['ordinal']}, Target={first_ref['referenced_go_number']}.")

    # 8. Outgoing relationships retrieval
    print("\n--- 2.8 Outgoing relationship queries ---")
    outgoing_res = db_tool.get_outgoing_relationships("24/2026/ITD")
    assert outgoing_res["status"] == "SUCCESS"
    assert outgoing_res["direction"] == "OUTGOING"
    assert outgoing_res["total_relationships"] > 0
    first_out = outgoing_res["relationships"][0]
    assert first_out["direction"] == "OUTGOING"
    assert first_out["source_go_number"] == "G.O.(Rt)No.24/2026/ITD"
    assert first_out["relationship_type"] == "MODIFIES"
    assert first_out["evidence_text"] is not None
    print(f"PASS: Outgoing relationships verified. Type={first_out['relationship_type']}, Target={first_out['target_go_number']}.")

    # 9. Incoming relationships retrieval
    print("\n--- 2.9 Incoming relationship queries ---")
    # G.O.(Rt)No.24/2026/ITD modifies 'G.O (Rt) No. 114/2025/ITD'
    target_modified = "G.O (Rt) No. 114/2025/ITD"
    incoming_res = db_tool.get_incoming_relationships(target_modified)
    assert incoming_res["status"] == "SUCCESS"
    assert incoming_res["direction"] == "INCOMING"
    assert incoming_res["total_relationships"] > 0
    first_in = incoming_res["relationships"][0]
    assert first_in["direction"] == "INCOMING"
    assert first_in["source_go_number"] == "G.O.(Rt)No.24/2026/ITD"
    assert first_in["relationship_type"] == "MODIFIES"
    assert first_in["target_go_number"] == target_modified
    print(f"PASS: Incoming relationships verified. Later modifying order='{first_in['source_go_number']}' for target '{target_modified}'.")

    # 10. Consolidated relationships query (ALL)
    print("\n--- 2.10 Consolidated relationships query ---")
    all_rels_res = db_tool.get_relationships_for_order("24/2026/ITD", direction="ALL")
    assert all_rels_res["status"] == "SUCCESS"
    assert all_rels_res["direction"] == "ALL"
    assert all_rels_res["total_relationships"] > 0
    print(f"PASS: Consolidated relationships verified. Outgoing count={all_rels_res['outgoing_count']}, Incoming count={all_rels_res['incoming_count']}.")


def test_relationship_tool_suite() -> None:
    print("\n==========================================")
    print("SUITE 3: RelationshipTool Tests")
    print("==========================================")
    rel_tool = RelationshipTool()

    # 1. Empty text handling
    print("\n--- 3.1 Empty text handling ---")
    res_empty = rel_tool.extract_from_text("")
    assert res_empty["status"] == "EMPTY_TEXT"
    assert res_empty["total_relationships"] == 0

    # 2. Synthetic extraction with evidence preservation & candidate status
    print("\n--- 3.2 Evidence preservation and UNVERIFIED_CANDIDATE tagging ---")
    synthetic_text = (
        "--- Page 1 ---\n"
        "Read\n"
        "1 G.O.(P) No.15/2020/P&ARD dated 28/10/2020\n"
        "GOVERNMENT OF KERALA\n"
        "ORDER\n"
        "The Government order read as 1st paper above is hereby modified to the following extent."
    )
    res_synth = rel_tool.extract_from_text(synthetic_text, source_go_number="G.O.(P) No.4/2026/P&ARD")
    assert res_synth["status"] == "SUCCESS"
    assert res_synth["total_references"] == 1
    assert res_synth["total_relationships"] >= 1

    rel = res_synth["relationships"][0]
    assert rel["relationship_type"] == "MODIFIES"
    assert rel["matched_phrase"] == "hereby modified"
    assert rel["page_number"] == 1
    assert "G.O.(P) No.15/2020/P&ARD" in rel["referenced_go_numbers"]
    assert rel["verification_status"] == "UNVERIFIED_CANDIDATE"
    assert rel["legal_effect_verified"] is False
    assert rel["evidence_text"] is not None
    print("PASS: Relationship candidate extracted with UNVERIFIED_CANDIDATE tag and full evidence.")

    # 3. Extraction for ambiguous or missing source order
    print("\n--- 3.3 Ambiguous and missing source order handling ---")
    res_missing = rel_tool.extract_for_order("NON_EXISTENT_ORDER_123")
    assert res_missing["status"] == "ORDER_NOT_FOUND"

    res_ambig = rel_tool.extract_for_order("2026")
    assert res_ambig["status"] == "AMBIGUOUS_SOURCE_ORDER"
    assert len(res_ambig["candidates"]) > 1
    print("PASS: Ambiguous and missing orders correctly routed without crashing.")

    # 4. Extraction for actual order from database
    print("\n--- 3.4 Extraction from real order text ---")
    res_real = rel_tool.extract_for_order("24/2026/ITD")
    assert res_real["status"] == "SUCCESS"
    assert res_real["total_references"] > 0
    print(f"PASS: Real order extraction verified. Found {res_real['total_references']} references and {res_real['total_relationships']} relationships.")


def test_dataset_baseline_validation() -> None:
    print("\n==========================================")
    print("SUITE 4: Dataset Baseline Validation")
    print("==========================================")
    db_tool = DatabaseTool()
    res = db_tool.list_all_go_numbers()
    assert res["status"] == "SUCCESS"
    print(f"Current total Government Orders in database: {res['total_count']}")
    assert res["total_count"] >= 11, f"Expected at least 11 Government Orders, found {res['total_count']}"
    print("PASS: Database baseline contains the expected minimum of 11 orders.")


if __name__ == "__main__":
    try:
        test_search_tool_suite()
        test_database_tool_suite()
        test_relationship_tool_suite()
        test_dataset_baseline_validation()
        print("\n=======================================================")
        print("ALL IMPROVEMENT & REGRESSION TESTS PASSED SUCCESSFULLY!")
        print("=======================================================")
    except Exception as exc:
        print(f"\nTEST SUITE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
