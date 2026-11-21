"""
Keyword search service for Government Orders.
Searches metadata and extracted document text in PostgreSQL.
"""

from sqlalchemy import or_, select

from app.database.connection import SessionLocal
from app.database.models import GovernmentOrderDB


class SearchService:
    """Search Government Orders using PostgreSQL."""

    def search_orders(
        self,
        keyword: str,
        limit: int = 10,
    ) -> list[dict]:
        """Search by GO number, department, filename or document text."""

        keyword = keyword.strip()

        if not keyword:
            raise ValueError("Search keyword cannot be empty.")

        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        search_pattern = f"%{keyword}%"

        statement = (
            select(GovernmentOrderDB)
            .where(
                or_(
                    GovernmentOrderDB.go_number.ilike(search_pattern),
                    GovernmentOrderDB.department.ilike(search_pattern),
                    GovernmentOrderDB.source_filename.ilike(search_pattern),
                    GovernmentOrderDB.extracted_text.ilike(search_pattern),
                )
            )
            .order_by(GovernmentOrderDB.id)
            .limit(limit)
        )

        with SessionLocal() as session:
            orders = session.scalars(statement).all()

            results = []

            for order in orders:
                results.append(
                    {
                        "go_number": order.go_number,
                        "issue_date": (
                            order.issue_date.isoformat()
                            if order.issue_date
                            else None
                        ),
                        "department": order.department,
                        "source_filename": order.source_filename,
                        "verification_status": order.verification_status,
                    }
                )

            return results