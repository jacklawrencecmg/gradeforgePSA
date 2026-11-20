"""
Card valuation service - integrates eBay and Card Ladder pricing data.
Estimates market value based on recent sales and comparable listings.
"""

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CardValuation:
    """Fetch and cache card valuations from multiple sources."""

    def __init__(self):
        self.ebay_key = os.getenv("EBAY_API_KEY", "")
        self.ebay_secret = os.getenv("EBAY_API_SECRET", "")
        self.cardladder_key = os.getenv("CARDLADDER_API_KEY", "")
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 86400  # 24 hours

    def get_valuation(
        self,
        card_name: str,
        set_name: str,
        card_number: str,
        grade: Optional[int] = None,
        finish: str = "standard"
    ) -> Dict[str, Any]:
        """
        Get card valuation from multiple sources.

        Args:
            card_name: e.g. "Charizard ex"
            set_name: e.g. "Scarlet & Violet"
            card_number: e.g. "223"
            grade: PSA grade (1-10), None if ungraded
            finish: standard, chrome, refractor, etc.

        Returns:
            {
                "status": "success" or "error",
                "estimated_value": "$45-$65",
                "low_estimate": 45,
                "high_estimate": 65,
                "mid_estimate": 55,
                "data_sources": ["ebay", "cardladder"],
                "grade_impact": "10% premium over PSA 9",
                "recent_sales_count": 12,
                "last_updated": "2026-11-20T10:30:00Z",
                "confidence": "high",
                "currency": "USD",
                "market_trend": "stable",
                "links": {
                    "ebay": "https://ebay.com/sch/...",
                    "cardladder": "https://cardladder.com/..."
                }
            }
        """

        # Create cache key
        cache_key = self._make_cache_key(card_name, set_name, card_number, grade, finish)

        # Check cache
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.fromisoformat(cached["cached_at"]) + timedelta(seconds=self.cache_ttl) > datetime.now():
                logger.info(f"Cache hit: {cache_key}")
                return cached["data"]

        # Fetch from sources
        results = {
            "status": "pending",
            "card_name": card_name,
            "set_name": set_name,
            "card_number": card_number,
            "grade": grade,
            "finish": finish,
            "data_sources": [],
            "low_estimate": None,
            "high_estimate": None,
            "mid_estimate": None,
            "recent_sales_count": 0,
            "confidence": "low",
            "currency": "USD",
            "market_trend": "unknown",
            "last_updated": datetime.now().isoformat() + "Z",
        }

        # Try eBay
        if self.ebay_key:
            try:
                ebay_data = self._fetch_ebay_prices(card_name, set_name, card_number, grade, finish)
                if ebay_data:
                    results["data_sources"].append("ebay")
                    results.update(ebay_data)
                    logger.info(f"eBay valuation: {ebay_data}")
            except Exception as e:
                logger.error(f"eBay fetch failed: {e}")

        # Try Card Ladder
        if self.cardladder_key:
            try:
                cl_data = self._fetch_cardladder_prices(card_name, set_name, card_number, grade)
                if cl_data:
                    results["data_sources"].append("cardladder")
                    # Merge Card Ladder data
                    if results["low_estimate"] is None:
                        results.update(cl_data)
                    logger.info(f"Card Ladder valuation: {cl_data}")
            except Exception as e:
                logger.error(f"Card Ladder fetch failed: {e}")

        # Calculate estimates if we have data
        if results["low_estimate"] is not None and results["high_estimate"] is not None:
            results["mid_estimate"] = int((results["low_estimate"] + results["high_estimate"]) / 2)
            results["estimated_value"] = f"${results['low_estimate']}-${results['high_estimate']}"
            results["confidence"] = "high" if len(results["data_sources"]) > 1 else "medium"
            results["status"] = "success"
        else:
            results["status"] = "no_data"
            results["estimated_value"] = "Data unavailable"

        # Cache the result
        self.cache[cache_key] = {
            "data": results,
            "cached_at": datetime.now().isoformat()
        }

        return results

    def _make_cache_key(self, card_name: str, set_name: str, card_number: str, grade: Optional[int], finish: str) -> str:
        """Create cache key from card details."""
        return f"{card_name}|{set_name}|{card_number}|{grade}|{finish}".lower()

    def _fetch_ebay_prices(
        self,
        card_name: str,
        set_name: str,
        card_number: str,
        grade: Optional[int],
        finish: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch prices from eBay API.
        Requires EBAY_API_KEY and EBAY_API_SECRET environment variables.

        Uses eBay Shopping API or Browse API to find recently sold listings.
        """

        if not self.ebay_key or not self.ebay_secret:
            return None

        try:
            # Build search query
            query = f"{card_name} {set_name}"
            if card_number:
                query += f" #{card_number}"
            if grade and grade >= 9:
                query += f" PSA {grade}"

            # TODO: Implement actual eBay API call
            # This is a placeholder that would call:
            # https://api.ebay.com/buy/browse/v1/item_summary/search
            #
            # For now, return None to indicate not implemented
            # User will need to add EBAY_API_KEY and implement the call

            logger.debug(f"eBay search query: {query}")
            return None

        except Exception as e:
            logger.error(f"eBay API error: {e}")
            return None

    def _fetch_cardladder_prices(
        self,
        card_name: str,
        set_name: str,
        card_number: str,
        grade: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch prices from Card Ladder.
        Requires CARDLADDER_API_KEY environment variable.

        Card Ladder provides aggregated pricing data for TCG cards.
        """

        if not self.cardladder_key:
            return None

        try:
            # Build search query
            query = f"{card_name} {set_name}"
            if card_number:
                query += f" #{card_number}"

            # TODO: Implement actual Card Ladder API call
            # Card Ladder may provide REST API for price lookups
            # This would call their pricing endpoint
            #
            # For now, return None to indicate not implemented
            # User will need to add CARDLADDER_API_KEY and implement the call

            logger.debug(f"Card Ladder search query: {query}")
            return None

        except Exception as e:
            logger.error(f"Card Ladder API error: {e}")
            return None

    def get_valuation_for_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method: get valuation from a GradeForge result.

        Extracts card info from the analysis result and fetches valuation.
        """

        card_name = result_data.get("label", "Unknown").split("#")[0].strip()
        card_number = result_data.get("label", "").split("#")[-1].strip() if "#" in result_data.get("label", "") else None

        grade = None
        if result_data.get("gem"):
            gem = result_data["gem"]
            # Try to extract grade from gem data
            if gem.get("most_likely_grade"):
                try:
                    grade = int(gem["most_likely_grade"])
                except (ValueError, TypeError):
                    pass

        finish = result_data.get("finish", "standard")

        return self.get_valuation(
            card_name=card_name,
            set_name="Unknown",  # May not be available
            card_number=card_number,
            grade=grade,
            finish=finish
        )

    def clear_cache(self):
        """Clear valuation cache."""
        self.cache.clear()
        logger.info("Valuation cache cleared")


# Global instance
_valuation_service: Optional[CardValuation] = None


def get_valuation_service() -> CardValuation:
    """Get or create global valuation service."""
    global _valuation_service
    if _valuation_service is None:
        _valuation_service = CardValuation()
    return _valuation_service
