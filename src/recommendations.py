"""
Phase 12: Business Recommendation Engine.
Provides transparent, data-grounded business recommendations for customer segments
and real-time prescriptive actions for individual purchase prediction scenarios.
"""

from typing import Dict, List, Any
import pandas as pd


class RecommendationEngine:
    def __init__(self):
        self.segment_rules = self._init_segment_rules()

    def _init_segment_rules(self) -> List[Dict[str, str]]:
        return [
            {
                "customer_segment": "Champions",
                "problem": "High-value loyalists may experience fatigue or lack of exclusive brand recognition.",
                "evidence": "985 customers (1.06% of base) driving R$ 366.9k in GMV with highest average spend of R$ 372.53 and 88.9 days recency.",
                "recommended_action": "Enroll in VIP Tier with early access to premium launches, dedicated customer concierge, and tier-based loyalty perks.",
                "business_objective": "Protect top-tier lifetime value (LTV) and foster organic brand advocacy."
            },
            {
                "customer_segment": "Potential Loyalists",
                "problem": "High first-order spenders at risk of dropping off if not engaged during the 60-90 day re-order window.",
                "evidence": "14,405 customers generating R$ 4.35M (28.2% of platform revenue) with high average spend (R$ 302.21) and fresh recency (91.9 days).",
                "recommended_action": "Trigger cross-category recommendation email journeys within 45 days of order completion, offering threshold discount for 2nd purchase.",
                "business_objective": "Accelerate conversion from one-time high spender to recurring repeat buyer."
            },
            {
                "customer_segment": "At Risk",
                "problem": "Critical revenue cohort drifting into dormancy; represents largest historical revenue generator now disengaged for over a year.",
                "evidence": "14,457 customers holding R$ 4.42M (28.7% of total revenue) with average recency of 391.9 days (over 13 months inactive).",
                "recommended_action": "Deploy high-impact 'We Miss You' win-back campaigns featuring dynamic personalized product catalogs and a limited-time 15% discount.",
                "business_objective": "Reactivate high-ticket purchasers before permanent churn."
            },
            {
                "customer_segment": "Can't Lose Them",
                "problem": "Former top repeat high-spenders who have completely ceased platform interactions.",
                "evidence": "323 customers who historically spent R$ 352.25 on multiple orders but are now 462.8 days inactive (15+ months).",
                "recommended_action": "Conduct executive outreach or personalized direct mail/SMS with high-value store credits and feedback survey to identify churn drivers.",
                "business_objective": "Recover high-value relationships and diagnose structural churn reasons."
            },
            {
                "customer_segment": "New Customers",
                "problem": "First-time buyers unfamiliar with marketplace catalog breadth; post-purchase engagement drops rapidly after 30 days.",
                "evidence": "10,886 customers with fresh recency (45.2 days) and modest initial spend (R$ 72.70).",
                "recommended_action": "Implement a structured 3-part onboarding sequence: Order satisfaction check-in -> Category discovery guide -> 10% welcome back incentive.",
                "business_objective": "Shorten the time to second purchase and establish ongoing buying habits."
            },
            {
                "customer_segment": "Promising",
                "problem": "Recent single-purchase buyers showing interest but lower initial basket size.",
                "evidence": "10,849 customers (11.6% of base) with 135.6 days average recency and R$ 73.16 average spend.",
                "recommended_action": "Promote bundle deals and free-shipping thresholds to increase basket size and encourage exploratory purchases in complementary categories.",
                "business_objective": "Increase Average Order Value (AOV) and stimulate catalog discovery."
            },
            {
                "customer_segment": "Loyal Customers",
                "problem": "Consistent repeat purchasers whose frequency has plateaued.",
                "evidence": "820 customers with steady repeat activity (avg 2.4 orders) generating R$ 204.6k with average recency of 183.6 days.",
                "recommended_action": "Introduce subscription replenishment for consumable categories (health/beauty, pet goods) and referral reward incentives.",
                "business_objective": "Lock in recurring subscription revenue and lower customer acquisition costs via referrals."
            },
            {
                "customer_segment": "Hibernating",
                "problem": "Large volume of low-to-medium spenders disengaged for 8 to 12 months.",
                "evidence": "29,202 customers (31.3% of total base) holding R$ 3.55M in historical spend with average recency of 257.2 days.",
                "recommended_action": "Re-engage via low-cost automated channels (push notifications, clearance sale alerts, seasonal holiday previews).",
                "business_objective": "Low-CAC reactivation during peak retail seasons (Black Friday, End of Year)."
            },
            {
                "customer_segment": "Lost",
                "problem": "Dormant single-order purchasers with minimal historical monetary value.",
                "evidence": "11,431 customers with average recency of 473.5 days and low average spend of R$ 72.49.",
                "recommended_action": "Suppress from expensive paid retargeting ad campaigns; maintain low-frequency drip email list only.",
                "business_objective": "Minimize ad spend waste and optimize marketing capital efficiency."
            }
        ]

    def get_all_recommendations(self) -> List[Dict[str, str]]:
        return self.segment_rules

    def get_individual_action(self, repeat_prob: float, segment: str, is_delayed: int = 0) -> Dict[str, str]:
        """Provides real-time prescriptive advice for individual customer prediction."""
        prob_pct = repeat_prob * 100

        if is_delayed == 1:
            return {
                "urgency": "CRITICAL / RETENTION INTERVENTION",
                "diagnosis": f"Delivery was delayed on initial order. Probability of repeat purchase is {prob_pct:.1f}%.",
                "action": "Dispatch an automated customer apology credit (R$ 20 voucher) + proactive customer support ticket to restore trust before attempting cross-sell.",
                "playbook": "Service Recovery Protocol"
            }

        if repeat_prob >= 0.50:
            return {
                "urgency": "HIGH PRIORITY / NURTURING",
                "diagnosis": f"High probability of repeat purchase ({prob_pct:.1f}%). Belongs to {segment} segment.",
                "action": "Trigger targeted 2nd-purchase cross-sell campaign within 30 days featuring top-rated complementary products in their category.",
                "playbook": "Growth & LTV Acceleration"
            }
        elif repeat_prob >= 0.35:
            return {
                "urgency": "MODERATE / VALUE INCENTIVE",
                "diagnosis": f"Moderate repeat purchase likelihood ({prob_pct:.1f}%).",
                "action": "Offer a limited-time Free Shipping voucher on their next order of R$ 100+ to remove shipping friction.",
                "playbook": "Friction Reduction Play"
            }
        else:
            return {
                "urgency": "STANDARD / LOW-COST NURTURE",
                "diagnosis": f"Lower baseline probability ({prob_pct:.1f}%). Common for one-time marketplace shoppers.",
                "action": "Include in standard weekly promotional newsletter. Avoid burning high-cost discount margins unless triggered by catalog browsing.",
                "playbook": "Organic Catalog Engagement"
            }


if __name__ == "__main__":
    engine = RecommendationEngine()
    recs = engine.get_all_recommendations()
    print(f"Generated {len(recs)} strategic business recommendation playbooks.")
    sample_action = engine.get_individual_action(0.62, "Potential Loyalists", is_delayed=0)
    print("Sample individual action:", sample_action)
