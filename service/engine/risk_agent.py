"""
Financial Risk Agent
Computes financial ratios, scores risk deterministically, and generates
recommendations — replacing LLM guesswork with real calculated math.

Input contract:
    financial_data: dict with (all values in same currency unit, floats)
        current_assets, current_liabilities, inventory,
        total_debt, total_equity, total_assets,
        ebit, interest_expense, net_income, revenue,
        ebitda, operating_cash_flow
    Missing keys are treated as unavailable — that ratio is skipped,
    not treated as zero (avoids silently wrong conclusions).
"""

from typing import Optional


# ---------------------------------------------------------------------------
# 1. Ratio Calculator
# ---------------------------------------------------------------------------

def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def calculate_ratios(data: dict) -> dict:
    """Compute the 10 core ratios. Returns None for any ratio whose inputs
    are missing or invalid, rather than guessing."""
    g = data.get  # shorthand

    return {
        "current_ratio": _safe_div(g("current_assets"), g("current_liabilities")),
        "quick_ratio": _safe_div(
            None if g("current_assets") is None or g("inventory") is None
            else g("current_assets") - g("inventory"),
            g("current_liabilities"),
        ),
        "debt_to_equity": _safe_div(g("total_debt"), g("total_equity")),
        "debt_ratio": _safe_div(g("total_debt"), g("total_assets")),
        "interest_coverage_ratio": _safe_div(g("ebit"), g("interest_expense")),
        "profit_margin": _safe_div(g("net_income"), g("revenue")),
        "roa": _safe_div(g("net_income"), g("total_assets")),
        "roe": _safe_div(g("net_income"), g("total_equity")),
        "ebitda_margin": _safe_div(g("ebitda"), g("revenue")),
        "cash_flow_ratio": _safe_div(g("operating_cash_flow"), g("current_liabilities")),
    }


# ---------------------------------------------------------------------------
# 2. Industry Threshold Rules
# ---------------------------------------------------------------------------
# Each ratio maps to ascending (low_bound, health_points) bands.
# health_points: 0 = critical, 25 = high risk, 60 = moderate, 100 = healthy.
# These are general-purpose defaults; swap in per-industry tables later if
# the engine needs to differentiate sectors (e.g. retail vs. banking carry
# very different "normal" debt ratios).

THRESHOLDS = {
    "current_ratio":           [(1.0, 0), (1.5, 25), (3.0, 60), (float("inf"), 100)],
    "quick_ratio":              [(0.5, 0), (1.0, 25), (1.5, 60), (float("inf"), 100)],
    "debt_to_equity":          [(2.0, 100), (1.0, 60), (0.5, 25), (0.0, 0)],  # lower is better -> reversed below
    "debt_ratio":               [(0.3, 100), (0.5, 60), (0.7, 25), (1.0, 0)],  # lower is better -> reversed below
    "interest_coverage_ratio": [(1.0, 0), (1.5, 25), (3.0, 60), (float("inf"), 100)],
    "profit_margin":            [(0.0, 0), (0.05, 25), (0.10, 60), (float("inf"), 100)],
    "roa":                       [(0.0, 0), (0.05, 25), (0.10, 60), (float("inf"), 100)],
    "roe":                       [(0.0, 0), (0.10, 25), (0.15, 60), (float("inf"), 100)],
    "ebitda_margin":            [(0.0, 0), (0.10, 25), (0.20, 60), (float("inf"), 100)],
    "cash_flow_ratio":          [(0.5, 0), (1.0, 25), (1.5, 60), (float("inf"), 100)],
}

# Ratios where LOWER is riskier-when-high (need descending bands instead)
_LOWER_IS_BETTER = {"debt_to_equity", "debt_ratio"}


def score_ratio(name: str, value: Optional[float]) -> Optional[int]:
    """Map a single ratio value to a 0-100 health score using threshold bands."""
    if value is None or name not in THRESHOLDS:
        return None

    bands = THRESHOLDS[name]

    if name in _LOWER_IS_BETTER:
        # value below first bound -> best score; higher value -> worse
        for bound, points in bands:
            if value <= bound:
                return points
        return 0
    else:
        for bound, points in bands:
            if value <= bound:
                return points
        return 100


# ---------------------------------------------------------------------------
# 3. Risk Scoring Engine
# ---------------------------------------------------------------------------

def score_risk(ratios: dict) -> dict:
    """Aggregate per-ratio health scores into an overall 0-100 score and
    risk level. Ratios that couldn't be computed are excluded, not zeroed."""
    per_ratio_scores = {}
    for name, value in ratios.items():
        s = score_ratio(name, value)
        if s is not None:
            per_ratio_scores[name] = s

    if not per_ratio_scores:
        return {"score": None, "risk_level": "Unknown", "per_ratio_scores": {}}

    overall = round(sum(per_ratio_scores.values()) / len(per_ratio_scores))

    if overall >= 80:
        risk_level = "Low"
    elif overall >= 60:
        risk_level = "Moderate"
    elif overall >= 40:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return {"score": overall, "risk_level": risk_level, "per_ratio_scores": per_ratio_scores}


# ---------------------------------------------------------------------------
# 4. Recommendation Generator
# ---------------------------------------------------------------------------

_RECOMMENDATIONS = {
    "current_ratio": "Current ratio of {v:.2f} is below a healthy level — consider reducing short-term liabilities or building up liquid current assets.",
    "quick_ratio": "Quick ratio of {v:.2f} suggests limited ability to cover short-term obligations without selling inventory — improve cash/receivables position.",
    "debt_to_equity": "Debt-to-equity of {v:.2f} indicates high leverage relative to equity — prioritize debt reduction or equity raises before taking on new liabilities.",
    "debt_ratio": "Debt ratio of {v:.2f} means a large share of assets are financed by debt — monitor solvency closely and avoid further borrowing.",
    "interest_coverage_ratio": "Interest coverage of {v:.2f}x is low — earnings may struggle to cover interest obligations if performance dips.",
    "profit_margin": "Profit margin of {v:.1%} is weak — review cost structure and pricing to improve profitability.",
    "roa": "Return on assets of {v:.1%} is low — assets are not being used efficiently to generate earnings.",
    "roe": "Return on equity of {v:.1%} is low relative to typical benchmarks — investigate capital efficiency.",
    "ebitda_margin": "EBITDA margin of {v:.1%} is under industry norms — assess operating cost pressures.",
    "cash_flow_ratio": "Cash flow ratio of {v:.2f} shows weak operating cash coverage of current liabilities — monitor cash flow closely.",
}


def generate_recommendations(ratios: dict, per_ratio_scores: dict, threshold: int = 60) -> list:
    """Flag any ratio scoring below `threshold` (moderate) with a targeted recommendation."""
    recs = []
    for name, score in per_ratio_scores.items():
        if score < threshold and ratios.get(name) is not None:
            template = _RECOMMENDATIONS.get(name)
            if template:
                recs.append(template.format(v=ratios[name]))
    if not recs:
        recs.append("All computed ratios are within healthy ranges — no immediate action needed.")
    return recs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def analyze_financial_risk(financial_data: dict) -> dict:
    """Main entry point. Given extracted financial line items, returns a
    structured risk assessment — output shape mirrors response_generator's
    JSON contract so the two can be merged or compared downstream."""
    ratios = calculate_ratios(financial_data)
    scoring = score_risk(ratios)
    recommendations = generate_recommendations(ratios, scoring["per_ratio_scores"])

    return {
        "ratios": ratios,
        "score": scoring["score"],
        "risk_level": scoring["risk_level"],
        "recommendations": recommendations,
    }