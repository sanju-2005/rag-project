import re


def verify_citations(answer, sources):
    """Verify that page citations mentioned by the answer exist in retrieved sources."""
    available_pages = {str(s["page"]) for s in sources}
    cited_pages = set(re.findall(r"Page\s*(\d+)", answer, flags=re.I))
    if not cited_pages:
        return {"verified": False, "cited_pages": [], "reason": "No page citation found."}
    missing = sorted(cited_pages - available_pages)
    return {
        "verified": not missing,
        "cited_pages": sorted(cited_pages),
        "missing_pages": missing,
        "reason": "All cited pages were retrieved." if not missing else "Answer cites a page that was not retrieved."
    }
