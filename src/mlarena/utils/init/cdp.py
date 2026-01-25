"""CDP (Chrome DevTools Protocol) integration for Kaggle scraping."""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit
from typing import Optional

PLAYWRIGHT_OVERVIEW_SCRIPT = """
() => {
  const normalize = (text) => (text || '').replace(/\\s+/g, ' ').trim();
  const headingTags = ['H1','H2','H3','H4','H5','H6'];
  const headings = Array.from(document.querySelectorAll(headingTags.join(',')));
  const collectSiblings = (start) => {
    const parts = [];
    let cursor = start.nextElementSibling;
    while (cursor) {
      if (headingTags.includes(cursor.tagName)) {
        break;
      }
      const text = normalize(cursor.innerText || cursor.textContent || '');
      if (text) {
        parts.push(text);
      }
      cursor = cursor.nextElementSibling;
    }
    return parts.join('\\n\\n');
  };
  const extractSection = (title) => {
    const heading = headings.find(
      (node) => normalize(node.textContent).toLowerCase() === title
    );
    if (heading) {
      const section = heading.closest('section');
      if (section) {
        const sectionText = normalize(section.innerText || section.textContent || '');
        if (sectionText) {
          return sectionText;
        }
      }
      const fallbackText = collectSiblings(heading);
      if (fallbackText) {
        return fallbackText;
      }
    }
    const blocks = Array.from(document.querySelectorAll('section, article, div'));
    for (const block of blocks) {
      const text = normalize(block.innerText || block.textContent || '');
      if (text.toLowerCase().startsWith(title)) {
        return text;
      }
    }
    return '';
  };
  return {
    description: extractSection('description'),
    evaluation: extractSection('evaluation'),
  };
}
"""


def _normalize_cdp_url(url: Optional[str]) -> Optional[str]:
    """Force IPv4 loopback when a URL uses localhost."""
    if not url or "localhost" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url.replace("localhost", "127.0.0.1")
    if parts.hostname != "localhost":
        return url
    netloc = parts.netloc.replace("localhost", "127.0.0.1", 1)
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def resolve_cdp_url(custom_url: Optional[str]) -> Optional[str]:
    """Resolve CDP endpoint URL from custom param or environment."""
    if custom_url is not None:
        return _normalize_cdp_url(custom_url) or None
    env_url = os.environ.get("KAGGLE_CDP_URL") or os.environ.get("CDP_URL")
    if env_url is not None:
        return _normalize_cdp_url(env_url) or None
    return "http://127.0.0.1:9222"


async def fetch_overview_sections_via_cdp(
    competition_slug: str, cdp_url: str
) -> dict:
    """Connect to Chrome via CDP and scrape Description/Evaluation section text."""
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright

    playwright = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        if not contexts:
            raise RuntimeError("No browser contexts available via CDP")
        context = contexts[0]
        pages = context.pages
        page = pages[0] if pages else await context.new_page()
        url = f"https://www.kaggle.com/competitions/{competition_slug}/overview"
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(1000)
        except PlaywrightTimeoutError:
            return ""
        sections = await page.evaluate(PLAYWRIGHT_OVERVIEW_SCRIPT)
        if not isinstance(sections, dict):
            return {}
        return {k: (v or "").strip() for k, v in sections.items()}
    finally:
        if playwright:
            await playwright.stop()


def _combine_overview_sections(sections: dict) -> str:
    description = (sections.get("description") or "").strip()
    evaluation = (sections.get("evaluation") or "").strip()
    parts = []
    if description:
        parts.append(f"Description\\n{description}")
    if evaluation:
        parts.append(f"Evaluation\\n{evaluation}")
    return "\\n\\n".join(parts).strip()


def fetch_kaggle_evaluation(
    competition_slug: str, cdp_url: Optional[str] = None
) -> str:
    """Retrieve Evaluation section text for a Kaggle competition.

    Requires an active Chrome instance with remote debugging enabled.
    """
    import asyncio

    resolved_cdp = resolve_cdp_url(cdp_url)
    if not resolved_cdp:
        raise RuntimeError(
            "CDP endpoint not configured. Set KAGGLE_CDP_URL or pass --cdp-url to scrape the Evaluation section."
        )

    try:
        sections = asyncio.run(
            fetch_overview_sections_via_cdp(competition_slug, resolved_cdp)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch evaluation via CDP ({resolved_cdp}): {exc}"
        ) from exc

    evaluation = _combine_overview_sections(sections or {})
    if not evaluation:
        raise RuntimeError(
            f"Could not extract Description/Evaluation sections via CDP ({resolved_cdp}). "
            "Ensure the Kaggle page is accessible and you are logged in."
        )
    return evaluation
