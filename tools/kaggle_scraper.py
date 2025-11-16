#!/usr/bin/env python3
"""
Kaggle Competition Scraper via CDP
Automatically fetch leaderboard and submissions data from Kaggle competitions
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import argparse
from playwright.async_api import async_playwright, TimeoutError


class KaggleScraper:
    """Scrape Kaggle competition data via CDP connection"""

    def __init__(self, cdp_url: str = "http://localhost:9222"):
        self.cdp_url = cdp_url
        self.browser = None
        self.page = None

    async def connect(self):
        """Connect to browser via CDP"""
        print(f"Connecting to browser via CDP on {self.cdp_url}...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)

        # Get existing page or create new one
        pages = self.browser.contexts[0].pages
        if not pages:
            print("No pages found. Creating new page...")
            self.page = await self.browser.contexts[0].new_page()
        else:
            self.page = pages[0]

        print(f"Connected to page: {self.page.url}")

    async def close(self):
        """Close connection"""
        if self.playwright:
            await self.playwright.stop()

    async def navigate_to_leaderboard(self, competition: str):
        """Navigate to competition leaderboard"""
        url = f"https://www.kaggle.com/competitions/{competition}/leaderboard"
        print(f"Navigating to {url}...")

        try:
            await self.page.goto(url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(2)  # Wait for dynamic content
            print("✓ Page loaded")
        except TimeoutError:
            print("Warning: Timeout waiting for page load, continuing anyway...")

    async def navigate_to_submissions(self, competition: str):
        """Navigate to user's submissions page"""
        url = f"https://www.kaggle.com/competitions/{competition}/submissions"
        print(f"Navigating to {url}...")

        try:
            await self.page.goto(url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(2)  # Wait for dynamic content
            print("✓ Page loaded")
        except TimeoutError:
            print("Warning: Timeout waiting for page load, continuing anyway...")

    async def scrape_leaderboard(self, limit: int = 20) -> List[Dict]:
        """
        Scrape leaderboard data

        Returns list of:
        {
            "rank": int,
            "team_name": str,
            "score": float,
            "entries": int,
            "last_submission": str
        }
        """
        print(f"Scraping leaderboard (top {limit})...")

        # Wait for leaderboard table to load
        try:
            await self.page.wait_for_selector('table, [role="table"]', timeout=10000)
        except TimeoutError:
            print("Warning: Leaderboard table not found")
            return []

        # Get accessibility snapshot
        snapshot = await self.page.accessibility.snapshot()

        # Parse leaderboard from snapshot
        leaderboard_data = []

        def extract_leaderboard(node, depth=0):
            """Recursively extract leaderboard data from accessibility tree"""
            if not node:
                return

            # Look for table rows
            role = node.get('role', '')
            name = node.get('name', '')

            # Try to find leaderboard entries
            if 'listitem' in role.lower() or 'row' in role.lower():
                # Extract text content
                if name and any(char.isdigit() for char in str(name)):
                    # This might be a leaderboard entry
                    children = node.get('children', [])
                    if children:
                        entry_text = ' '.join([
                            c.get('name', '') for c in children
                            if c.get('name')
                        ])
                        if entry_text:
                            leaderboard_data.append(entry_text)

            # Recurse into children
            for child in node.get('children', []):
                extract_leaderboard(child, depth + 1)

        extract_leaderboard(snapshot)

        print(f"✓ Found {len(leaderboard_data)} leaderboard entries")
        return leaderboard_data[:limit]

    async def scrape_submissions(self) -> List[Dict]:
        """
        Scrape user's submissions

        Returns list of:
        {
            "filename": str,
            "date": str,
            "status": str,
            "public_score": float,
            "private_score": float
        }
        """
        print("Scraping submissions...")

        # Wait for submissions table
        try:
            await self.page.wait_for_selector('table, [role="table"]', timeout=10000)
        except TimeoutError:
            print("Warning: Submissions table not found")
            return []

        # Get accessibility snapshot
        snapshot = await self.page.accessibility.snapshot()

        # Parse submissions
        submissions_data = []

        def extract_submissions(node, depth=0):
            """Recursively extract submissions from accessibility tree"""
            if not node:
                return

            role = node.get('role', '')
            name = node.get('name', '')

            # Look for submission entries
            if name and ('submission' in name.lower() or '.csv' in name.lower()):
                submissions_data.append({
                    'text': name,
                    'role': role
                })

            # Recurse
            for child in node.get('children', []):
                extract_submissions(child, depth + 1)

        extract_submissions(snapshot)

        print(f"✓ Found {len(submissions_data)} submissions")
        return submissions_data

    async def get_current_page_data(self) -> Dict:
        """Get comprehensive data from current page"""
        snapshot = await self.page.accessibility.snapshot()

        return {
            'url': self.page.url,
            'title': await self.page.title(),
            'snapshot': snapshot,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


async def scrape_competition(
    competition: str,
    cdp_url: str = "http://localhost:9222",
    output_dir: Optional[Path] = None
):
    """
    Scrape competition leaderboard and submissions

    Args:
        competition: Competition name (e.g., 'playground-series-s5e11')
        cdp_url: CDP connection URL
        output_dir: Directory to save results
    """
    scraper = KaggleScraper(cdp_url)

    try:
        await scraper.connect()

        results = {
            'competition': competition,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'leaderboard': {},
            'submissions': {}
        }

        # Scrape leaderboard
        print("\n=== LEADERBOARD ===")
        await scraper.navigate_to_leaderboard(competition)
        leaderboard_data = await scraper.get_current_page_data()
        results['leaderboard'] = leaderboard_data

        # Scrape submissions
        print("\n=== SUBMISSIONS ===")
        await scraper.navigate_to_submissions(competition)
        submissions_data = await scraper.get_current_page_data()
        results['submissions'] = submissions_data

        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save full data
            full_path = output_dir / f"kaggle_scrape_{timestamp}.json"
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Full data saved to: {full_path}")

            # Save leaderboard snapshot
            lb_path = output_dir / f"leaderboard_{timestamp}.json"
            with open(lb_path, 'w', encoding='utf-8') as f:
                json.dump(leaderboard_data['snapshot'], f, indent=2, ensure_ascii=False)
            print(f"✓ Leaderboard snapshot: {lb_path}")

            # Save submissions snapshot
            sub_path = output_dir / f"submissions_{timestamp}.json"
            with open(sub_path, 'w', encoding='utf-8') as f:
                json.dump(submissions_data['snapshot'], f, indent=2, ensure_ascii=False)
            print(f"✓ Submissions snapshot: {sub_path}")

        return results

    finally:
        await scraper.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape Kaggle competition data via CDP"
    )
    parser.add_argument(
        'competition',
        help='Competition name (e.g., playground-series-s5e11)'
    )
    parser.add_argument(
        '--cdp-url',
        default='http://localhost:9222',
        help='CDP connection URL (default: http://localhost:9222)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory (default: competition/data/kaggle_scrapes/)'
    )
    parser.add_argument(
        '--project-root',
        help='Project root directory (for auto output path)'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.project_root:
        output_dir = Path(args.project_root) / "data" / "kaggle_scrapes"
    else:
        # Default: current directory
        script_dir = Path(__file__).parent.parent
        comp_dir = script_dir / args.competition
        if comp_dir.exists():
            output_dir = comp_dir / "data" / "kaggle_scrapes"
        else:
            output_dir = Path.cwd() / "kaggle_scrapes"

    print(f"Competition: {args.competition}")
    print(f"Output directory: {output_dir}")
    print(f"CDP URL: {args.cdp_url}")
    print()

    await scrape_competition(
        competition=args.competition,
        cdp_url=args.cdp_url,
        output_dir=output_dir
    )

    print("\n✓ Scraping complete!")
    print("\nUsage:")
    print("1. Review scraped data in JSON files")
    print("2. Update submissions tracker with public/private scores:")
    print(f"   python tools/submissions_tracker.py --project {args.competition} update <ID> --public <SCORE>")


if __name__ == "__main__":
    asyncio.run(main())
