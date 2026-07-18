#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Release Notes Generator for flagos-ai repositories.

This tool generates standardized release notes by analyzing merged PRs
between two git references (tags, branches, or commits).

Usage:
    python generate_release_notes.py --repo flagos-ai/FlagScale --from v0.9.0 --to v1.0.0

The output follows mainstream open-source community standards (PyTorch, vLLM, Triton).
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import yaml, use fallback if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class PRInfo:
    """Represents a merged pull request."""
    number: int
    title: str
    author: str
    merged_at: str
    labels: List[str] = field(default_factory=list)
    url: str = ""

    @property
    def clean_title(self) -> str:
        """Remove prefix tags like [Fix], [Model] from title."""
        return re.sub(r'^\[[\w\-]+\]\s*', '', self.title).strip()


@dataclass
class CategoryConfig:
    """Configuration for PR classification."""
    name: str
    prefixes: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    order: int = 0


# Default classification configuration
DEFAULT_CATEGORIES = [
    CategoryConfig("Highlights", prefixes=["highlight", "Highlight"], labels=["highlight"], order=0),
    CategoryConfig("Breaking Changes", prefixes=["Breaking", "breaking"], labels=["breaking-change"], order=1),
    CategoryConfig("New Features", prefixes=["Feature", "feature", "Model", "model"], labels=["enhancement"], order=2),
    CategoryConfig("Bug Fixes", prefixes=["Fix", "fix", "BugFix", "bugfix", "Bugfix"], labels=["bug"], order=3),
    CategoryConfig("Performance", prefixes=["Perf", "perf", "Performance", "Optimization"], labels=["performance"], order=4),
    CategoryConfig("Improvements", prefixes=["Refactor", "refactor", "Improve", "improve"], labels=["improvement"], order=5),
    CategoryConfig("Documentation", prefixes=["Doc", "Docs", "doc", "docs", "Documentation"], labels=["documentation"], order=6),
    CategoryConfig("CI/Infrastructure", prefixes=["CI", "CICD", "ci", "cicd", "Infra", "infra"], labels=["ci"], order=7),
    CategoryConfig("Testing", prefixes=["Test", "test", "Benchmark", "benchmark"], labels=["test"], order=8),
    CategoryConfig("Dependencies", prefixes=["Dep", "deps", "Dependencies"], labels=["dependencies"], order=9),
    CategoryConfig("Hardware Support", prefixes=["MLU", "CUDA", "NPU", "ROCm", "Hardware"], order=10),
    CategoryConfig("Other", order=99),
]

# Hardware/platform prefixes that should go to Hardware Support
HARDWARE_PREFIXES = ["MLU", "CUDA", "NPU", "ROCm", "GPU", "CPU", "TPU"]


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML or JSON file."""
    if not config_path or not Path(config_path).exists():
        return {}

    # Try JSON first (no dependency required)
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)

    # For YAML files, check if PyYAML is available
    if not YAML_AVAILABLE:
        print(f"Warning: PyYAML not installed. Cannot load YAML config: {config_path}", file=sys.stderr)
        print("Install PyYAML: pip install pyyaml", file=sys.stderr)
        print("Or use JSON config file instead.", file=sys.stderr)
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def check_gh_cli() -> bool:
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_gh_command(args: List[str]) -> str:
    """Run gh CLI command and return output."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def get_tag_date(repo: str, tag: str) -> str:
    """Get the date when a tag was created."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/git/refs/tags/{tag}"],
            capture_output=True,
            text=True,
            check=True
        )
        tag_info = json.loads(result.stdout)

        # Get the object info (could be tag or commit)
        obj_info = tag_info.get("object", {})
        obj_url = obj_info.get("url", "")
        obj_type = obj_info.get("type", "")

        if obj_url:
            result = subprocess.run(
                ["gh", "api", obj_url],
                capture_output=True,
                text=True,
                check=True
            )
            obj_data = json.loads(result.stdout)

            # Annotated tag: use tagger.date
            # Lightweight tag (commit): use committer.date
            if obj_type == "tag":
                return obj_data.get("tagger", {}).get("date", "")
            else:
                return obj_data.get("committer", {}).get("date", "")
    except subprocess.CalledProcessError:
        pass
    return ""


def get_repo_created_at(repo: str) -> str:
    """Get the creation date of the repository."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}"],
            capture_output=True,
            text=True,
            check=True
        )
        repo_info = json.loads(result.stdout)
        return repo_info.get("created_at", "")
    except subprocess.CalledProcessError:
        pass
    return ""


def get_merged_prs_since_creation(repo: str, to_tag: str) -> List[PRInfo]:
    """Get all merged PRs from repo creation to a specific tag (first release)."""
    # Get repo creation date and tag date
    from_date = get_repo_created_at(repo)
    to_date = get_tag_date(repo, to_tag)

    if not from_date:
        print(f"Warning: Could not get repo creation date.", file=sys.stderr)
        return []

    if not to_date:
        print(f"Warning: Could not get date for tag {to_tag}.", file=sys.stderr)
        return []

    print(f"Repository created: {from_date}", file=sys.stderr)
    print(f"Tag {to_tag} date: {to_date}", file=sys.stderr)

    # Use gh pr list with search query
    search_query = f"merged:>={from_date} merged:<{to_date}"

    try:
        result = run_gh_command([
            "pr", "list",
            "--repo", repo,
            "--state", "merged",
            "--search", search_query,
            "--json", "number,title,author,mergedAt,labels,url",
            "--limit", "1000"
        ])
    except SystemExit:
        print("Failed to fetch PRs.", file=sys.stderr)
        return []

    prs = []
    for item in json.loads(result):
        prs.append(PRInfo(
            number=item["number"],
            title=item["title"],
            author=item.get("author", {}).get("login", "unknown"),
            merged_at=item.get("mergedAt", ""),
            labels=[label.get("name", "") for label in item.get("labels", [])],
            url=item.get("url", f"https://github.com/{repo}/pull/{item['number']}")
        ))

    # Sort by merged date
    prs.sort(key=lambda x: x.merged_at)
    return prs


def get_merged_prs_between(repo: str, from_tag: str, to_tag: str) -> List[PRInfo]:
    """Get merged PRs between two tags using gh CLI."""
    # Use gh pr list with date range
    # First get the tag dates
    from_date = get_tag_date(repo, from_tag)
    to_date = get_tag_date(repo, to_tag)

    if not from_date or not to_date:
        print(f"Warning: Could not get dates for tags. Falling back to commit method.", file=sys.stderr)
        return get_merged_prs_by_commits(repo, from_tag, to_tag)

    # Use gh pr list with search query
    search_query = f"merged:>={from_date} merged:<{to_date}"

    try:
        result = run_gh_command([
            "pr", "list",
            "--repo", repo,
            "--state", "merged",
            "--search", search_query,
            "--json", "number,title,author,mergedAt,labels,url",
            "--limit", "500"
        ])
    except SystemExit:
        # Fallback to commit method if search fails
        print("Falling back to commit comparison method...", file=sys.stderr)
        return get_merged_prs_by_commits(repo, from_tag, to_tag)

    prs = []
    for item in json.loads(result):
        prs.append(PRInfo(
            number=item["number"],
            title=item["title"],
            author=item.get("author", {}).get("login", "unknown"),
            merged_at=item.get("mergedAt", ""),
            labels=[label.get("name", "") for label in item.get("labels", [])],
            url=item.get("url", f"https://github.com/{repo}/pull/{item['number']}")
        ))

    # Sort by merged date
    prs.sort(key=lambda x: x.merged_at)
    return prs


def get_merged_prs_by_commits(repo: str, from_ref: str, to_ref: str) -> List[PRInfo]:
    """Get merged PRs by comparing commits (alternative method)."""
    # Get commits between two refs
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/compare/{from_ref}...{to_ref}"],
            capture_output=True,
            text=True,
            check=True
        )
        compare_data = json.loads(result.stdout)
    except subprocess.CalledProcessError:
        print(f"Warning: Could not compare refs {from_ref}...{to_ref}", file=sys.stderr)
        return []

    # Extract PR numbers from commit messages
    pr_numbers = set()
    commits = compare_data.get("commits", [])
    for commit in commits:
        message = commit.get("commit", {}).get("message", "")
        # Look for PR number in merge commit message (e.g., " (#123)")
        match = re.search(r'\(#(\d+)\)$', message)
        if match:
            pr_numbers.add(int(match.group(1)))

    # Fetch PR details
    prs = []
    for num in pr_numbers:
        try:
            result = subprocess.run(
                ["gh", "pr", "view", str(num), "--repo", repo,
                 "--json", "number,title,author,mergedAt,labels,url"],
                capture_output=True,
                text=True,
                check=True
            )
            pr_data = json.loads(result.stdout)
            prs.append(PRInfo(
                number=pr_data["number"],
                title=pr_data["title"],
                author=pr_data.get("author", {}).get("login", "unknown"),
                merged_at=pr_data.get("mergedAt", ""),
                labels=[label.get("name", "") for label in pr_data.get("labels", [])],
                url=pr_data.get("url", f"https://github.com/{repo}/pull/{pr_data['number']}")
            ))
        except subprocess.CalledProcessError:
            continue

    prs.sort(key=lambda x: x.merged_at)
    return prs


def classify_pr(pr: PRInfo, categories: List[CategoryConfig]) -> str:
    """Classify a PR into a category based on title prefix or labels."""
    # Extract prefix from title (e.g., [Fix], [Model], [CI/CD])
    # Allow alphanumeric, slash, hyphen, underscore, and space in prefix
    prefix_match = re.match(r'^\[([\w\-/\s]+)\]', pr.title)
    title_prefix = prefix_match.group(1) if prefix_match else ""

    # Normalize prefix: remove spaces, convert to lowercase for comparison
    normalized_prefix = title_prefix.strip().lower().replace(" ", "")

    # Check each category
    for cat in categories:
        # Check prefix match (case-insensitive, ignoring spaces)
        if title_prefix:
            for prefix in cat.prefixes:
                normalized_cat_prefix = prefix.strip().lower().replace(" ", "")
                if normalized_prefix == normalized_cat_prefix or normalized_prefix.startswith(normalized_cat_prefix):
                    return cat.name

        # Check label match
        for label in pr.labels:
            for cat_label in cat.labels:
                if label.lower() == cat_label.lower():
                    return cat.name

    # Check for hardware-specific prefixes in the title
    for hw_prefix in HARDWARE_PREFIXES:
        if normalized_prefix == hw_prefix.lower() or normalized_prefix.startswith(hw_prefix.lower()):
            return "Hardware Support"

    return "Other"


def generate_markdown(
    prs: List[PRInfo],
    categories: List[CategoryConfig],
    version: str,
    repo: str,
    from_version: str = "",
    include_contributors: bool = True,
) -> str:
    """Generate Markdown release notes."""
    lines = []

    # Title with version range info
    lines.append(f"# Release {version}")
    lines.append("")
    if from_version:
        lines.append(f"**Changes since {from_version}**")
    else:
        lines.append("**Initial Release**")
    lines.append("")

    # Classify PRs
    prs_by_category: Dict[str, List[PRInfo]] = defaultdict(list)
    for pr in prs:
        category = classify_pr(pr, categories)
        prs_by_category[category].append(pr)

    # Track contributors
    contributor_pr_counts: Dict[str, int] = defaultdict(int)
    first_time_contributors = set()

    # Get all previous contributors to repo (for first-time detection)
    # For simplicity, we'll consider all contributors in this release
    # Real first-time detection would require historical data
    all_contributors = set(pr.author for pr in prs)

    for pr in prs:
        contributor_pr_counts[pr.author] += 1

    # Sort categories by order
    sorted_categories = sorted(categories, key=lambda x: x.order)

    # Generate sections
    for cat in sorted_categories:
        cat_prs = prs_by_category.get(cat.name, [])
        if not cat_prs:
            continue

        lines.append(f"## {cat.name}")
        lines.append("")

        for pr in cat_prs:
            # Format: - Description ([#PR](url)) by @author
            description = pr.clean_title
            # Capitalize first letter
            if description and description[0].islower():
                description = description[0].upper() + description[1:]

            pr_link = f"https://github.com/{repo}/pull/{pr.number}"
            lines.append(f"- {description} ([#{pr.number}]({pr_link})) by @{pr.author}")

        lines.append("")

    # Contributors section
    if include_contributors and all_contributors:
        lines.append("## Contributors")
        lines.append("")
        lines.append("Thanks to all contributors who made this release possible:")
        lines.append("")

        # Sort contributors by PR count (descending), then alphabetically
        sorted_contributors = sorted(
            contributor_pr_counts.items(),
            key=lambda x: (-x[1], x[0])
        )

        for author, count in sorted_contributors:
            if count == 1:
                lines.append(f"- @{author}")
            else:
                lines.append(f"- @{author} ({count} PRs)")

        lines.append("")

    # Footer with generation info
    lines.append("---")
    lines.append(f"*Generated by [Release Notes Generator](https://github.com/flagos-ai/FlagRelease/tree/main/tools)*")

    return "\n".join(lines)


def main():
    # Check GitHub CLI availability
    if not check_gh_cli():
        print("Error: GitHub CLI (gh) is not installed or not authenticated.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Please install GitHub CLI:", file=sys.stderr)
        print("  macOS: brew install gh", file=sys.stderr)
        print("  Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md", file=sys.stderr)
        print("  Windows: winget install GitHub.cli", file=sys.stderr)
        print("", file=sys.stderr)
        print("Then authenticate:", file=sys.stderr)
        print("  gh auth login", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate release notes for flagos-ai repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate release notes between two tags
  python generate_release_notes.py --repo flagos-ai/FlagScale --from v0.9.0 --to v1.0.0

  # First release (no previous tag)
  python generate_release_notes.py --repo flagos-ai/FlagScale --to v0.1.0 --first-release

  # Save to a file
  python generate_release_notes.py --repo flagos-ai/FlagScale --from v0.9.0 --to v1.0.0 -o release.md

  # Use custom config
  python generate_release_notes.py --repo flagos-ai/FlagScale --from v0.9.0 --to v1.0.0 --config my_config.yaml

Prerequisites:
  - GitHub CLI (gh) installed and authenticated
  - PyYAML: pip install pyyaml
        """
    )
    parser.add_argument("--repo", required=True, help="Repository in format 'owner/repo'")
    parser.add_argument("--from", dest="from_ref", help="Starting git reference (tag, branch, or commit). Required unless --first-release is set.")
    parser.add_argument("--to", dest="to_ref", required=True, help="Ending git reference (tag, branch, or commit)")
    parser.add_argument("--first-release", action="store_true", help="Use this for the first release (includes all PRs since repo creation)")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--method", choices=["date", "commit"], default="date",
                        help="Method to determine PR range: 'date' uses tag dates, 'commit' uses commit comparison")

    args = parser.parse_args()

    # Validate arguments
    if not args.from_ref and not args.first_release:
        parser.error("--from is required unless --first-release is specified")
    if args.from_ref and args.first_release:
        parser.error("Cannot use both --from and --first-release")

    # Load config if provided
    config = load_config(args.config)

    # Override categories if in config
    categories = DEFAULT_CATEGORIES
    if config.get("categories"):
        categories = [
            CategoryConfig(
                name=cat.get("name", ""),
                prefixes=cat.get("prefixes", []),
                labels=cat.get("labels", []),
                order=cat.get("order", i)
            )
            for i, cat in enumerate(config["categories"])
        ]

    # Get PRs
    if args.first_release:
        print(f"Fetching merged PRs since repo creation up to {args.to_ref}...", file=sys.stderr)
        prs = get_merged_prs_since_creation(args.repo, args.to_ref)
    else:
        print(f"Fetching merged PRs between {args.from_ref} and {args.to_ref}...", file=sys.stderr)
        if args.method == "commit":
            prs = get_merged_prs_by_commits(args.repo, args.from_ref, args.to_ref)
        else:
            prs = get_merged_prs_between(args.repo, args.from_ref, args.to_ref)

    print(f"Found {len(prs)} merged PRs", file=sys.stderr)

    if not prs:
        print("No merged PRs found in the specified range.", file=sys.stderr)
        sys.exit(1)

    # Generate release notes
    version = args.to_ref
    from_version = "" if args.first_release else args.from_ref
    markdown = generate_markdown(prs, categories, version, args.repo, from_version)

    # Output
    if args.output:
        Path(args.output).write_text(markdown)
        print(f"Release notes written to {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
