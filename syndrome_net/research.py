"""Research integration pipeline for syndrome-net.

Provides infrastructure for tracking QEC research papers, monitoring arXiv,
and integrating new quantum error correction techniques into the codebase.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import re


class ImplementationStatus(Enum):
    """Status of research paper implementation."""
    
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()
    ABANDONED = auto()
    BLOCKED = auto()


@dataclass(frozen=True)
class ResearchPaper:
    """Representation of a QEC research paper.
    
    Attributes:
        title: Paper title
        authors: List of authors
        arxiv_id: arXiv identifier (e.g., "2309.10033")
        doi: Digital Object Identifier
        published: Publication date
        abstract: Paper abstract
        techniques: List of QEC techniques described
        code_relevance: 0-1 score for relevance to syndrome-net
        implementation_status: Current implementation status
        implementation_notes: Notes about implementation
        citations: Number of citations (if known)
        url: Direct URL to paper
    """
    title: str
    authors: list[str]
    arxiv_id: str | None = None
    doi: str | None = None
    published: datetime | None = None
    abstract: str = ""
    techniques: list[str] = field(default_factory=list)
    code_relevance: float = 0.0
    implementation_status: ImplementationStatus = ImplementationStatus.PENDING
    implementation_notes: str = ""
    citations: int | None = None
    url: str | None = None
    
    def __post_init__(self) -> None:
        if not 0 <= self.code_relevance <= 1:
            raise ValueError("code_relevance must be between 0 and 1")


@dataclass
class TechniqueSpec:
    """Specification for implementing a QEC technique.
    
    Links research papers to concrete implementation tasks.
    """
    name: str
    description: str
    papers: list[str]  # arXiv IDs
    estimated_effort: str  # "small", "medium", "large"
    prerequisites: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    implementation_path: str = ""  # Module path where implemented


@runtime_checkable
class ResearchSource(Protocol):
    """Protocol for research paper sources."""
    
    def fetch_recent(self, days: int = 7) -> Iterator[ResearchPaper]:
        """Fetch recent papers from this source."""
        ...
    
    def search(self, query: str, max_results: int = 10) -> list[ResearchPaper]:
        """Search for papers matching query."""
        ...


class ArxivSource:
    """arXiv paper source for QEC research."""
    
    CATEGORIES = ["quant-ph"]
    KEYWORDS = [
        "error correction",
        "surface code",
        "Floquet",
        "LDPC",
        "honeycomb",
        "stabilizer",
        "decoder",
        "threshold",
        "logical qubit",
    ]
    
    def __init__(self, categories: list[str] | None = None) -> None:
        self.categories = categories or self.CATEGORIES
    
    def fetch_recent(self, days: int = 7) -> Iterator[ResearchPaper]:
        """Fetch recent papers from arXiv.
        
        Args:
            days: Number of days to look back
            
        Yields:
            ResearchPaper instances
        """
        try:
            import feedparser
        except ImportError:
            raise ImportError("feedparser required for arXiv monitoring. "
                            "Install with: pip install feedparser")
        
        # Build query
        cat_query = "+OR+".join(f"cat:{cat}" for cat in self.categories)
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={cat_query}&"
            f"sortBy=submittedDate&sortOrder=descending&max_results=100"
        )
        
        feed = feedparser.parse(url)
        
        for entry in feed.entries:
            # Check if paper matches keywords
            text = (entry.title + entry.get("summary", "")).lower()
            if any(kw in text for kw in self.keywords):
                paper = self._parse_entry(entry)
                if paper:
                    yield paper
    
    def search(self, query: str, max_results: int = 10) -> list[ResearchPaper]:
        """Search arXiv for papers matching query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of matching papers
        """
        try:
            import feedparser
        except ImportError:
            raise ImportError("feedparser required for arXiv search")
        
        # Escape query
        query = query.replace(" ", "+")
        
        cat_query = "+OR+".join(f"cat:{cat}" for cat in self.categories)
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=({query})+AND+({cat_query})&"
            f"max_results={max_results}"
        )
        
        feed = feedparser.parse(url)
        papers = []
        
        for entry in feed.entries:
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)
        
        return papers
    
    def _parse_entry(self, entry: Any) -> ResearchPaper | None:
        """Parse a feedparser entry into a ResearchPaper."""
        try:
            # Extract arXiv ID from link or ID field
            arxiv_id = None
            if hasattr(entry, "id"):
                match = re.search(r"(\d+\.\d+)", entry.id)
                if match:
                    arxiv_id = match.group(1)
            
            # Parse authors
            authors = []
            if hasattr(entry, "authors"):
                authors = [a.get("name", "") for a in entry.authors]
            
            # Parse published date
            published = None
            if hasattr(entry, "published_parsed"):
                published = datetime(*entry.published_parsed[:6])
            
            # Extract techniques from abstract
            abstract = entry.get("summary", "")
            techniques = self._extract_techniques(abstract)
            
            # Calculate relevance score
            relevance = self._calculate_relevance(entry.title, abstract, techniques)
            
            return ResearchPaper(
                title=entry.title,
                authors=authors,
                arxiv_id=arxiv_id,
                published=published,
                abstract=abstract,
                techniques=techniques,
                code_relevance=relevance,
                url=entry.get("link")
            )
        except Exception:
            return None
    
    def _extract_techniques(self, abstract: str) -> list[str]:
        """Extract QEC techniques mentioned in abstract."""
        techniques = []
        abstract_lower = abstract.lower()
        
        technique_keywords = {
            "surface code": "surface_code",
            "floquet": "floquet_code",
            "ldpc": "ldpc_code",
            "honeycomb": "honeycomb_code",
            "color code": "color_code",
            "stabilizer": "stabilizer_code",
            "cat code": "cat_code",
            "repetition code": "repetition_code",
            "mwpm": "mwpm_decoder",
            "union-find": "union_find_decoder",
            "belief propagation": "bp_decoder",
            "neural decoder": "neural_decoder",
            "machine learning": "ml_decoder",
            "deep learning": "dl_decoder",
        }
        
        for keyword, technique in technique_keywords.items():
            if keyword in abstract_lower:
                techniques.append(technique)
        
        return techniques
    
    def _calculate_relevance(self, title: str, abstract: str, techniques: list[str]) -> float:
        """Calculate relevance score for syndrome-net."""
        score = 0.0
        text = (title + abstract).lower()
        
        # Base score for having QEC techniques
        score += len(techniques) * 0.1
        
        # Bonus for circuit-level simulation
        if "circuit" in text or "stim" in text or "simulation" in text:
            score += 0.2
        
        # Bonus for threshold estimation
        if "threshold" in text:
            score += 0.15
        
        # Bonus for decoder focus
        if "decoder" in text:
            score += 0.15
        
        # Bonus for RL/decoding
        if "reinforcement" in text or "rl" in text:
            score += 0.1
        
        return min(score, 1.0)


class ResearchTracker:
    """Tracks QEC research papers and their implementation status.
    
    Provides a centralized database of relevant papers with status tracking
    for continual learning and research integration.
    """
    
    def __init__(self, storage_path: str | None = None) -> None:
        """
        Args:
            storage_path: Path to persistent storage (JSON/SQLite)
        """
        self.storage_path = storage_path
        self._papers: dict[str, ResearchPaper] = {}
        self._sources: list[ResearchSource] = []
        
        if storage_path:
            self._load()
    
    def add_source(self, source: ResearchSource) -> None:
        """Add a paper source (arXiv, etc.)."""
        self._sources.append(source)
    
    def add_paper(self, paper: ResearchPaper) -> None:
        """Add a paper to the tracker.
        
        Args:
            paper: Research paper to track
        """
        key = paper.arxiv_id or paper.doi or paper.title
        self._papers[key] = paper
        self._save()
    
    def get_paper(self, arxiv_id: str) -> ResearchPaper | None:
        """Get a paper by arXiv ID."""
        return self._papers.get(arxiv_id)
    
    def update_status(
        self,
        arxiv_id: str,
        status: ImplementationStatus,
        notes: str = ""
    ) -> None:
        """Update implementation status of a paper.
        
        Args:
            arxiv_id: arXiv ID of the paper
            status: New implementation status
            notes: Implementation notes
        """
        if arxiv_id in self._papers:
            paper = self._papers[arxiv_id]
            # Create updated paper (dataclass is frozen)
            updated = ResearchPaper(
                title=paper.title,
                authors=paper.authors,
                arxiv_id=paper.arxiv_id,
                doi=paper.doi,
                published=paper.published,
                abstract=paper.abstract,
                techniques=paper.techniques,
                code_relevance=paper.code_relevance,
                implementation_status=status,
                implementation_notes=notes,
                citations=paper.citations,
                url=paper.url
            )
            self._papers[arxiv_id] = updated
            self._save()
    
    def fetch_new_papers(self, days: int = 7) -> list[ResearchPaper]:
        """Fetch new papers from all sources.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of new papers not already in tracker
        """
        new_papers = []
        
        for source in self._sources:
            for paper in source.fetch_recent(days):
                key = paper.arxiv_id or paper.doi or paper.title
                if key not in self._papers:
                    self._papers[key] = paper
                    new_papers.append(paper)
        
        if new_papers:
            self._save()
        
        return new_papers
    
    def get_pending(self, min_relevance: float = 0.5) -> list[ResearchPaper]:
        """Get pending papers above relevance threshold.
        
        Args:
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of pending papers
        """
        return [
            p for p in self._papers.values()
            if p.implementation_status == ImplementationStatus.PENDING
            and p.code_relevance >= min_relevance
        ]
    
    def suggest_next(self) -> ResearchPaper | None:
        """Suggest the next paper to implement.
        
        Returns the highest-relevance pending paper.
        """
        pending = self.get_pending(min_relevance=0.6)
        if not pending:
            return None
        
        return max(pending, key=lambda p: p.code_relevance)
    
    def list_by_status(self, status: ImplementationStatus) -> list[ResearchPaper]:
        """List all papers with given status."""
        return [p for p in self._papers.values() if p.implementation_status == status]
    
    def list_by_technique(self, technique: str) -> list[ResearchPaper]:
        """List all papers mentioning a technique."""
        return [
            p for p in self._papers.values()
            if technique in p.techniques
        ]
    
    def _save(self) -> None:
        """Save papers to persistent storage."""
        if not self.storage_path:
            return
        
        import json
        
        data = []
        for paper in self._papers.values():
            paper_dict = {
                "title": paper.title,
                "authors": paper.authors,
                "arxiv_id": paper.arxiv_id,
                "doi": paper.doi,
                "published": paper.published.isoformat() if paper.published else None,
                "abstract": paper.abstract,
                "techniques": paper.techniques,
                "code_relevance": paper.code_relevance,
                "implementation_status": paper.implementation_status.name,
                "implementation_notes": paper.implementation_notes,
                "citations": paper.citations,
                "url": paper.url
            }
            data.append(paper_dict)
        
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load papers from persistent storage."""
        import json
        from pathlib import Path
        
        if not Path(self.storage_path).exists():
            return
        
        with open(self.storage_path, "r") as f:
            data = json.load(f)
        
        for paper_dict in data:
            paper = ResearchPaper(
                title=paper_dict["title"],
                authors=paper_dict["authors"],
                arxiv_id=paper_dict.get("arxiv_id"),
                doi=paper_dict.get("doi"),
                published=datetime.fromisoformat(paper_dict["published"]) if paper_dict.get("published") else None,
                abstract=paper_dict.get("abstract", ""),
                techniques=paper_dict.get("techniques", []),
                code_relevance=paper_dict.get("code_relevance", 0.0),
                implementation_status=ImplementationStatus[paper_dict.get("implementation_status", "PENDING")],
                implementation_notes=paper_dict.get("implementation_notes", ""),
                citations=paper_dict.get("citations"),
                url=paper_dict.get("url")
            )
            key = paper.arxiv_id or paper.doi or paper.title
            self._papers[key] = paper


class ImplementationPlanner:
    """Plans implementation of research techniques.
    
    Converts research papers into actionable implementation tasks.
    """
    
    def __init__(self, tracker: ResearchTracker) -> None:
        self.tracker = tracker
    
    def create_plan(self, paper: ResearchPaper) -> list[TechniqueSpec]:
        """Create implementation plan from a research paper.
        
        Args:
            paper: Research paper to implement
            
        Returns:
            List of implementation specifications
        """
        specs = []
        
        for technique in paper.techniques:
            spec = self._technique_to_spec(technique, paper)
            if spec:
                specs.append(spec)
        
        return specs
    
    def _technique_to_spec(self, technique: str, paper: ResearchPaper) -> TechniqueSpec | None:
        """Convert technique name to implementation spec."""
        # Map technique names to implementation details
        technique_map = {
            "floquet_code": TechniqueSpec(
                name="FloquetCodeBuilder",
                description="Floquet topological code with dynamic stabilizers",
                papers=[paper.arxiv_id] if paper.arxiv_id else [],
                estimated_effort="large",
                prerequisites=["stabilizer_cycle", "dynamic_scheduling"],
                implementation_path="syndrome_net.codes.floquet"
            ),
            "ldpc_code": TechniqueSpec(
                name="LDPCCodeBuilder",
                description="Low-density parity check quantum code",
                papers=[paper.arxiv_id] if paper.arxiv_id else [],
                estimated_effort="large",
                prerequisites=["tanner_graph", "bp_decoder"],
                implementation_path="syndrome_net.codes.ldpc"
            ),
            "mwpm_decoder": TechniqueSpec(
                name="MWPMDecoder",
                description="Minimum weight perfect matching decoder",
                papers=[paper.arxiv_id] if paper.arxiv_id else [],
                estimated_effort="medium",
                prerequisites=["matching_graph", "blossom_v"],
                implementation_path="syndrome_net.decoders.mwpm"
            ),
            "neural_decoder": TechniqueSpec(
                name="NeuralDecoder",
                description="Neural network-based syndrome decoder",
                papers=[paper.arxiv_id] if paper.arxiv_id else [],
                estimated_effort="large",
                prerequisites=["pytorch", "training_pipeline"],
                implementation_path="syndrome_net.decoders.neural"
            ),
        }
        
        return technique_map.get(technique)


def create_default_tracker(storage_path: str = "research_tracker.json") -> ResearchTracker:
    """Create a research tracker with arXiv source.
    
    Args:
        storage_path: Path to persistent storage
        
    Returns:
        Configured ResearchTracker
    """
    tracker = ResearchTracker(storage_path)
    tracker.add_source(ArxivSource())
    return tracker
