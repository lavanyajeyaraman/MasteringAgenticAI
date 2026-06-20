from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass(frozen=True)
class SkillNode:
    name: str
    domain: str
    equivalent: tuple[str, ...] = ()
    related: tuple[str, ...] = ()


@dataclass
class SkillGraph:
    nodes: dict[str, SkillNode] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "SkillGraph":
        graph = cls()
        graph.add(
            "Playwright",
            "Browser Automation",
            equivalent=("Selenium", "Cypress", "Puppeteer"),
            related=("API Testing", "Test Framework Design", "End-to-End Testing"),
        )
        graph.add(
            "LangGraph",
            "Agent Orchestration",
            equivalent=("CrewAI", "AutoGen", "LlamaIndex Workflows"),
            related=("LangChain", "Agentic AI", "Workflow Orchestration"),
        )
        graph.add(
            "Kubernetes",
            "Container Orchestration",
            equivalent=("ECS", "GKE", "OpenShift"),
            related=("Helm", "Docker Swarm", "Docker", "Containerization"),
        )
        graph.add(
            "Vertex AI",
            "Cloud AI Platform",
            equivalent=("Azure AI", "SageMaker", "Bedrock"),
            related=("MLOps", "Model Deployment", "Managed ML"),
        )
        graph.add(
            "Python",
            "Programming Language",
            equivalent=("Python 3",),
            related=("FastAPI", "Pytest", "Data Engineering"),
        )
        return graph

    def add(
        self,
        name: str,
        domain: str,
        *,
        equivalent: tuple[str, ...] = (),
        related: tuple[str, ...] = (),
    ) -> None:
        node = SkillNode(name=name, domain=domain, equivalent=equivalent, related=related)
        self.nodes[self._key(name)] = node
        for skill in equivalent:
            self.nodes.setdefault(
                self._key(skill),
                SkillNode(name=skill, domain=domain, equivalent=(name,), related=related),
            )

    def expand(self, skill: str) -> list[str]:
        node = self.nodes.get(self._key(skill))
        if not node:
            return [skill]
        return [node.name, *node.equivalent, *node.related]

    def match(self, jd_skill: str, candidate_skill: str) -> dict[str, object] | None:
        jd_node = self.nodes.get(self._key(jd_skill))
        candidate_node = self.nodes.get(self._key(candidate_skill))
        if self._key(jd_skill) == self._key(candidate_skill):
            return self._result(jd_skill, candidate_skill, "direct", 1.0, jd_node)
        if jd_node and self._key(candidate_skill) in {self._key(s) for s in jd_node.equivalent}:
            return self._result(jd_skill, candidate_skill, "equivalent", 0.92, jd_node)
        if jd_node and self._key(candidate_skill) in {self._key(s) for s in jd_node.related}:
            return self._result(jd_skill, candidate_skill, "related", 0.76, jd_node)
        if jd_node and candidate_node and jd_node.domain == candidate_node.domain:
            return self._result(jd_skill, candidate_skill, "same_domain", 0.82, jd_node)
        fuzzy = SequenceMatcher(None, self._key(jd_skill), self._key(candidate_skill)).ratio()
        if fuzzy >= 0.86:
            return self._result(jd_skill, candidate_skill, "near_text", fuzzy, jd_node)
        return None

    def transferability_score(self, jd_skills: list[str], candidate_skills: list[str]) -> float:
        if not jd_skills:
            return 0.0
        best_scores = []
        for jd_skill in jd_skills:
            matches = [
                self.match(jd_skill, candidate_skill) for candidate_skill in candidate_skills
            ]
            scores = [float(match["confidence"]) for match in matches if match]
            best_scores.append(max(scores, default=0.0))
        return round(sum(best_scores) / len(jd_skills), 3)

    @staticmethod
    def _key(value: str) -> str:
        return value.strip().casefold()

    @staticmethod
    def _result(
        jd_skill: str,
        candidate_skill: str,
        relation: str,
        confidence: float,
        node: SkillNode | None,
    ) -> dict[str, object]:
        return {
            "jd_skill": jd_skill,
            "matched_skill": candidate_skill,
            "relation": relation,
            "confidence": round(confidence, 3),
            "domain": node.domain if node else None,
        }
