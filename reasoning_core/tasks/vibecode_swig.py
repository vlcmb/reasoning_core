"""
Experimental SWIG object.

This file is intentionally separate from ``swig.py``. It sketches a richer
Single-World Intervention Graph representation that remains a
``DiscreteBayesianNetwork`` while carrying enough metadata to avoid confusing
fixed intervention values with ordinary random variables.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


Node = str


@dataclass(frozen=True)
class SwigNodeInfo:
    """Semantic information attached to one SWIG node."""

    label: Node
    source: Node
    kind: str
    value: Optional[Any] = None
    fixed_ancestors: Tuple[Node, ...] = ()

    @property
    def is_fixed(self) -> bool:
        return self.kind == "fixed"

    @property
    def is_random(self) -> bool:
        return self.kind == "random"


@dataclass
class SwigSpec:
    """Metadata that vanilla Bayesian-network serialization would lose."""

    source_nodes: Tuple[Node, ...] = ()
    source_edges: Tuple[Tuple[Node, Node], ...] = ()
    interventions: Dict[Node, Any] = field(default_factory=dict)
    random_of: Dict[Node, Node] = field(default_factory=dict)
    fixed_of: Dict[Node, Node] = field(default_factory=dict)
    source_of: Dict[Node, Node] = field(default_factory=dict)
    node_info: Dict[Node, SwigNodeInfo] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["node_info"] = {
            label: asdict(info) for label, info in self.node_info.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SwigSpec":
        raw_info = data.get("node_info", {})
        node_info = {
            str(label): SwigNodeInfo(
                label=str(info["label"]),
                source=str(info["source"]),
                kind=str(info["kind"]),
                value=info.get("value"),
                fixed_ancestors=tuple(info.get("fixed_ancestors", ())),
            )
            for label, info in raw_info.items()
        }
        return cls(
            source_nodes=tuple(map(str, data.get("source_nodes", ()))),
            source_edges=tuple(
                (str(u), str(v)) for u, v in data.get("source_edges", ())
            ),
            interventions={str(k): v for k, v in data.get("interventions", {}).items()},
            random_of={str(k): str(v) for k, v in data.get("random_of", {}).items()},
            fixed_of={str(k): str(v) for k, v in data.get("fixed_of", {}).items()},
            source_of={str(k): str(v) for k, v in data.get("source_of", {}).items()},
            node_info=node_info,
        )


class VibecodeSwig(DiscreteBayesianNetwork):
    """
    A SWIG as a Bayesian-network-shaped object with explicit semantics.

    The object has two kinds of nodes:
    - random nodes: stochastic variables, possibly indexed by fixed ancestors;
    - fixed nodes: intervention-value nodes represented as deterministic
      point-mass variables so pgmpy graph/CPD machinery can still be reused.

    The important design point is that this class does not rely only on labels
    such as ``Y(X=1)``. Labels are readable, while ``self.swig`` carries the
    authoritative mapping back to the source causal Bayesian network.
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[Tuple[Node, Node]]] = None,
        *,
        source_bn: Optional[DiscreteBayesianNetwork] = None,
        spec: Optional[SwigSpec] = None,
    ):
        super().__init__(ebunch)
        self.source_bn = source_bn.copy() if source_bn is not None else None
        self.swig = spec if spec is not None else SwigSpec()
        self.trace: List[str] = []

    @classmethod
    def from_bn(
        cls,
        bn: DiscreteBayesianNetwork,
        interventions: Mapping[Any, Any],
        *,
        copy_cpds: bool = True,
    ) -> "VibecodeSwig":
        """
        Build the SWIG induced by a source causal BN and intervention values.

        ``interventions`` maps source variables to fixed values. The split
        random half of an intervened variable remains available for factual
        evidence; the fixed half sends outgoing causal influence downstream.
        """

        source_nodes = tuple(str(n) for n in bn.nodes())
        source_edges = tuple((str(u), str(v)) for u, v in bn.edges())
        normalized_interventions = {
            str(var): value for var, value in interventions.items()
        }

        missing = set(normalized_interventions) - set(source_nodes)
        if missing:
            raise ValueError(f"Unknown intervention variables: {sorted(missing)}")

        fixed_of = {
            var: cls._fixed_label(var, value)
            for var, value in normalized_interventions.items()
        }

        temp_graph = nx.DiGraph()
        temp_graph.add_nodes_from(source_nodes)
        temp_graph.add_nodes_from(fixed_of.values())

        for parent, child in source_edges:
            swig_parent = fixed_of[parent] if parent in fixed_of else parent
            temp_graph.add_edge(swig_parent, child)

        fixed_labels = set(fixed_of.values())
        fixed_var_by_label = {label: var for var, label in fixed_of.items()}

        random_of: Dict[Node, Node] = {}
        node_info: Dict[Node, SwigNodeInfo] = {}
        source_of: Dict[Node, Node] = {}

        for source_var in source_nodes:
            fixed_ancestor_labels = sorted(
                nx.ancestors(temp_graph, source_var).intersection(fixed_labels)
            )
            fixed_ancestor_vars = tuple(
                fixed_var_by_label[label] for label in fixed_ancestor_labels
            )
            label = cls._random_label(
                source_var,
                fixed_ancestor_vars,
                normalized_interventions,
            )
            random_of[source_var] = label
            source_of[label] = source_var
            node_info[label] = SwigNodeInfo(
                label=label,
                source=source_var,
                kind="random",
                fixed_ancestors=fixed_ancestor_vars,
            )

        for source_var, label in fixed_of.items():
            source_of[label] = source_var
            node_info[label] = SwigNodeInfo(
                label=label,
                source=source_var,
                kind="fixed",
                value=normalized_interventions[source_var],
            )

        expected_node_count = len(source_nodes) + len(fixed_of)
        if len(node_info) != expected_node_count:
            raise ValueError(
                "SWIG labels are not unique. Use a less ambiguous fixed/random "
                "label scheme before building this graph."
            )

        spec = SwigSpec(
            source_nodes=source_nodes,
            source_edges=source_edges,
            interventions=normalized_interventions,
            random_of=random_of,
            fixed_of=fixed_of,
            source_of=source_of,
            node_info=node_info,
        )

        swig = cls(source_bn=bn, spec=spec)
        swig.add_nodes_from(node_info)

        for parent, child in source_edges:
            swig_parent = fixed_of[parent] if parent in fixed_of else random_of[parent]
            swig_child = random_of[child]
            swig.add_edge(swig_parent, swig_child)

        if copy_cpds:
            swig.attach_source_mechanisms()

        swig.trace.append(
            "Built SWIG from source BN with interventions "
            f"{normalized_interventions}."
        )
        return swig

    @classmethod
    def get_random(
        cls,
        *,
        n_nodes: int = 4,
        edge_prob: float = 0.5,
        n_states: Any = 2,
        n_interventions: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "VibecodeSwig":
        """
        Generate a random source BN with pgmpy, then apply a random SWIG split.

        This deliberately leverages the superclass random-BN machinery instead
        of generating arbitrary SWIG-looking graphs.
        """

        rng = random.Random(seed)
        source_bn = DiscreteBayesianNetwork.get_random(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            n_states=n_states,
            seed=seed,
            **kwargs,
        )
        variables = [str(v) for v in source_bn.nodes()]
        chosen = rng.sample(variables, min(n_interventions, len(variables)))
        interventions = {
            var: rng.choice(cls._states_for(source_bn, var)) for var in chosen
        }
        return cls.from_bn(source_bn, interventions)

    def attach_source_mechanisms(self) -> None:
        """
        Attach CPDs induced by the source BN to the SWIG nodes.

        Fixed intervention nodes receive deterministic point-mass CPDs over the
        source variable's state space. Random nodes receive copies of the source
        mechanisms, with parent names redirected to either random or fixed SWIG
        nodes according to the node split.
        """

        if self.source_bn is None:
            raise ValueError("Cannot attach CPDs without a source BN.")

        cpds = []
        for source_var, fixed_label in self.swig.fixed_of.items():
            states = self._states_for(self.source_bn, source_var)
            cpds.append(
                self._point_mass_cpd(
                    fixed_label,
                    states,
                    self.swig.interventions[source_var],
                )
            )

        for source_var in self.swig.source_nodes:
            source_cpd = self.source_bn.get_cpds(self._source_key(source_var))
            if source_cpd is None:
                continue
            if not isinstance(source_cpd, TabularCPD):
                raise TypeError(
                    "VibecodeSwig currently copies TabularCPD mechanisms only; "
                    f"got {type(source_cpd)!r} for {source_var}."
                )
            cpds.append(self._redirect_cpd(source_cpd))

        self.add_cpds(*cpds)

    def _redirect_cpd(self, cpd: TabularCPD) -> TabularCPD:
        source_var = str(cpd.variable)
        swig_var = self.swig.random_of[source_var]

        raw_evidence = list(cpd.variables[1:])
        source_evidence = [str(v) for v in raw_evidence]
        swig_evidence = [
            self.swig.fixed_of[p] if p in self.swig.fixed_of else self.swig.random_of[p]
            for p in source_evidence
        ]
        evidence_card = [int(card) for card in cpd.cardinality[1:]]

        state_names = {
            swig_var: self._cpd_states(cpd, cpd.variable, int(cpd.variable_card))
        }
        for raw_parent, swig_parent, parent_card in zip(
            raw_evidence,
            swig_evidence,
            evidence_card,
        ):
            state_names[swig_parent] = self._cpd_states(
                cpd,
                raw_parent,
                parent_card,
            )

        kwargs: Dict[str, Any] = {
            "variable": swig_var,
            "variable_card": int(cpd.variable_card),
            "values": np.array(cpd.get_values(), dtype=float, copy=True),
            "state_names": state_names,
        }
        if swig_evidence:
            kwargs["evidence"] = swig_evidence
            kwargs["evidence_card"] = evidence_card
        return TabularCPD(**kwargs)

    @staticmethod
    def _cpd_states(cpd: TabularCPD, variable: Any, cardinality: int) -> List[Any]:
        if variable in cpd.state_names:
            return list(cpd.state_names[variable])
        variable_as_str = str(variable)
        if variable_as_str in cpd.state_names:
            return list(cpd.state_names[variable_as_str])
        return list(range(cardinality))

    @staticmethod
    def _point_mass_cpd(label: Node, states: List[Any], value: Any) -> TabularCPD:
        if value not in states:
            raise ValueError(f"Intervention value {value!r} is not in states {states!r}")
        values = np.zeros((len(states), 1))
        values[states.index(value), 0] = 1.0
        return TabularCPD(
            variable=label,
            variable_card=len(states),
            values=values,
            state_names={label: states},
        )

    @staticmethod
    def _states_for(bn: DiscreteBayesianNetwork, variable: Node) -> List[Any]:
        bn_key = VibecodeSwig._bn_node_key(bn, variable)
        if hasattr(bn, "states"):
            if bn_key in bn.states:
                return list(bn.states[bn_key])
            if str(variable) in bn.states:
                return list(bn.states[str(variable)])

        cpd = bn.get_cpds(bn_key)
        if cpd is not None and hasattr(cpd, "state_names"):
            if bn_key in cpd.state_names:
                return list(cpd.state_names[bn_key])
            if str(variable) in cpd.state_names:
                return list(cpd.state_names[str(variable)])

        try:
            return list(range(int(bn.get_cardinality(bn_key))))
        except Exception as exc:
            raise ValueError(f"Cannot infer states for variable {variable!r}") from exc

    @staticmethod
    def _bn_node_key(bn: DiscreteBayesianNetwork, variable: Any) -> Any:
        if variable in bn.nodes():
            return variable
        variable_as_str = str(variable)
        for node in bn.nodes():
            if str(node) == variable_as_str:
                return node
        return variable

    def _source_key(self, source_var: Any) -> Any:
        if self.source_bn is None:
            return source_var
        return self._bn_node_key(self.source_bn, source_var)

    @staticmethod
    def _fixed_label(variable: Node, value: Any) -> Node:
        return f"do({variable}={value})"

    @staticmethod
    def _random_label(
        variable: Node,
        fixed_ancestors: Tuple[Node, ...],
        interventions: Mapping[Node, Any],
    ) -> Node:
        if not fixed_ancestors:
            return variable
        suffix = ",".join(
            f"{ancestor}={interventions[ancestor]}"
            for ancestor in sorted(fixed_ancestors)
        )
        return f"{variable}({suffix})"

    def random_node_for(self, source_var: Any) -> Node:
        return self.swig.random_of[str(source_var)]

    def fixed_node_for(self, source_var: Any) -> Node:
        return self.swig.fixed_of[str(source_var)]

    def source_variable(self, swig_node: Any) -> Node:
        return self.swig.source_of[str(swig_node)]

    def node_info(self, swig_node: Any) -> SwigNodeInfo:
        return self.swig.node_info[str(swig_node)]

    def is_fixed_node(self, swig_node: Any) -> bool:
        return self.node_info(swig_node).is_fixed

    def is_random_node(self, swig_node: Any) -> bool:
        return self.node_info(swig_node).is_random

    def counterfactual_nodes(self) -> List[Node]:
        return sorted(
            label
            for label, info in self.swig.node_info.items()
            if info.is_random and info.fixed_ancestors
        )

    def validate_swig_metadata(self) -> None:
        """Check the SWIG-specific invariants without requiring CPDs."""

        graph_nodes = set(map(str, self.nodes()))
        metadata_nodes = set(self.swig.node_info)
        if graph_nodes != metadata_nodes:
            raise ValueError(
                "SWIG metadata/node mismatch: "
                f"graph_only={sorted(graph_nodes - metadata_nodes)}, "
                f"metadata_only={sorted(metadata_nodes - graph_nodes)}"
            )

        for source_var in self.swig.source_nodes:
            if source_var not in self.swig.random_of:
                raise ValueError(f"Missing random node for source variable {source_var}")

        for source_var in self.swig.interventions:
            if source_var not in self.swig.fixed_of:
                raise ValueError(f"Missing fixed node for intervention on {source_var}")

    def to_nl(self) -> str:
        """Human-readable description that keeps SWIG semantics explicit."""

        lines = ["Single-World Intervention Graph."]
        if self.swig.interventions:
            intervention_text = ", ".join(
                f"{var} set to {value!r}"
                for var, value in sorted(self.swig.interventions.items())
            )
            lines.append(f"Interventions: {intervention_text}.")
        else:
            lines.append("Interventions: none.")

        fixed = [
            info for info in self.swig.node_info.values() if info.kind == "fixed"
        ]
        random_nodes = [
            info for info in self.swig.node_info.values() if info.kind == "random"
        ]

        if fixed:
            lines.append("Fixed nodes:")
            for info in sorted(fixed, key=lambda x: x.label):
                lines.append(
                    f"- {info.label}: fixed value for source variable {info.source}."
                )

        lines.append("Random nodes:")
        for info in sorted(random_nodes, key=lambda x: x.label):
            if info.fixed_ancestors:
                ancestors = ", ".join(info.fixed_ancestors)
                lines.append(
                    f"- {info.label}: source variable {info.source} under "
                    f"fixed ancestors {ancestors}."
                )
            else:
                lines.append(f"- {info.label}: source variable {info.source}.")

        lines.append("Edges:")
        for parent, child in sorted(self.edges()):
            lines.append(f"- {parent} -> {child}")
        return "\n".join(lines)

    def to_serializable(self) -> Dict[str, Any]:
        """
        Serialize the SWIG-specific part.

        This is not a replacement for a BIF writer. It is the metadata payload a
        SWIG-aware BIF/JSON writer should preserve next to the graph and CPDs.
        """

        return {
            "spec": self.swig.to_dict(),
            "nodes": list(self.nodes()),
            "edges": list(self.edges()),
            "trace": list(self.trace),
        }

    @classmethod
    def from_serializable(
        cls,
        data: Mapping[str, Any],
        *,
        source_bn: Optional[DiscreteBayesianNetwork] = None,
    ) -> "VibecodeSwig":
        spec = SwigSpec.from_dict(data["spec"])
        swig = cls(source_bn=source_bn, spec=spec)
        swig.add_nodes_from(data.get("nodes", ()))
        swig.add_edges_from(data.get("edges", ()))
        swig.trace = list(data.get("trace", ()))
        return swig

    def copy(self) -> "VibecodeSwig":
        copied = VibecodeSwig(
            source_bn=self.source_bn,
            spec=SwigSpec.from_dict(self.swig.to_dict()),
        )
        copied.add_nodes_from(self.nodes())
        copied.add_edges_from(self.edges())
        copied.trace = list(self.trace)
        for cpd in self.get_cpds():
            copied.add_cpds(cpd.copy())
        return copied


@dataclass(frozen=True)
class _ResponseMechanism:
    source: Node
    source_key: Any
    cpd: TabularCPD
    parents: Tuple[Node, ...]
    parent_keys: Tuple[Any, ...]
    states: Tuple[Any, ...]
    parent_states: Tuple[Tuple[Any, ...], ...]
    parent_configs: Tuple[Tuple[Any, ...], ...]
    probabilities: Dict[Tuple[Any, ...], Tuple[float, ...]]


class SwigCounterfactualEngine:
    """
    Exact counterfactual engine for small discrete SWIGs.

    Important semantic assumption:
    a Bayesian network CPD does not uniquely determine every counterfactual
    response. This engine lifts each CPD to a canonical Markovian SCM where each
    node has a latent response table. For every parent configuration, the table
    entry is independently distributed according to the corresponding CPD
    column. The resulting SCM matches the source BN observational distribution,
    but it is still an explicit modeling choice.

    Under that choice, queries are evaluated by exact enumeration of response
    tables:
    - abduction: keep only response tables compatible with factual evidence;
    - action: evaluate the same response tables under the SWIG intervention;
    - prediction: accumulate the target variable values.
    """

    def __init__(
        self,
        swig: VibecodeSwig,
        *,
        max_response_functions: int = 200_000,
    ):
        if swig.source_bn is None:
            raise ValueError("Counterfactual inference requires swig.source_bn.")
        self.swig = swig
        self.max_response_functions = max_response_functions
        self.mechanisms = self._build_mechanisms()
        self.topological_order = self._source_topological_order()
        self.trace: List[str] = []
        self.last_evidence_probability: Optional[float] = None

    def response_space_size(self, *, positive_only: bool = True) -> int:
        size = 1
        for mechanism in self.mechanisms.values():
            for config in mechanism.parent_configs:
                probs = mechanism.probabilities[config]
                if positive_only:
                    n_choices = sum(prob > 0 for prob in probs)
                else:
                    n_choices = len(probs)
                size *= n_choices
        return size

    def explain_query(
        self,
        target: Any,
        factual_evidence: Optional[Mapping[Any, Any]] = None,
    ) -> str:
        target_source = self._target_source(target)
        target_node = self.swig.random_node_for(target_source)
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        response_space = self.response_space_size()

        self.trace = [
            f"Goal: compute distribution for {target_node}.",
            "Semantics: canonical discrete response-table SCM induced by the "
            "source BN CPDs.",
            f"Response functions to enumerate: {response_space}.",
            f"Abduction: condition response tables on factual evidence {evidence}.",
            f"Action: apply interventions {self.swig.swig.interventions}.",
            f"Prediction: read source variable {target_source} in the intervened world.",
        ]
        return "\n".join(self.trace)

    def query(
        self,
        target: Any,
        factual_evidence: Optional[Mapping[Any, Any]] = None,
        *,
        n_round: Optional[int] = None,
    ) -> Dict[Any, float]:
        target_source = self._target_source(target)
        evidence = self._normalize_factual_evidence(factual_evidence or {})
        response_space = self.response_space_size()
        if response_space > self.max_response_functions:
            raise RuntimeError(
                "Exact response-function enumeration would visit "
                f"{response_space} functions, above max_response_functions="
                f"{self.max_response_functions}."
            )

        target_states = self.mechanisms[target_source].states
        weights = {state: 0.0 for state in target_states}
        evidence_weight = 0.0
        visited = 0

        for response_tables, response_weight in self._iter_response_tables():
            visited += 1
            factual_world = self._evaluate_world(response_tables, interventions={})
            if not self._world_matches(factual_world, evidence):
                continue

            evidence_weight += response_weight
            counterfactual_world = self._evaluate_world(
                response_tables,
                interventions=self.swig.swig.interventions,
            )
            weights[counterfactual_world[target_source]] += response_weight

        if evidence_weight <= 0:
            raise ValueError(
                "The factual evidence has zero probability under the canonical "
                f"SCM: {evidence}."
            )

        distribution = {
            state: weight / evidence_weight for state, weight in weights.items()
        }
        if n_round is not None:
            distribution = {
                state: round(prob, n_round)
                for state, prob in distribution.items()
            }

        self.last_evidence_probability = evidence_weight
        self.trace = [
            f"Goal: compute distribution for {self.swig.random_node_for(target_source)}.",
            "Semantics: canonical discrete response-table SCM induced by the "
            "source BN CPDs.",
            f"Enumerated response functions: {visited}.",
            f"Abduction evidence: {evidence}.",
            f"P(evidence) = {evidence_weight}.",
            f"Action interventions: {self.swig.swig.interventions}.",
            f"Result: {distribution}.",
        ]
        return distribution

    def probability_of_factual_evidence(
        self,
        factual_evidence: Mapping[Any, Any],
    ) -> float:
        evidence = self._normalize_factual_evidence(factual_evidence)
        response_space = self.response_space_size()
        if response_space > self.max_response_functions:
            raise RuntimeError(
                "Exact response-function enumeration would visit "
                f"{response_space} functions, above max_response_functions="
                f"{self.max_response_functions}."
            )

        probability = 0.0
        for response_tables, response_weight in self._iter_response_tables():
            factual_world = self._evaluate_world(response_tables, interventions={})
            if self._world_matches(factual_world, evidence):
                probability += response_weight
        return probability

    def _build_mechanisms(self) -> Dict[Node, _ResponseMechanism]:
        mechanisms = {}
        for source in self.swig.swig.source_nodes:
            source_key = self.swig._source_key(source)
            cpd = self.swig.source_bn.get_cpds(source_key)
            if cpd is None:
                raise ValueError(f"Missing CPD for source variable {source!r}.")
            if not isinstance(cpd, TabularCPD):
                raise TypeError(
                    "SwigCounterfactualEngine currently supports TabularCPD "
                    f"only; got {type(cpd)!r} for {source!r}."
                )

            parent_keys = tuple(cpd.variables[1:])
            parents = tuple(str(parent) for parent in parent_keys)
            states = tuple(
                VibecodeSwig._cpd_states(cpd, cpd.variable, int(cpd.variable_card))
            )
            parent_states = tuple(
                tuple(
                    VibecodeSwig._cpd_states(cpd, parent, int(cardinality))
                )
                for parent, cardinality in zip(parent_keys, cpd.cardinality[1:])
            )
            parent_configs = tuple(product(*parent_states)) if parent_states else ((),)

            probabilities = {}
            for config in parent_configs:
                probs = tuple(
                    self._cpd_probability(cpd, state, parent_keys, config)
                    for state in states
                )
                total = sum(probs)
                if total <= 0:
                    raise ValueError(
                        f"CPD for {source!r} assigns zero mass to parent "
                        f"configuration {config!r}."
                    )
                if not np.isclose(total, 1.0):
                    probs = tuple(prob / total for prob in probs)
                probabilities[config] = probs

            mechanisms[source] = _ResponseMechanism(
                source=source,
                source_key=source_key,
                cpd=cpd,
                parents=parents,
                parent_keys=parent_keys,
                states=states,
                parent_states=parent_states,
                parent_configs=parent_configs,
                probabilities=probabilities,
            )
        return mechanisms

    def _source_topological_order(self) -> List[Node]:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.swig.swig.source_nodes)
        graph.add_edges_from(self.swig.swig.source_edges)
        return list(nx.topological_sort(graph))

    def _iter_response_tables(self):
        sources = list(self.topological_order)

        def rec(index, current_tables, current_weight):
            if index == len(sources):
                yield current_tables.copy(), current_weight
                return

            source = sources[index]
            mechanism = self.mechanisms[source]
            for table, table_weight in self._iter_mechanism_tables(mechanism):
                current_tables[source] = table
                yield from rec(index + 1, current_tables, current_weight * table_weight)
                del current_tables[source]

        yield from rec(0, {}, 1.0)

    def _iter_mechanism_tables(self, mechanism: _ResponseMechanism):
        choices_by_config = []
        for config in mechanism.parent_configs:
            choices = [
                (state, probability)
                for state, probability in zip(
                    mechanism.states,
                    mechanism.probabilities[config],
                )
                if probability > 0
            ]
            choices_by_config.append((config, choices))

        for selected_choices in product(
            *(choices for _, choices in choices_by_config)
        ):
            table = {}
            weight = 1.0
            for (config, _), (state, probability) in zip(
                choices_by_config,
                selected_choices,
            ):
                table[config] = state
                weight *= probability
            if weight > 0:
                yield table, weight

    def _evaluate_world(
        self,
        response_tables: Mapping[Node, Mapping[Tuple[Any, ...], Any]],
        *,
        interventions: Mapping[Node, Any],
    ) -> Dict[Node, Any]:
        values = {}
        normalized_interventions = {
            str(source): value for source, value in interventions.items()
        }

        for source in self.topological_order:
            if source in normalized_interventions:
                values[source] = normalized_interventions[source]
                continue

            mechanism = self.mechanisms[source]
            parent_config = tuple(values[parent] for parent in mechanism.parents)
            values[source] = response_tables[source][parent_config]
        return values

    @staticmethod
    def _world_matches(world: Mapping[Node, Any], evidence: Mapping[Node, Any]) -> bool:
        return all(world[source] == value for source, value in evidence.items())

    def _normalize_factual_evidence(
        self,
        evidence: Mapping[Any, Any],
    ) -> Dict[Node, Any]:
        normalized = {}
        for raw_key, value in evidence.items():
            key = str(raw_key)
            if key in self.swig.swig.source_nodes:
                source = key
            elif key in self.swig.swig.node_info:
                info = self.swig.swig.node_info[key]
                if info.is_fixed:
                    raise ValueError(
                        "Factual evidence should use source/random variables, "
                        f"not fixed intervention node {key!r}."
                    )
                if info.fixed_ancestors:
                    raise ValueError(
                        "This engine conditions on factual-world evidence only; "
                        f"{key!r} is an intervention-indexed SWIG node."
                    )
                source = info.source
            else:
                raise ValueError(f"Unknown factual evidence variable {raw_key!r}.")
            normalized[source] = value
        return normalized

    def _target_source(self, target: Any) -> Node:
        key = str(target)
        if key in self.swig.swig.source_nodes:
            return key
        if key in self.swig.swig.node_info:
            info = self.swig.swig.node_info[key]
            if info.is_fixed:
                raise ValueError(f"Cannot query fixed intervention node {key!r}.")
            return info.source
        raise ValueError(f"Unknown SWIG target node {target!r}.")

    @staticmethod
    def _cpd_probability(
        cpd: TabularCPD,
        state: Any,
        parent_keys: Tuple[Any, ...],
        parent_config: Tuple[Any, ...],
    ) -> float:
        kwargs = {str(cpd.variable): state}
        kwargs.update(
            {str(parent): value for parent, value in zip(parent_keys, parent_config)}
        )
        try:
            return float(cpd.get_value(**kwargs))
        except Exception:
            return SwigCounterfactualEngine._cpd_probability_from_table(
                cpd,
                state,
                parent_keys,
                parent_config,
            )

    @staticmethod
    def _cpd_probability_from_table(
        cpd: TabularCPD,
        state: Any,
        parent_keys: Tuple[Any, ...],
        parent_config: Tuple[Any, ...],
    ) -> float:
        child_states = VibecodeSwig._cpd_states(
            cpd,
            cpd.variable,
            int(cpd.variable_card),
        )
        child_index = child_states.index(state)

        column_index = 0
        for parent, parent_value, cardinality in zip(
            parent_keys,
            parent_config,
            cpd.cardinality[1:],
        ):
            parent_states = VibecodeSwig._cpd_states(cpd, parent, int(cardinality))
            column_index = column_index * len(parent_states)
            column_index += parent_states.index(parent_value)

        return float(cpd.get_values()[child_index, column_index])


class SwigInferenceSketch(SwigCounterfactualEngine):
    """Backward-compatible name for the now-implemented experimental engine."""


__all__ = [
    "SwigCounterfactualEngine",
    "SwigInferenceSketch",
    "SwigNodeInfo",
    "SwigSpec",
    "VibecodeSwig",
]
