import dataclasses as dc
from typing import Dict, List, Literal, Set

from pydrake.all import AddFrame, AddWeld, ModelDirective, ScopedName

"""
===============================================================================
DirectivesTree: A set of `add_weld` and `add_frame` directives induces a tree.
===============================================================================

• Nodes of the tree
  -----------------
    • Each added frame is a node in the tree.
    • Each model instance (identified by its namespace is a node in the tree).

• Edges of the tree
  -----------------
    • Each `add_weld` directive is a directed edge from its parent frame/model
        to its child frame/model.
    • Each `add_frame` directive is a directed edge from its base frame/model
        to its child frame/model.
"""


@dc.dataclass(frozen=True)
class Node:
    name: str
    type: Literal["frame", "model"]


@dc.dataclass(frozen=True)
class Edge:
    parent: Node
    child: Node
    directive: AddWeld | AddFrame


class DirectivesTree:
    def __init__(self, flattened_directives: List[ModelDirective]):
        # Names of frames added by `add_frame` directives
        self.frame_names: Set[str] = {}

        # Names of models added by `add_model` directives
        self.model_names: Set[str] = {}

        # Mapping from parent nodes to the set of its outgoing edges.
        self.edges: Dict[Node, Set[Edge]] = dict()

        # Read node names.
        for d in flattened_directives:
            if d.add_model:
                self.model_names.add(d.add_model.name)
            if d.add_frame:
                self.frame_names.add(d.add_frame.name)

        # Create edges.
        for d in flattened_directives:
            if d.add_weld:
                parent = self.make_node(d.add_weld.parent)
                child = self.make_node(d.add_weld.child)
                edge = Edge(parent, child, d.add_weld)
                if parent not in self.edges:
                    self.edges[parent] = {edge}
                else:
                    self.edges[parent].add(edge)

            if d.add_frame:
                parent = self.make_node(d.add_frame.X_PF.base_frame)
                child = self.make_node(d.add_frame.name)
                edge = Edge(parent, child, d.add_weld)
                if parent not in self.edges:
                    self.edges[parent] = {edge}
                else:
                    self.edges[parent].add(edge)

    def make_node(self, name: str):
        # Check if this is an added frame
        if name in self.frame_names:
            return self.Node(name, "frame")

        # Check if this corresponds to an added model
        model_name = ScopedName.Parse(name).get_namespace()
        if model_name in self.model_names:
            return self.Node(name, "model")

        raise ValueError(
            f"Node {name} not found in the tree. It neither corresponds to a ",
            f"frame [{self.frame_names}] nor a model instance [{self.model_names}].",
        )
