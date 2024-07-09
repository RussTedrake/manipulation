import dataclasses as dc
import typing

from pydrake.all import ModelDirective, ScopedName

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
    type: typing.Literal["frame", "model"]


@dc.dataclass(frozen=True)
class Edge:
    parent: Node
    child: Node
    directive: ModelDirective  # either AddFrame or AddWeld


class DirectivesTree:
    def __init__(self, flattened_directives: typing.List[ModelDirective]):
        self.flattened_directives = flattened_directives
        self.add_model_directives: typing.Dict[str, ModelDirective] = dict()

        # Names of frames added by `add_frame` directives.
        # The default "world" frame exists implicitly.
        self.frame_names: typing.Set[str] = {"world"}

        # Names of models added by `add_model` directives.
        self.model_names: typing.Set[str] = set()

        # Mapping from parent nodes to the set of its outgoing edges.
        self.edges: typing.Dict[Node, typing.Set[Edge]] = dict()

        # Read node names.
        for d in self.flattened_directives:
            if d.add_model:
                self.model_names.add(d.add_model.name)
                self.add_model_directives[d.add_model.name] = d
            if d.add_frame:
                self.frame_names.add(d.add_frame.name)

        # Create edges.
        for d in self.flattened_directives:
            if d.add_weld:
                parent = self._MakeNode(d.add_weld.parent)
                child = self._MakeNode(d.add_weld.child)
                edge = Edge(parent, child, d)
                if parent not in self.edges:
                    self.edges[parent] = {edge}
                else:
                    self.edges[parent].add(edge)

            if d.add_frame:
                parent = self._MakeNode(d.add_frame.X_PF.base_frame)
                child = self._MakeNode(d.add_frame.name)
                edge = Edge(parent, child, d)
                if parent not in self.edges:
                    self.edges[parent] = {edge}
                else:
                    self.edges[parent].add(edge)

    def _MakeNode(self, name: str):
        # Check if this is an added frame.
        if name in self.frame_names:
            return Node(name, "frame")

        # Check if this corresponds to an added model.
        model_name = ScopedName.Parse(name).get_namespace()
        if model_name in self.model_names:
            return Node(model_name, "model")

        raise ValueError(
            f"Node {name} not found in the tree. It neither corresponds to a "
            + f"frame {self.frame_names} nor a model instance {self.model_names}."
        )

    def GetWeldedDescendantsAndDirectives(
        self, model_instance_names: typing.List[str]
    ) -> typing.Tuple[typing.Set[str], typing.List[ModelDirective]]:

        def _RecursionCall(
            node: Node,
        ) -> typing.Tuple[typing.Set[str], typing.Set[ModelDirective]]:
            """
            Args:
                node (Node): The node to start this recursion call from.

            Returns:
                Set[str]: Names of proper descendant models that are welded to
                    the input node.
                Set[ModelDirective]: The directives that need to be added to
                    weld the proper descendant models to the input node.
            """
            descendants: typing.Set[str] = set()
            directives: typing.Set[ModelDirective] = set()

            for edge in self.edges.get(node, set()):
                _descendants, _directives = _RecursionCall(edge.child)

                # If the child is a model instance, add its AddModel directive,
                # and add its model's name to the set of descendants.
                if edge.child.type == "model":
                    _descendants.add(edge.child.name)
                    _directives.add(self.add_model_directives[edge.child.name])

                # If the child node has non-zero descendants, add the edge
                # directive that leads to the child node.
                if len(_descendants) > 0:
                    directives.add(edge.directive)

                    # Add the recursion results.
                    descendants.update(_descendants)
                    directives.update(_directives)

            return descendants, directives

        descendants: typing.Set[str] = set()
        directives: typing.Set[ModelDirective] = set()
        for model_instance_name in model_instance_names:
            assert model_instance_name in self.add_model_directives
            node = Node(model_instance_name, "model")
            _descendants, _directives = _RecursionCall(node)
            descendants.update(_descendants)
            directives.update(_directives)

        return descendants, self.TopologicallySortDirectives(directives)

    def GetWeldToWorldDirectives(
        self, model_instance_names: typing.List[str]
    ) -> typing.List[ModelDirective]:

        def _RecursionCall(node: Node) -> typing.Set[ModelDirective]:
            """
            Args:
                node (Node): The node to start this recursion call from.

            Returns:
                Set[ModelDirective]: The directives that need to be added to
                    weld all `model_instance_names` to the "world" frame.
            """
            directives: typing.Set[ModelDirective] = set()

            # Base case: if the node is one of the model instances, add its
            # AddModel directive and return immediately.
            if node.type == "model" and node.name in model_instance_names:
                directives.add(self.add_model_directives[node.name])
                return directives

            for edge in self.edges.get(node, set()):
                _directives = _RecursionCall(edge.child)

                # If the child node has non-zero directives, add the edge
                # directive that leads to the child node.
                if len(_directives) > 0:
                    directives.add(edge.directive)

                    # Add the recursion results.
                    directives.update(_directives)

            return directives

        world_node = Node("world", "frame")
        return self.TopologicallySortDirectives(_RecursionCall(world_node))

    def TopologicallySortDirectives(
        self, directives: typing.Set[ModelDirective]
    ) -> typing.List[ModelDirective]:
        return [d for d in self.flattened_directives if d in directives]
