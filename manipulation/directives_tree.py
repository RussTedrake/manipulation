import dataclasses as dc
import typing

from pydrake.all import ModelDirective, ScopedName


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
    """
    ===========================================================================
    DirectivesTree: A set of `add_weld` & `add_frame` directives induce a tree
    ===========================================================================

    • Nodes of the tree
      -----------------
        • Each added frame is a node in the tree.
        • Each model instance (identified by its namespace is a node in the tree).

    • Edges of the tree
      -----------------
        • Each `add_weld` directive is a directed edge from its parent
          frame/model to its child frame/model.
        • Each `add_frame` directive is a directed edge from its base
          frame/model to its child frame/model.
    """

    def __init__(self, flattened_directives: typing.List[ModelDirective]):
        self.flattened_directives = flattened_directives

        # Dictionary of all `add_model` directives indexed by the model names.
        self.add_model_directives: typing.Dict[str, ModelDirective] = dict()

        # Names of frames added by `add_frame` directives.
        # The default "world" frame exists implicitly.
        self.frame_names: typing.Set[str] = {"world"}

        # Names of models added by `add_model` directives.
        self.model_names: typing.Set[str] = set()

        # Mapping from parent nodes to the set of its outgoing child edges.
        self.child_edges: typing.Dict[Node, typing.Set[Edge]] = dict()

        # Mapping from child nodes to its incoming parent edge.
        self.parent_edge: typing.Dict[Node, Edge] = dict()

        # Read node names.
        for d in self.flattened_directives:
            if d.add_model:
                model_name = d.add_model.name
                self.model_names.add(model_name)
                self.add_model_directives[model_name] = d
            if d.add_frame:
                frame_name = d.add_frame.name
                self.frame_names.add(frame_name)

        # Create edges.
        for d in self.flattened_directives:
            if d.add_weld:
                parent_name = d.add_weld.parent
                child_name = d.add_weld.child
                self._AddEdge(parent_name, child_name, d)

            if d.add_frame:
                parent_name = d.add_frame.X_PF.base_frame
                child_name = d.add_frame.name
                self._AddEdge(parent_name, child_name, d)

    def _MakeNode(self, name: str) -> None:
        # Check if this is an added frame.
        if name in self.frame_names:
            return Node(name, "frame")

        # Check if this corresponds to an added model.
        model_name = ScopedName.Parse(name).get_namespace()
        if model_name in self.model_names:
            return Node(model_name, "model")

        raise ValueError(
            f"Node {name} not found in the tree. It neither corresponds to a "
            f"frame {self.frame_names} nor a model instance {self.model_names}."
        )

    def _AddEdge(
        self, parent_name: str, child_name: str, directive: ModelDirective
    ) -> None:
        parent = self._MakeNode(parent_name)
        child = self._MakeNode(child_name)
        edge = Edge(parent, child, directive)

        # Add child edge.
        if parent not in self.child_edges:
            self.child_edges[parent] = {edge}
        else:
            self.child_edges[parent].add(edge)

        # Add parent edge.
        if child in self.parent_edge:
            edge = self.parent_edge[child]
            raise ValueError(
                f"Child {child} already has a parent {edge.parent} with directive "
                f"{edge.directive}. Directives must form a tree where each node "
                "has at most one parent."
            )
        self.parent_edge[child] = edge

    def GetWeldedDescendantsAndDirectives(
        self, model_instance_names: typing.List[str]
    ) -> typing.Tuple[typing.Set[str], typing.Set[ModelDirective]]:
        """
        Returns:
            Set[str]: Names of proper descendant models of the input
                `model_instance_names` in this directives tree.
            Set[ModelDirective]: The directives that need to be added to weld
                the proper descendant models to the input model instances.
        """

        def _RecursiveCall(
            node: Node,
        ) -> typing.Tuple[typing.Set[str], typing.Set[ModelDirective]]:
            """
            Args:
                node (Node): The node to start this recursive call from.

            Returns:
                Set[str]: Names of proper descendant models that are welded to
                    the input node.
                Set[ModelDirective]: The directives that need to be added to
                    weld the proper descendant models to the input node.
            """
            descendants: typing.Set[str] = set()
            directives: typing.Set[ModelDirective] = set()

            for edge in self.child_edges.get(node, set()):
                _descendants, _directives = _RecursiveCall(edge.child)

                # If the child is a model instance, add its AddModel directive,
                # and add its model's name to the set of descendants.
                if edge.child.type == "model":
                    child_model_name = edge.child.name
                    _descendants.add(child_model_name)
                    _directives.add(self.add_model_directives[child_model_name])

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
            _descendants, _directives = _RecursiveCall(node)
            descendants.update(_descendants)
            directives.update(_directives)

        # Don't include the input model instances in the descendants set.
        descendants.difference_update(model_instance_names)

        return descendants, directives

    def GetDirectivesFromModelsToRoot(
        self, model_instance_names: typing.List[str]
    ) -> typing.Set[ModelDirective]:
        """
        Recursively traverses the tree from `model_instance_names` to the root
        nodes of the tree (which may or may not be "world").

        Returns:
            Set[ModelDirective]: The directives that need to be added to
                weld all `model_instance_names`. This is the minimal set of
                directives necessary to support all `model_instance_names` in
                a plant.
        """
        directives: typing.Set[ModelDirective] = set()
        for model_instance_name in model_instance_names:
            # Add input model_instance's add_model directive.
            assert model_instance_name in self.add_model_directives
            node = Node(model_instance_name, "model")
            directives.add(self.add_model_directives[node.name])

            # Loop until we reach the root node.
            while node in self.parent_edge:
                edge = self.parent_edge[node]
                directives.add(edge.directive)
                node = edge.parent
                if node.type == "model":
                    directives.add(self.add_model_directives[node.name])

        return directives

    def TopologicallySortDirectives(
        self, directives: typing.Set[ModelDirective]
    ) -> typing.List[ModelDirective]:
        """
        This assumes that the flattened directives are in a valid topologically
        sorted order.
        """
        return [d for d in self.flattened_directives if d in directives]
