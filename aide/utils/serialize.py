import copy
import json
from pathlib import Path, PurePath
from typing import Type, TypeVar

import dataclasses_json
from ..journal import Journal


class PathEncoder(json.JSONEncoder):
    """JSON encoder that handles Path objects."""
    def default(self, obj):
        if isinstance(obj, PurePath):
            return str(obj)
        return super().default(obj)


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize AIDE dataclasses (such as Journals) to JSON."""
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        # Store parent as ID string (not object) to avoid circular references
        for n in obj.nodes:
            n.parent = n.parent.id if n.parent is not None else None  # type: ignore
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["__version"] = "3"

    return json.dumps(obj_dict, indent=2, cls=PathEncoder)


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to AIDE dataclasses."""
    obj_dict = json.loads(s)
    version = obj_dict.get("__version", "1")

    # For v3 format, extract parent IDs before from_dict (which expects Node objects)
    parent_ids = {}
    if cls == Journal and version == "3":
        for node_dict in obj_dict.get("nodes", []):
            if node_dict.get("parent") is not None:
                parent_ids[node_dict["id"]] = node_dict["parent"]
                node_dict["parent"] = None  # Clear so from_dict doesn't fail

    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}

        if version == "3":
            # New format: restore parent relationships from extracted IDs
            for child_id, parent_id in parent_ids.items():
                if child_id in id2nodes and parent_id in id2nodes:
                    id2nodes[child_id].parent = id2nodes[parent_id]
                    id2nodes[child_id].__post_init__()
        else:
            # Old format (v1/v2): parent relationships in node2parent
            for child_id, parent_id in obj_dict.get("node2parent", {}).items():
                id2nodes[child_id].parent = id2nodes[parent_id]
                id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)
