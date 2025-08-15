#
# This file is part of the GNU Radio LLM project.
#

import json

from flowgraph.schema import Flowgraph


class FlowgraphBuilder:
    @staticmethod
    def from_dict(data: dict) -> Flowgraph:
        """
        Validate and construct a flowgraph from a dictionary
        """
        return Flowgraph(**data)

    @staticmethod
    def from_json(json_str: str) -> Flowgraph:
        """
        Validate and construct a flowgraph from a JSON string
        """
        try:
            data = json.loads(json_str)
            return FlowgraphBuilder.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON: {e}')
