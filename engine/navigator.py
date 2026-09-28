import os
from typing import Dict, Any, Optional, List
from protocols.messages import MessagePayload, RequestType, ConflictPacket
from not_needed.storage.contract_graph import ContractGraph
from not_needed.blast_radius import BlastRadiusAnalyzer

class Navigator:
    def __init__(self, graph: ContractGraph):
        self.graph = graph
        self.analyzer = BlastRadiusAnalyzer(graph)
        self.active_workers: Dict[str, Any] = {}
        self.task_history: List[str] = []

    def register_worker(self, worker_id: str, worker_instance):
        self.active_workers[worker_id] = worker_instance

    def query_contract(self, target_symbol: str) -> dict:
        contract = self.graph.get_contract(target_symbol)
        if contract:
            return {"status": "SUCCESS", "contract": contract.model_dump()}
        
        # Check if symbol exists but is dirty
        raw_node = self.graph.nodes.get(target_symbol)
        if raw_node and raw_node.status == "DIRTY":
            return {"status": "STALE", "message": f"Contract for {target_symbol} is DIRTY. Re-verification required."}
            
        return {"status": "NOT_FOUND", "message": f"Symbol {target_symbol} not found in interface graph."}

    def fetch_code_slice(self, target_symbol: str, start: int, end: int) -> str:
        node = self.graph.nodes.get(target_symbol)
        if not node or not os.path.exists(node.file_path):
            return "File not found."
        with open(node.file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[max(0, start - 1):end])

    def handle_request(self, packet: MessagePayload, workers: Dict[str, Any]) -> Dict[str, Any]:
        if packet.request_type == RequestType.DEMAND_CONTRACT:
            return self.query_contract(packet.target_node)

        target_worker = workers.get(packet.target_node) or self.active_workers.get(packet.target_node)
        if not target_worker:
            return {"status": "ERROR", "message": f"Worker for {packet.target_node} not running."}

        if packet.request_type == RequestType.DEMAND_CODE:
            lines = packet.line_range or [1, 25]
            code_slice = target_worker.extract_slice(lines[0], lines[1])
            return {"status": "SUCCESS", "verbatim_slice": code_slice}

        return {"status": "UNHANDLED"}

    def mediate_conflict(self, worker_1_id: str, target_1: str,
                         worker_2_id: str, target_2: str) -> ConflictPacket:
        r1_count, r1_files = self.analyzer.compute_radius(target_1)
        r2_count, r2_files = self.analyzer.compute_radius(target_2)

        return ConflictPacket(
            worker_1_id=worker_1_id,
            worker_2_id=worker_2_id,
            mismatch_description=f"Contract mismatch between {target_1} and {target_2}",
            blast_radius_worker_1=r1_count,
            blast_radius_worker_2=r2_count,
            affected_files=list(r1_files.union(r2_files)),
            options=[
                f"Align {worker_1_id} to match {worker_2_id} (Touches {r1_count} file(s))",
                f"Align {worker_2_id} to match {worker_1_id} (Touches {r2_count} file(s))",
                "Custom developer instruction"
            ]
        )