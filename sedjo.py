import abc
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.qubit import Qubit
from pydynaa import EventExpression
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import create_two_node_network

class SEDJOProgram(Program, abc.ABC):
    PEER: str

    def __init__(self, num_epr: int):
        self._num_epr = num_epr
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)

    def _create_epr_pairs(self, context: ProgramContext, is_init: bool) -> Generator[EventExpression, None, List[Qubit]]:
        epr_socket = context.epr_sockets[self.PEER]
        conn = context.connection
        self.logger.info(f"{'Alice' if is_init else 'Bob'}: Requesting/Receiving {self._num_epr} EPR pairs.")
        if is_init:
            qubits = epr_socket.create_keep(self._num_epr)
        else:
            qubits = epr_socket.recv_keep(self._num_epr)
        self.logger.info(f"{'Alice' if is_init else 'Bob'}: EPR pairs created/received, flushing.")
        yield from conn.flush()
        self.logger.info(f"{'Alice' if is_init else 'Bob'}: EPR pairs ready.")
        return qubits

    def _prepare_local_qubit(self, context: ProgramContext, owner: str) -> Qubit:
        conn = context.connection
        self.logger.info(f"{owner}: Creating local qubit |0>.")
        qloc = Qubit(conn)
        qloc.X()
        qloc.H()
        self.logger.info(f"{owner}: Local qubit prepared in (H*X)|0> state.")
        return qloc

    def _apply_key_sedjo(self, qubits: List[Qubit], qloc: Qubit, key: List[int], owner: str):
        self.logger.info(f"{owner}: Applying DJ oracle based on key {key}.")
        for i, bit in enumerate(key):
            if bit == 1:
                qubits[i].Z()
        for q in qubits:
            q.cnot(qloc)
        for i, bit in enumerate(key):
            if bit == 1:
                qubits[i].X()
        self.logger.info(f"{owner}: Oracle application based on key done.")

    def _apply_h_to_all(self, qubits: List[Qubit], owner: str):
        self.logger.info(f"{owner}: Applying H to all EPR qubits.")
        for q in qubits:
            q.H()
        self.logger.info(f"{owner}: H applied to all EPR qubits.")

    def _xor_bits(self, a: List[int], b: List[int]) -> List[int]:
        return [x ^ y for x, y in zip(a, b)]

    def _measure_all(self, context: ProgramContext, qubits: List[Qubit], owner: str) -> Generator[EventExpression, None, List[int]]:
        conn = context.connection
        self.logger.info(f"{owner}: Measuring all EPR qubits.")
        measurement_futures = []
        for q in qubits:
            m = q.measure()  # returns a Future
            measurement_futures.append(m)

        self.logger.info(f"{owner}: All measurement scheduled, flushing to retrieve results.")
        yield from conn.flush()  # ensure all measurement results are now available
        outcomes = [int(m) for m in measurement_futures]
        self.logger.info(f"{owner}: Measurement outcomes: {outcomes}")
        return outcomes


class AliceProgram(SEDJOProgram):
    PEER = "Bob"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._num_epr + 1,
        )

    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        csocket = context.csockets[self.PEER]

        qubits = yield from self._create_epr_pairs(context, is_init=True)
        qloc = self._prepare_local_qubit(context, "Alice")

        s_A = [random.randint(0, 1) for _ in range(self._num_epr)]
        self.logger.info(f"Alice: Chosen s_A = {s_A}")

        self._apply_key_sedjo(qubits, qloc, s_A, "Alice")
        self._apply_h_to_all(qubits, "Alice")

        z0 = yield from self._measure_all(context, qubits, "Alice")
        self.logger.info(f"Alice: z0 = {z0}")

        # Wait for Bob's message
        self.logger.info("Alice: Waiting for Bob's message (s_B or Abort).")
        msg = yield from csocket.recv_structured()
        self.logger.info(f"Alice: Received message {msg}, {msg.payload}")
        if msg.payload == "Abort":
            self.logger.info("Alice: Received abort message from Bob. Protocol failed.")
            return {"final_key": None}

        s_B = msg.payload
        self.logger.info(f"Alice: Received s_B = {s_B}")

        final_key = self._xor_bits(self._xor_bits(s_A, s_B), z0)
        self.logger.info(f"Alice: Computed final key = {final_key}")

        return {"final_key": final_key, "s_A": s_A, "z0": z0}


class BobProgram(SEDJOProgram):
    PEER = "Alice"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._num_epr + 1,
        )

    def run(
        self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        csocket = context.csockets[self.PEER]

        qubits = yield from self._create_epr_pairs(context, is_init=False)
        qloc = self._prepare_local_qubit(context, "Bob")

        s_B = [random.randint(0, 1) for _ in range(self._num_epr)]
        self.logger.info(f"Bob: Chosen s_B = {s_B}")

        self._apply_key_sedjo(qubits, qloc, s_B, "Bob")
        self._apply_h_to_all(qubits, "Bob")

        result = yield from self._measure_all(context, qubits, "Bob")
        self.logger.info(f"Bob: result (s_A⊕s_B⊕z0) = {result}")

        if all(r == 0 for r in result):
            csocket.send_structured(StructuredMessage("Abort", None))
            self.logger.info("Bob: Result all zeros, aborting protocol.")
            return {"final_result": None, "s_B": s_B, "measurement_result": result}
        else:
            csocket.send_structured(StructuredMessage("s_B", s_B))
            self.logger.info("Bob: s_B sent to Alice.")
            return {"s_B": s_B, "measurement_result": result}


if __name__ == "__main__":
    cfg = create_two_node_network(node_names=["Alice", "Bob"], link_noise=0.1)
    num_epr = 5

    alice_program = AliceProgram(num_epr=num_epr)
    bob_program = BobProgram(num_epr=num_epr)

    # Set logging to INFO for detailed logs
    alice_program.logger.setLevel(logging.INFO)
    bob_program.logger.setLevel(logging.INFO)

    alice_results, bob_results = run(
        config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=1
    )

    for i, (alice_result, bob_result) in enumerate(zip(alice_results, bob_results)):
        print(f"run {i}:")
        print(f"Alice final key: {alice_result.get('final_key')}")
        print(f"Alice s_A: {alice_result.get('s_A')}")
        print(f"Alice z0: {alice_result.get('z0')}")
        print(f"Bob measurement result (s_A⊕s_B⊕z0): {bob_result.get('measurement_result')}")
        print(f"Bob s_B: {bob_result.get('s_B')}")
