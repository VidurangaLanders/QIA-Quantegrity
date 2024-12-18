import abc
import logging
import random
from typing import Any, Dict, Generator, List, Optional, Tuple

from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.qubit import Qubit
from pydynaa import EventExpression
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import create_two_node_network

##############################
# Logging Configuration
##############################
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("netsquid").setLevel(logging.WARNING)
logging.getLogger("netqasm").setLevel(logging.WARNING)
logging.getLogger("squidasm").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

##############################
# QRNG Service (Simulated)
##############################
class QRNGService:
    @staticmethod
    def generate_random_bits(n: int) -> List[int]:
        return [random.randint(0,1) for _ in range(n)]

    @staticmethod
    def generate_random_str(n: int) -> str:
        return "".join(str(b) for b in QRNGService.generate_random_bits(n))

##############################
# XOR-based Encryption (placeholder)
##############################
def xor_encrypt_str(message: str, key_str: str) -> str:
    # XOR each char of message with key bits
    key_bits = [int(x) for x in key_str]
    enc = []
    for i, ch in enumerate(message):
        ch_val = ord(ch)
        bit = key_bits[i % len(key_bits)]
        ch_val_xor = ch_val ^ bit
        enc.append(chr(ch_val_xor))
    return "".join(enc)

##############################
# SEDJO QKD Protocol Classes
##############################
class SEDJOProgram(Program, abc.ABC):
    PEER: str

    def __init__(self, num_epr: int, s_input: Optional[List[int]]=None):
        self._num_epr = num_epr
        self._s_input = s_input
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)

    def _create_epr_pairs(self, context: ProgramContext, is_init: bool) -> Generator[EventExpression, None, List[Qubit]]:
        epr_socket = context.epr_sockets[self.PEER]
        conn = context.connection
        qubits = epr_socket.create_keep(self._num_epr) if is_init else epr_socket.recv_keep(self._num_epr)
        yield from conn.flush()
        return qubits

    def _prepare_local_qubit(self, context: ProgramContext) -> Qubit:
        qloc = Qubit(context.connection)
        qloc.X()
        qloc.H()
        return qloc

    def _apply_key_sedjo(self, qubits: List[Qubit], qloc: Qubit, key: List[int]):
        # Step 1: Apply Z to each EPR qubit corresponding to a '1' bit
        for i, bit in enumerate(key):
            if bit == 1:
                qubits[i].Z()
        
        # Step 2: Apply CNOT from each EPR qubit to the local qubit
        # For the SEDJO protocol, we do this for all qubits, not just the ones with bit=1.
        for q in qubits:
            q.cnot(qloc)
        
        # Step 3: Apply X to each EPR qubit corresponding to a '1' bit
        for i, bit in enumerate(key):
            if bit == 1:
                qubits[i].X()


    def _apply_h_to_all(self, qubits: List[Qubit]):
        for q in qubits:
            q.H()

    def _xor_bits(self, a: List[int], b: List[int]) -> List[int]:
        return [x ^ y for x, y in zip(a, b)]

    def _measure_all(self, context: ProgramContext, qubits: List[Qubit]) -> Generator[EventExpression, None, List[int]]:
        conn = context.connection
        futures = [q.measure() for q in qubits]
        yield from conn.flush()
        return [int(m) for m in futures]

class AliceSEDJO(SEDJOProgram):
    PEER = "Bob"
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_sedjo",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._num_epr + 1,
        )

    def run(self, context: ProgramContext) -> Generator[EventExpression, None, Dict[str, Any]]:
        csocket = context.csockets[self.PEER]
        qubits = yield from self._create_epr_pairs(context, True)
        qloc = self._prepare_local_qubit(context)
        s_A = self._s_input if self._s_input is not None else [random.randint(0,1) for _ in range(self._num_epr)]
        self._apply_key_sedjo(qubits, qloc, s_A)
        self._apply_h_to_all(qubits)
        z0 = yield from self._measure_all(context, qubits)
        msg = yield from csocket.recv_structured()
        if msg == "Abort":
            return {"final_key": None}
        s_B = msg.payload
        final_key = self._xor_bits(self._xor_bits(s_A, s_B), z0)
        return {"final_key": final_key, "s_A": s_A, "z0": z0}

class BobSEDJO(SEDJOProgram):
    PEER = "Alice"
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_sedjo",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=self._num_epr + 1,
        )

    def run(self, context: ProgramContext) -> Generator[EventExpression, None, Dict[str, Any]]:
        csocket = context.csockets[self.PEER]
        qubits = yield from self._create_epr_pairs(context, False)
        qloc = self._prepare_local_qubit(context)
        s_B = self._s_input if self._s_input is not None else [random.randint(0,1) for _ in range(self._num_epr)]
        self._apply_key_sedjo(qubits, qloc, s_B)
        self._apply_h_to_all(qubits)
        result = yield from self._measure_all(context, qubits)
        if all(r == 0 for r in result):
            csocket.send_structured(StructuredMessage("Abort", None))
            return {"final_result": None}
        csocket.send_structured(StructuredMessage("s_B", s_B))
        return {"s_B": s_B, "measurement_result": result}

def run_sedjo_qkd(num_epr: int, s_A: Optional[List[int]]=None, s_B: Optional[List[int]]=None) -> Optional[List[int]]:
    cfg = create_two_node_network(node_names=["Alice", "Bob"], link_noise=0.0)
    alice_program = AliceSEDJO(num_epr=num_epr, s_input=s_A)
    bob_program = BobSEDJO(num_epr=num_epr, s_input=s_B)
    alice_program.logger.setLevel(logging.ERROR)
    bob_program.logger.setLevel(logging.ERROR)

    alice_results, bob_results = run(
        config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=1
    )
    return alice_results[0]["final_key"]

##############################
# Mixnet and Scantegrity-like Tables
##############################
class Mixnet:
    def __init__(self):
        self.votes = []  # store tuples (encrypted_vote, ballot_id)
        self.confirmation_codes = {} # {BallotID: ConfirmationCode}

    def add_vote(self, ballot_id: str, confirmation_code: str, vote: str):
        self.votes.append((ballot_id, confirmation_code, vote))

    def run_mixnet(self):
        # Shuffle votes to break link between voter and vote
        random.shuffle(self.votes)

    def generate_tables(self):
        # For demonstration, we create P,Q,R,S tables:
        # P: Original (BallotID -> ConfirmationCode)
        # Q: Shuffled confirmation codes
        # R: Mappings from Q to P
        # S: Final tally

        # P table
        P = [(i+1, b_id, c_c) for i, (b_id, c_c, v) in enumerate(self.votes)]

        # Q table: Shuffled confirmation codes
        shuffled = self.votes[:]
        random.shuffle(shuffled)
        Q = [(i+1, c_c) for i, (b_id, c_c, v) in enumerate(shuffled)]

        # R table: Map from Q index to P index
        # Just an example: R shows from which original entry the Q entry came
        R = []
        for qidx, (q_index, q_c_c) in enumerate(Q, start=1):
            for pidx, (p_index, p_b_id, p_c_c) in enumerate(P, start=1):
                if q_c_c == p_c_c:
                    R.append((qidx, pidx))
                    break

        # S table: Final tally (just count votes for demonstration)
        tally = {}
        for (b_id, c_c, v) in self.votes:
            tally[v] = tally.get(v,0)+1
        S = [(cand, count) for cand, count in tally.items()]

        return P, Q, R, S


##############################
# Election Authority and Voter Classes
##############################
class ElectionAuthority:
    def __init__(self):
        self.database = {}
        self.preconfigured_ballots = [("Ballot_0001", "CCode123"),
                                      ("Ballot_0002", "CCode456"),
                                      ("Ballot_0003", "CCode789")]
        # In real scenario, larger sets and complexity
        self.mixnet = Mixnet()

    def register_voter(self, national_id: str) -> Tuple[str, str, str, str]:
        bs_k = QRNGService.generate_random_str(8)
        q_k1 = QRNGService.generate_random_str(8)
        q_k2 = QRNGService.generate_random_str(8)
        v_id = "V" + QRNGService.generate_random_str(4)

        logger.info(f"Registering voter with NID={national_id}")
        logger.info(f"Generated BS_K={bs_k}, Q_K1={q_k1}, Q_K2={q_k2}, V_ID={v_id}")

        q_k1_bits = [int(x) for x in q_k1]
        bs_k_bits = [int(x) for x in bs_k]
        length = min(len(q_k1_bits), len(bs_k_bits))
        enc_q_k1_bits = [(q_k1_bits[i] ^ bs_k_bits[i]) for i in range(length)]
        enc_q_k1 = "".join(str(b) for b in enc_q_k1_bits)

        self.database[v_id] = {"BS_K": bs_k, "Q_K1": q_k1, "Q_K2": q_k2, "NID": national_id}
        logger.info(f"Voter {v_id} registered. Enc_Q_K1 on ID card={enc_q_k1}")
        return v_id, enc_q_k1, q_k2, bs_k

    def verify_device(self, v_id: str, bs_k: str) -> str:
        q_k2 = self.database[v_id]["Q_K2"]
        otp = "DEVICE_OTP"
        otp_enc = xor_encrypt_str(otp, q_k2)
        return otp_enc

    def initiate_election_login(self, v_id: str) -> str:
        q_k1 = self.database[v_id]["Q_K1"]
        return q_k1

    def send_voting_otp(self, aq_k1: str) -> str:
        otp = "VOTING_OTP"
        return xor_encrypt_str(otp, aq_k1)

    def send_ballot_id(self, vq_k1: str) -> Tuple[str, str]:
        # Pick a ballot from preconfigured
        ballot_id, c_c = random.choice(self.preconfigured_ballots)
        enc_b_id = xor_encrypt_str(ballot_id, vq_k1)
        return enc_b_id, c_c

    def send_confirmation_code(self, vq_k1: str, c_c: str) -> str:
        return xor_encrypt_str(c_c, vq_k1)

    def record_vote(self, b_id: str, c_c: str, vote: str):
        # Add to mixnet for tallying later
        self.mixnet.add_vote(b_id, c_c, vote)

class Voter:
    def __init__(self, ea: ElectionAuthority, national_id: str, attempt_scenario: str):
        # attempt_scenario could be: "success", "qk1_mismatch", "device_fail", etc.
        self.ea = ea
        self.national_id = national_id
        self.attempt_scenario = attempt_scenario

        self.v_id = None
        self.enc_q_k1 = None
        self.q_k2 = None
        self.bs_k = None
        self.q_k1 = None
        self.aq_k1 = None
        self.vq_k1 = None
        self.vote_cast = None
        self.status = "Not Attempted"

    def registration(self):
        self.v_id, self.enc_q_k1, self.q_k2, self.bs_k = self.ea.register_voter(self.national_id)

    def device_registration(self):
        otp_enc = self.ea.verify_device(self.v_id, self.bs_k)
        otp = xor_encrypt_str(otp_enc, self.q_k2)  # decrypt
        if self.attempt_scenario == "device_fail":
            # Let's say we pretend OTP is wrong
            otp = "WRONG_OTP"
        if otp == "DEVICE_OTP":
            self.status = "Device Verified"
        else:
            self.status = "Device Verification Failed"

    def election_day_login(self):
        if self.status != "Device Verified":
            return
        q_k1_stored = self.ea.initiate_election_login(self.v_id)

        # Decrypt Q_K1
        enc_q_k1_bits = [int(x) for x in self.enc_q_k1]
        bs_k_bits = [int(x) for x in self.bs_k]
        length = min(len(enc_q_k1_bits), len(bs_k_bits))
        dec_q_k1_bits = [enc_q_k1_bits[i]^bs_k_bits[i] for i in range(length)]
        dec_q_k1 = "".join(str(b) for b in dec_q_k1_bits)

        if self.attempt_scenario == "qk1_mismatch":
            dec_q_k1 = "00000000" # Force mismatch

        if dec_q_k1 != q_k1_stored:
            self.status = "Q_K1 Mismatch - Login Failed"
            return

        self.q_k1 = dec_q_k1
        # Derive AQ_K1 via QKD
        s_A = [int(x) for x in self.q_k1]
        s_B = [int(x) for x in self.q_k1]
        aq_k1_bits = run_sedjo_qkd(num_epr=len(s_A), s_A=s_A, s_B=s_B)
        if aq_k1_bits is None:
            self.status = "QKD for AQ_K1 failed"
            return
        self.aq_k1 = "".join(str(b) for b in aq_k1_bits)

        # Verify identity with OTP
        otp_enc = self.ea.send_voting_otp(self.aq_k1)
        otp = xor_encrypt_str(otp_enc, self.aq_k1)
        if otp == "VOTING_OTP":
            self.status = "Fully Authenticated"
        else:
            self.status = "Voting OTP Verification Failed"

    def cast_vote(self):
        if self.status != "Fully Authenticated":
            return
        # Derive VQ_K1
        s_A = [int(x) for x in self.aq_k1]
        s_B = [int(x) for x in self.aq_k1]
        vq_k1_bits = run_sedjo_qkd(num_epr=len(s_A), s_A=s_A, s_B=s_B)
        if vq_k1_bits is None:
            self.status = "QKD for VQ_K1 failed"
            return
        self.vq_k1 = "".join(str(b) for b in vq_k1_bits)

        # Decrypt ballot ID
        enc_b_id, c_c = self.ea.send_ballot_id(self.vq_k1)
        b_id = xor_encrypt_str(enc_b_id, self.vq_k1)

        # Decide to vote for Candidate_A
        vote = "Candidate_A"
        enc_c_c = self.ea.send_confirmation_code(self.vq_k1, c_c)
        c_c_dec = xor_encrypt_str(enc_c_c, self.vq_k1)

        # Record vote
        self.ea.record_vote(b_id, c_c_dec, vote)
        self.vote_cast = vote
        self.status = "Voted Successfully"

##############################
# Run Multiple Test Cases
##############################
if __name__ == "__main__":
    logger.info("=== Quantegrity E-Voting Simulation with Multiple Test Cases ===")

    ea = ElectionAuthority()

    # Define scenarios
    # Each entry: (national_id, attempt_scenario)
    # scenario: "success", "qk1_mismatch", "device_fail"
    scenarios = [
        ("NID12345", "success"),
        ("NID54321", "qk1_mismatch"),
        ("NID99999", "device_fail"),
        ("NID22222", "success")
    ]

    voters = []
    for nid, scenario in scenarios:
        v = Voter(ea, nid, scenario)
        v.registration()
        v.device_registration()
        v.election_day_login()
        v.cast_vote()
        voters.append(v)

    # Run mixnet and tally
    ea.mixnet.run_mixnet()
    P, Q, R, S = ea.mixnet.generate_tables()

    # Print final tables
    logger.info("\n=== Final Scantegrity-like Tables ===")
    logger.info("P Table (Original):")
    logger.info("Index | Ballot_ID  | Confirmation_Code")
    for p in P:
        # p = (index, b_id, c_c)
        logger.info(f"{p[0]}     | {p[1]}    | {p[2]}")

    logger.info("\nQ Table (Shuffled Codes):")
    logger.info("Index | Confirmation_Code")
    for q in Q:
        # q = (index, c_c)
        logger.info(f"{q[0]}     | {q[1]}")

    logger.info("\nR Table (Mappings Q->P):")
    logger.info("Q_Index | P_Index")
    for r_ in R:
        logger.info(f"{r_[0]}       | {r_[1]}")

    logger.info("\nS Table (Final Tally):")
    logger.info("Candidate     | Votes")
    for s_ in S:
        logger.info(f"{s_[0]} | {s_[1]}")

    # Print voters table
    logger.info("\n=== Voters Summary Table ===")
    logger.info("NID      | V_ID     | Status                | Q_K1    | AQ_K1    | VQ_K1    | Vote")
    for v in voters:
        logger.info(f"{v.national_id} | {v.v_id} | {v.status:20} | {v.q_k1} | {v.aq_k1} | {v.vq_k1} | {v.vote_cast}")

    logger.info("\nSimulation complete.")
