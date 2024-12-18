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
    level=logging.ERROR,  # Default to ERROR to suppress internal logs
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Suppress lower-level logs from NetSquid, NetQASM, and SquidASM
logging.getLogger("netsquid").setLevel(logging.WARNING)
logging.getLogger("netqasm").setLevel(logging.WARNING)
logging.getLogger("squidasm").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Use INFO for demonstration logs

##############################
# QRNG Service (Simulated)
##############################
class QRNGService:
    @staticmethod
    def generate_random_bits(n: int) -> List[int]:
        return [random.randint(0,1) for _ in range(n)]

    @staticmethod
    def generate_random_str(n: int) -> str:
        # Return a binary string of length n
        return "".join(str(b) for b in QRNGService.generate_random_bits(n))

##############################
# Simple XOR-based "Encryption"
# (NOT SECURE, for DEMO only)
##############################
def xor_encrypt(message: str, key_bits: List[int]) -> List[int]:
    cipher_bits = []
    for i, ch in enumerate(message):
        ch_val = ord(ch)
        key_bit = key_bits[i % len(key_bits)]
        ch_val_xor = ch_val ^ key_bit
        char_bits = [(ch_val_xor >> b) & 1 for b in range(8)]
        cipher_bits.extend(char_bits)
    return cipher_bits

def xor_decrypt(cipher_bits: List[int], key_bits: List[int]) -> str:
    message = []
    chunk_size = 8
    for i in range(0, len(cipher_bits), chunk_size):
        char_bits = cipher_bits[i:i+chunk_size]
        ch_val_xor = 0
        for b, bit in enumerate(char_bits):
            ch_val_xor |= (bit << b)
        idx = (i // chunk_size) % len(key_bits)
        key_bit = key_bits[idx]
        ch_val = ch_val_xor ^ key_bit
        message.append(chr(ch_val))
    return "".join(message)

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

        # Apply s_A to EPR qubits
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

##############################
# QKD Helper Function
##############################
def run_sedjo_qkd(num_epr: int, s_A: Optional[List[int]]=None, s_B: Optional[List[int]]=None) -> Optional[List[int]]:
    # Run a QKD session (SEDJO) and return the final key from Alice's perspective.
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
# Election Authority (EA) and Voter (Alice) Simulation
##############################
class ElectionAuthority:
    def __init__(self):
        self.database = {}

    def register_voter(self, national_id: str) -> Tuple[str, str, str, str]:
        # Generate BS_K, Q_K1, Q_K2 using QRNG
        bs_k = QRNGService.generate_random_str(8)
        q_k1 = QRNGService.generate_random_str(8)
        q_k2 = QRNGService.generate_random_str(8)
        v_id = "V" + QRNGService.generate_random_str(4)

        logger.info("=== Registration Phase ===")
        logger.info(f"National ID: {national_id}")
        logger.info(f"Generated BS_K={bs_k}, Q_K1={q_k1}, Q_K2={q_k2} (binary)")

        # Encrypt Q_K1 with BS_K and store on Voter ID card
        q_k1_bits = [int(x) for x in q_k1]
        bs_k_bits = [int(x) for x in bs_k]
        length = min(len(q_k1_bits), len(bs_k_bits))
        enc_q_k1_bits = [(q_k1_bits[i] ^ bs_k_bits[i]) for i in range(length)]
        enc_q_k1 = "".join(str(b) for b in enc_q_k1_bits)

        self.database[v_id] = {"BS_K": bs_k, "Q_K1": q_k1, "Q_K2": q_k2, "NationalID": national_id}
        logger.info(f"Voter {v_id} registered. Encrypted Q_K1 stored on ID card: {enc_q_k1}")

        return v_id, enc_q_k1, q_k2, bs_k

    def verify_device(self, v_id: str, bs_k: str) -> str:
        q_k2 = self.database[v_id]["Q_K2"]
        otp = "DEVICE_OTP"
        otp_enc = self._xor_str_with_key(otp, q_k2)
        logger.info("=== Device Registration ===")
        logger.info(f"Using Q_K2={q_k2} to encrypt DEVICE_OTP.")
        return otp_enc

    def initiate_election_login(self, v_id: str) -> str:
        q_k1 = self.database[v_id]["Q_K1"]
        logger.info("=== Election Day Login ===")
        logger.info(f"Retrieved Q_K1={q_k1} from EA database.")
        return q_k1

    def send_voting_otp(self, aq_k1: str) -> str:
        otp = "VOTING_OTP"
        logger.info(f"Encrypting VOTING_OTP with AQ_K1={aq_k1}")
        return self._xor_str_with_key(otp, aq_k1)

    def send_ballot_id(self, vq_k1: str) -> str:
        b_id = "Ballot_0001"
        logger.info(f"Encrypting Ballot ID={b_id} with VQ_K1={vq_k1}")
        return self._xor_str_with_key(b_id, vq_k1)

    def send_confirmation_code(self, vq_k1: str) -> str:
        c_c = "ConfirmationCodeABC"
        logger.info(f"Encrypting Confirmation Code={c_c} with VQ_K1={vq_k1}")
        return self._xor_str_with_key(c_c, vq_k1)

    def _xor_str_with_key(self, message: str, key_str: str) -> str:
        key_bits = [int(x) for x in key_str]
        enc = []
        for i, ch in enumerate(message):
            ch_val = ord(ch)
            bit = key_bits[i % len(key_bits)]
            ch_val_xor = ch_val ^ bit
            enc.append(chr(ch_val_xor))
        return "".join(enc)

class Voter:
    def __init__(self, ea: ElectionAuthority):
        self.ea = ea
        self.v_id = None
        self.enc_q_k1 = None
        self.q_k2 = None
        self.bs_k = None
        self.q_k1 = None

    def registration(self, national_id: str):
        self.v_id, self.enc_q_k1, self.q_k2, self.bs_k = self.ea.register_voter(national_id)
        logger.info(f"Voter received V_ID={self.v_id}, Q_K2={self.q_k2}, BS_K={self.bs_k}, Enc_Q_K1={self.enc_q_k1}")

    def device_registration(self):
        otp_enc = self.ea.verify_device(self.v_id, self.bs_k)
        otp = self._xor_str_decrypt(otp_enc, self.q_k2)
        logger.info(f"Device Registration: Decrypted OTP with Q_K2. OTP={otp}, Device Verified.")

    def election_day_login(self):
        stored_q_k1 = self.ea.initiate_election_login(self.v_id)
        logger.info(f"Decrypting Q_K1 from enc_Q_K1={self.enc_q_k1} using BS_K={self.bs_k}")

        # Decrypt Q_K1
        enc_q_k1_bits = [int(x) for x in self.enc_q_k1]
        bs_k_bits = [int(x) for x in self.bs_k]
        length = min(len(enc_q_k1_bits), len(bs_k_bits))
        dec_q_k1_bits = [enc_q_k1_bits[i]^bs_k_bits[i] for i in range(length)]
        dec_q_k1 = "".join(str(b) for b in dec_q_k1_bits)

        if dec_q_k1 != stored_q_k1:
            logger.error("Q_K1 mismatch! Authentication failed.")
            return None

        self.q_k1 = dec_q_k1
        logger.info(f"Q_K1 successfully decrypted: {self.q_k1}")
        logger.info("Running QKD to derive AQ_K1 (s_A=Q_K1, s_B=Q_K1)...")

        s_A = [int(x) for x in self.q_k1]
        s_B = [int(x) for x in self.q_k1]
        aq_k1_bits = run_sedjo_qkd(num_epr=len(s_A), s_A=s_A, s_B=s_B)
        if aq_k1_bits is None:
            logger.error("QKD failed for AQ_K1.")
            return None
        aq_k1 = "".join(str(b) for b in aq_k1_bits)
        logger.info(f"AQ_K1 derived: {aq_k1}")

        # Decrypt Voting OTP with AQ_K1
        otp_enc = self.ea.send_voting_otp(aq_k1)
        otp = self._xor_str_decrypt(otp_enc, aq_k1)
        logger.info(f"Voting OTP decrypted with AQ_K1={aq_k1}: OTP={otp}, Identity fully verified.")

        return aq_k1

    def cast_vote(self, aq_k1: str):
        logger.info("Deriving VQ_K1 via QKD (s_A=AQ_K1, s_B=AQ_K1)...")
        s_A = [int(x) for x in aq_k1]
        s_B = [int(x) for x in aq_k1]
        vq_k1_bits = run_sedjo_qkd(num_epr=len(s_A), s_A=s_A, s_B=s_B)
        if vq_k1_bits is None:
            logger.error("QKD failed for VQ_K1.")
            return
        vq_k1 = "".join(str(b) for b in vq_k1_bits)
        logger.info(f"VQ_K1 derived: {vq_k1}")

        # Decrypt Ballot ID
        enc_b_id = self.ea.send_ballot_id(vq_k1)
        b_id = self._xor_str_decrypt(enc_b_id, vq_k1)
        logger.info(f"Ballot ID retrieved: {b_id}")

        # Encrypt vote with VQ_K1
        vote = "Candidate_A"
        enc_vote = self._xor_str_encrypt(vote, vq_k1)
        logger.info(f"Vote '{vote}' encrypted with VQ_K1='{vq_k1}' -> '{enc_vote}'")

        # Decrypt confirmation code
        enc_c_c = self.ea.send_confirmation_code(vq_k1)
        c_c = self._xor_str_decrypt(enc_c_c, vq_k1)
        logger.info(f"Confirmation Code received and decrypted: {c_c}")

    def _xor_str_encrypt(self, message: str, key_str: str) -> str:
        key_bits = [int(x) for x in key_str]
        enc = []
        for i, ch in enumerate(message):
            ch_val = ord(ch)
            bit = key_bits[i % len(key_bits)]
            ch_val_xor = ch_val ^ bit
            enc.append(chr(ch_val_xor))
        return "".join(enc)

    def _xor_str_decrypt(self, cipher: str, key_str: str) -> str:
        return self._xor_str_encrypt(cipher, key_str)

##############################
# Main Demonstration
##############################
if __name__ == "__main__":
    logger.info("=== Quantegrity E-Voting Proof of Concept ===")

    ea = ElectionAuthority()
    voter = Voter(ea)

    # Registration
    logger.info("Starting Registration Phase...")
    voter.registration(national_id="NID12345")

    # Device Registration
    logger.info("Starting Device Registration Phase...")
    voter.device_registration()

    # Election Day Login
    logger.info("Starting Election Day Login Phase...")
    aq_k1 = voter.election_day_login()

    # Vote Casting
    if aq_k1 is not None:
        logger.info("Starting Vote Casting Phase...")
        voter.cast_vote(aq_k1)

    logger.info("Demonstration complete.")
