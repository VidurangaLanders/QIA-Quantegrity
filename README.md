# Quantegrity E-Voting Proof-of-Concept

This repository contains a proof-of-concept demonstration of the Quantegrity e-voting system integrated with the SEDJO QKD protocol and a simplified Scantegrity-like mixnet. It simulates voter registration, device verification, quantum key distribution (QKD) steps, vote casting, and final tallying with mixnet tables.

## Features
- **Multiple Test Cases:** Demonstrates both successful and unsuccessful attempts at various stages (device verification failure, Q_K1 mismatch, etc.).
- **SEDJO QKD Protocol:** Uses SquidASM and NetSquid-based simulation to run the SEDJO QKD key exchange and derive keys (Q_K1, AQ_K1, VQ_K1).
- **Scantegrity-Like Mixnet:** Implements a simplified version of the P, Q, R, and S tables to illustrate how ballots and confirmation codes are shuffled and tallied.
- **Comprehensive Logging:** Shows generated keys, steps taken, success/failure status, and final tally tables.

## Requirements
- Python 3.8+ recommended
- [NetSquid](https://netsquid.org) installed
- [SquidASM](https://github.com/QuTech-Delft/netqasm) environment set up
- (Optional) Virtual environment recommended

## Setup
1. **Install NetSquid and dependencies**: Follow instructions at [https://netsquid.org/installation/](https://netsquid.org/installation/).
2. **Install SquidASM**: Instructions can be found in the SquidASM repository. Make sure you have a working NetQASM and SquidASM environment.
3. **Clone or copy this repository**: Place `quantegrity_demo.py` and `README.md` in a suitable directory.

## Running the Program
From the directory containing `quantegrity_demo.py`, run:
```bash
python3 quantegrity_demo.py
```

## Running the Multi User Program
From the directory containing `quantegrity_demo_mult.py`, run:
```bash
python3 quantegrity_demo_mult.py
```

## Notes
- Focus has been on implementing the SEDJO protocol and verification process.
- The Scantegrity Mixnet system has not been properly implemented. Included is a simplified AI generated version of the Scantegrity Mixnet system.
