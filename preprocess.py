import os
import re
import json
import logging
import argparse

import magic
import capstone

import angr
from angrutils import *
import networkx as nx
import numpy as np

def x86(instruction, idx, function_lpe):
    pretokens = []
    token_lpes = []

    # In most cases this is a single mnemonic, but can be two in the case
    # of special prefixes - e.g., REP
    mnemonics = instruction.mnemonic.split()
    for mnemonic in mnemonics:
        pretokens.append(mnemonic)
        token_lpes.append(function_lpe)
        #token_lpes.append(token_pe(function_lpe, idx))

    if instruction.op_str:
        operands = instruction.op_str.split(", ")

        for operand in operands:
            # Immediate values
            if operand.startswith("0x") or operand.startswith("-0x"):
                # Convert to relative offset for jumps
                # if capstone.CS_GRP_JUMP in instruction.groups:
                #    offset = int(operand, 16) - instruction.address
                #    operand = hex(offset)

                operand = str(int(operand, 16))
                pretokens.append(operand)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Memory addresses
            elif "[" in operand:
                # Optional size directives
                if "ptr" in operand:
                    size, _, operand = operand.split(maxsplit=2)
                    pretokens.append(size)
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                # Optional segment indicators
                if ":" in operand:
                    segment, operand = operand.split(":")
                    pretokens.append(segment)
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                operand = operand[1:-1]
                pretokens.append("[")
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                split = re.split(r"(\+|-|\*)", operand)
                split = [o.strip() for o in split]

                for op in split:
                    if op.startswith("0x") or op.startswith("-0x"):
                        op = str(int(op, 16))

                    pretokens.append(op)
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                pretokens.append("]")
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Everything else should be a register
            else:
                pretokens.append(operand)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

    return {"pretokens": pretokens, "token_lpes": token_lpes}


def arm(instruction, idx, function_lpe):
    pretokens = []
    token_lpes = []

    pretokens.append(instruction.mnemonic)
    token_lpes.append(function_lpe)
    #token_lpes.append(token_pe(function_lpe, idx))

    if instruction.op_str:
        operands = instruction.op_str.split(", ")

        expecting = False
        for operand in operands:
            # Continued offset syntax (see below)
            if expecting:
                if "]" in operand:
                    if operand[-1] == "!":
                        preindex = True
                        operand = operand[:-2]
                    else:
                        preindex = False
                        operand = operand[:-1]

                    # Register shifted offset
                    if " " in operand:
                        shift, operand = operand.split()
                        pretokens.append(shift)
                        token_lpes.append(function_lpe)
                        #token_lpes.append(token_pe(function_lpe, idx))

                    if operand.startswith("#"):
                        operand = str(int(operand[1:], 16))

                    pretokens.append(operand)
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                    pretokens.append("]")
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                    if preindex:
                        pretokens.append("!")
                        token_lpes.append(function_lpe)
                        #token_lpes.append(token_pe(function_lpe, idx))

                    expecting = False
                else:
                    pretokens.append(operand)
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
            # Offset syntax
            elif "[" in operand:
                pretokens.append("[")
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
                if "]" in operand:
                    pretokens.append(operand[1:-1])
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                    pretokens.append("]")
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                else:
                    pretokens.append(operand[1:])
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                    pretokens.append("+")
                    token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                    expecting = True
            # Immediate values:
            elif operand.startswith("#"):
                try:
                    operand = str(int(operand[1:], 16))
                except ValueError:
                    operand = str(float(operand[1:]))

                pretokens.append(operand)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Shifted immediate values
            elif " " in operand:
                shift, operand = operand.split()
                pretokens.append(shift)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                if operand.startswith("#"):
                    operand = str(int(operand[1:], 16))

                pretokens.append(operand)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

            # Everything else should be a register
            else:
                pretokens.append(operand)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

        assert expecting is False

    return {"pretokens": pretokens, "token_lpes": token_lpes}


def mips(instruction, idx, function_lpe):
    pretokens = []
    token_lpes = []

    pretokens.append(instruction.mnemonic)
    token_lpes.append(function_lpe)
    #token_lpes.append(token_pe(function_lpe, idx))

    if instruction.op_str:
        operands = instruction.op_str.split(", ")

        for operand in operands:
            # Offset syntax
            if "(" in operand:
                offset, operand = operand.split("(")

                pretokens.append("[")
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                assert operand[-1] == ")"

                pretokens.append(operand[1:-1])
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
                pretokens.append("+")
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                if offset.startswith("0x") or offset.startswith("-0x"):
                    offset = str(int(offset, 16))

                pretokens.append(offset)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                pretokens.append("]")
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Immediate value
            elif operand.startswith("0x") or operand.startswith("-0x"):
                operand = str(int(operand, 16))
                pretokens.append(operand)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Everything else should be a trivial value (register or base 10 immediate)
            else:
                if operand[0] == "$":
                    operand = operand[1:]

                pretokens.append(operand)
                token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

    return {"pretokens": pretokens, "token_lpes": token_lpes}


def preprocess(instructions, idx, function_lpe, parser):
    pretokens = []
    token_lpes = []

    for instruction in instructions:
        logging.debug(f"0x{instruction.address:x}\t{instruction.mnemonic} {instruction.op_str}")

        instruction_parsed = parser(instruction, idx, function_lpe)
        pretokens += instruction_parsed["pretokens"]
        token_lpes += instruction_parsed["token_lpes"]

        pretokens.append("[NEXT]")
        token_lpes.append(function_lpe) #we need all tokens to have the same pe
        #token_lpes.append(np.zeros(len(function_lpe)).tolist())
        #if len(function_lpe[0]): #non-single element lpe
            #token_lpes.append(np.zeros(len(function_lpe[0])).tolist())  #lpe for next token is 0 for all dimensions
        #else: #single element lpe
            #token_lpes.append(np.zeros(1).tolist())  #lpe is 0 for just one dimension
        
    return {"pretokens": pretokens, "token_lpes": token_lpes}


def parse(binary, output_path):
    with magic.Magic() as m:
        filemagic = m.id_filename(binary)

    if "x86-64" in filemagic:
        arch = capstone.CS_ARCH_X86
        mode = capstone.CS_MODE_64
        parser = x86
    elif "aarch64" in filemagic:
        arch = capstone.CS_ARCH_ARM64
        mode = capstone.CS_MODE_ARM
        parser = arm
    elif "MIPS64" in filemagic:
        arch = capstone.CS_ARCH_MIPS
        mode = capstone.CS_MODE_MIPS64
        parser = mips
    else:
        raise Exception(f"unsupported file type: {filemagic}")

    p = angr.Project(binary, load_options={'auto_load_libs':False})
    cfg =  p.analyses.CFGFast(normalize = True)

    with open(output_path, "w") as f:

        for function in cfg.functions.values():
            label = function.name

            if function.size == 0:
                logging.debug(f"skipping empty function: {function.name}(0x{function.addr:x})")
                continue

            logging.debug(f"function: {function.name}(0x{function.addr:x})")
            g = function.transition_graph.to_undirected()
            node_list = g.nodes()
            eigvecs = laplacian_pe(g)
            eigvec_norm = np.linalg.norm(eigvecs, axis=0).tolist() #get vector norm of eigenvectors
            function_lpe = pad_lpe(eigvec_norm)

            pretokens = []
            token_lpes = []
            for block in function.blocks:
                idx = get_node_index(hex(block.addr), node_list)
                instructions = block.capstone.insns

                block_preprocessed = preprocess(instructions, idx, function_lpe, parser)
                pretokens += block_preprocessed["pretokens"]
                token_lpes += block_preprocessed["token_lpes"]  #up to here so far

            
            if pretokens and pretokens[-1] == "[NEXT]":
                pretokens.pop()
                token_lpes.pop()
            
            json.dump({"function": label,
                       "pretokens": pretokens,
                       "position_ids": token_lpes}, f)
            f.write("\n") #output needs to be in jsonlines format


def laplacian_pe(g):
    laplacian = nx.normalized_laplacian_matrix(g).toarray()
    eigvals, eigvecs = np.linalg.eig(laplacian)
    index = eigvals.argsort()
    eigvals, eigvecs = eigvals[index], np.real(eigvecs[:,index])

    if eigvecs.shape[1] > 1: #returns all eigenvectors beyond first eigenvector
        return eigvecs[:, 1:dim+1] #remove first column corresponding to trivial eigenvec
    else: #returns first eigenvector
        #return eigvecs.tolist() #might do something different here for functions that have only 1 node
        return np.zeros(eigvecs.shape).tolist()

dim = 4 #placeholder, but probably want to get user input for this
def pad_lpe(function_lpe): #pad according to the lpe dimensions we chose
    #pad_width = dim - len(function_lpe[0])
    pad_width = dim - len(function_lpe)
    #padded_function_lpe = np.pad(function_lpe, ((0, 0), (0, pad_width)), 'constant')
    padded_function_lpe = function_lpe + ([0]*pad_width)
    return padded_function_lpe


def get_node_index(block_addr, node_list):
    for index, node in enumerate(node_list):
        if block_addr == hex(node.addr):
            return index


#def token_pe(function_lpe, idx):
    #if len(function_lpe) > 1: #function has more than 1 eigenvector
        #return function_lpe[idx,:].tolist()
    #else: #function has only the trivial eigenvector
        #return function_lpe[idx].tolist()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="preprocess binaries into texts that can be tokenized"
    )

    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("binaries", nargs="+", help="binaries to preprocess")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="run in verbose mode",
    )

    parser.add_argument(
        "-p",
        "--path-parts",
        type=int,
        default=1,
        help="the number of file path components to include in output file naming",
    )

    arguments = parser.parse_args()

    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if arguments.verbose else logging.INFO
    )

    logging.info(f"preprocessing {len(arguments.binaries)} binaries...")

    os.makedirs(arguments.output, exist_ok=True)

    for binary in arguments.binaries:
        name = os.path.normpath(binary).split(os.path.sep)
        name = "-".join(name[-arguments.path_parts :])

        path = os.path.join(arguments.output, name)
        path = f"{path}.json"

        parse(binary, path)
        logging.info(f"processed {binary} ({path})")

    logging.info("done")