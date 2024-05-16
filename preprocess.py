import os
import re
import pickle
import logging
import argparse

import magic
import pyhidra
import capstone

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

pyhidra.start()

from ghidra.program.model.block import BasicBlockModel
from ghidra.util.task import TaskMonitor

def x86(instruction):
    pretokens = []
    token_lpes = []

    # In most cases this is a single mnemonic, but can be two in the case
    # of special prefixes - e.g., REP
    mnemonics = instruction.getMnemonicString().split()
    for mnemonic in mnemonics:
        pretokens.append(mnemonic)
        ##token_lpes.append(function_lpe)
        #token_lpes.append(token_pe(function_lpe, idx))

    if instruction.getNumOperands():
        operands = []
        for i in range(instruction.getNumOperands()):
            operand = instruction.getDefaultOperandRepresentation(i)
            operands += operand.split(",")
        #print(operands)

        for operand in operands:
            # Immediate values
            if operand.startswith("0x") or operand.startswith("-0x"):
                # Convert to relative offset for jumps
                # if capstone.CS_GRP_JUMP in instruction.groups:
                #    offset = int(operand, 16) - instruction.address
                #    operand = hex(offset)

                operand = str(int(operand, 16))
                pretokens.append(operand)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Memory addresses
            elif "[" in operand:
                # Optional size directives
                if "ptr" in operand:
                    size, _, operand = operand.split(maxsplit=2)
                    pretokens.append(size)
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                # Optional segment indicators
                if ":" in operand:
                    segment, operand = operand.split(":")
                    pretokens.append(segment)
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                operand = operand[1:-1]
                pretokens.append("[")
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                split = re.split(r"(\+|-|\*)", operand)
                split = [o.strip() for o in split]

                for op in split:
                    if op.startswith("0x") or op.startswith("-0x"):
                        op = str(int(op, 16))

                    pretokens.append(op)
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                pretokens.append("]")
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Everything else should be a register
            else:
                pretokens.append(operand)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

    #return {"pretokens": pretokens, "token_lpes": token_lpes}
    return pretokens


def arm(instruction):
    pretokens = []
    token_lpes = []

    pretokens.append(instruction.getMnemonicString())
    #token_lpes.append(function_lpe)
    #token_lpes.append(token_pe(function_lpe, idx))

    if instruction.getNumOperands():
        operands = []
        for i in range(instruction.getNumOperands()):
            operand = instruction.getDefaultOperandRepresentation(i)
            operands += operand.replace(" ", "").split(",")
        #print(operands)

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
                        #token_lpes.append(function_lpe)
                        #token_lpes.append(token_pe(function_lpe, idx))

                    if operand.startswith("#"):
                        operand = str(int(operand[1:], 16))

                    pretokens.append(operand)
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                    pretokens.append("]")
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                    if preindex:
                        pretokens.append("!")
                        #token_lpes.append(function_lpe)
                        #token_lpes.append(token_pe(function_lpe, idx))

                    expecting = False
                else:
                    pretokens.append(operand)
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
            # Offset syntax
            elif "[" in operand:
                pretokens.append("[")
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
                if "]" in operand:
                    pretokens.append(operand[1:-1])
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                    pretokens.append("]")
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                else:
                    pretokens.append(operand[1:])
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))
                    pretokens.append("+")
                    #token_lpes.append(function_lpe)
                    #token_lpes.append(token_pe(function_lpe, idx))

                    expecting = True
            # Immediate values:
            elif operand.startswith("#"):
                try:
                    operand = str(int(operand[1:], 16))
                except ValueError:
                    operand = str(float(operand[1:]))

                pretokens.append(operand)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Shifted immediate values
            elif " " in operand:
                shift, operand = operand.split()
                pretokens.append(shift)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                if operand.startswith("#"):
                    operand = str(int(operand[1:], 16))

                pretokens.append(operand)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

            # Everything else should be a register
            else:
                pretokens.append(operand)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

        assert expecting is False

    #return {"pretokens": pretokens, "token_lpes": token_lpes}
    return pretokens


def mips(instruction):
    pretokens = []
    token_lpes = []

    pretokens.append(instruction.getMnemonicString())
    #token_lpes.append(function_lpe)
    #token_lpes.append(token_pe(function_lpe, idx))
    
    if instruction.getNumOperands():
        operands = []
        for i in range(instruction.getNumOperands()):
            operand = instruction.getDefaultOperandRepresentation(i)
            operands += operand.split(",")
        #print(operands)

        for operand in operands:
            # Offset syntax
            if "(" in operand:
                offset, operand = operand.split("(")

                pretokens.append("[")
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                assert operand[-1] == ")"

                pretokens.append(operand[1:-1])
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
                pretokens.append("+")
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                if offset.startswith("0x") or offset.startswith("-0x"):
                    offset = str(int(offset, 16))

                pretokens.append(offset)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

                pretokens.append("]")
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Immediate value
            elif operand.startswith("0x") or operand.startswith("-0x"):
                operand = str(int(operand, 16))
                pretokens.append(operand)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))
            # Everything else should be a trivial value (register or base 10 immediate)
            else:
                if operand[0] == "$":
                    operand = operand[1:]

                pretokens.append(operand)
                #token_lpes.append(function_lpe)
                #token_lpes.append(token_pe(function_lpe, idx))

    #return {"pretokens": pretokens, "token_lpes": token_lpes}
    return pretokens


def build_control_flow_graph(function, parser):
    monitor = TaskMonitor.DUMMY

    program = function.getProgram()
    listing = program.getListing()
    model = BasicBlockModel(program)

    def process(block, graph=None):
        logger.debug(f"processing block: {block}")

        graph = graph or nx.Graph()

        node = []
        tokens = []
        instructions = listing.getInstructions(block, True)
        while instructions.hasNext():
            instruction = instructions.next()
            logger.debug(f"instruction: {instruction.getAddress()} {instruction}")
            #print(instruction)
            #print(instruction.getMnemonicString())
            #for i in range(instruction.getNumOperands()):
                #print(instruction.getDefaultOperandRepresentation(i))
            pretokens = parser(instruction)
            pretokens.append("[NEXT]")
            #tokens += pretokens
            tokens.append(f"{instruction}")
            #print(pretokens)
            #node.append(f"{instruction}")
            node += pretokens
        #print(tokens)
        
        #if tokens and tokens[-1] == "[NEXT]":
            #tokens.pop()
            #print(tokens)
        #print("".join(tokens))

        #print(node)
        #node = "\n".join(node)
        node = ", ".join(node)
        #print(node)
        #print(tokens)

        if graph.has_node(node):
            logger.debug(f"reached repeated block: {block}")
            return graph, node

        graph.add_node(node)

        destinations = block.getDestinations(monitor)
        while destinations.hasNext():
            next = destinations.next()

            if listing.getFunctionAt(next.getDestinationAddress()):
                # If the next block is the start of a function then we've
                # exited the current function.

                continue

            next = next.getDestinationBlock()

            graph, next = process(next, graph)
            graph.add_edge(node, next)

        return graph, node

    address = function.getEntryPoint()
    block = model.getCodeBlockAt(address, monitor)

    graph, _ = process(block)

    return graph


def parse(binary, path):
    with magic.Magic() as m:
        filemagic = m.id_filename(binary)

    if "x86-64" in filemagic:
        arch = capstone.CS_ARCH_X86
        mode = capstone.CS_MODE_64
        parser = x86
    elif "80386" in filemagic and "32-bit" in filemagic:
        arch = capstone.CS_ARCH_X86
        mode = capstone.CS_MODE_32
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


    with pyhidra.open_program(binary, project_location=path) as api:
        program = api.getCurrentProgram()
        listing = program.getListing()

        functions = {}
        for function in listing.getFunctions(True):
            if function.isExternal() or function.isThunk():
                #logger.info(f"skipping external or thunk function {function}")
                continue
            #if function.getName().startswith('FUN_'): #skip unnamed function
                #continue

            #logger.info(f"processing {function}")

            graph = build_control_flow_graph(function, parser)

            identifier = (
                function.getName(),
                hex(function.getEntryPoint().getOffset()),
            )
            functions[identifier] = graph
        
    return functions


def laplacian_pe(g):
    laplacian = nx.normalized_laplacian_matrix(g).toarray() #Build the graph Laplacian matrix
    eigvals, eigvecs = np.linalg.eig(laplacian) #Compute the eigenvectors for the Laplacian matrix
    index = eigvals.argsort()
    eigvals, eigvecs = eigvals[index], np.real(eigvecs[:,index]) #Sorted by eigenvalue

    if eigvecs.shape[1] > 1: #returns all eigenvectors beyond first eigenvector
        return eigvecs[:, 1:dim+1] #remove first column corresponding to trivial eigenvec
    else: #returns first eigenvector
        #return eigvecs.tolist() #might do something different here for functions that have only 1 node
        return np.zeros((eigvecs.shape[0], dim))

dim = 4 #placeholder, but probably want to get user input for this
def pad_lpe(function_lpe): #pad according to the lpe dimensions we chose
    #pad_width = dim - len(function_lpe[0])
    pad_width = dim - len(function_lpe)
    #padded_function_lpe = np.pad(function_lpe, ((0, 0), (0, pad_width)), 'constant')
    for _ in range(0, pad_width):
        function_lpe = np.append(function_lpe, 0)
    return function_lpe


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
        functions = parse(binary, arguments.output)
        
        name = os.path.normpath(binary).split(os.path.sep)
        name = "-".join(name[-arguments.path_parts :])

        path = os.path.join(arguments.output, name)
        path = f"{path}.pickle"
        
        with open(path, "wb") as f:
            pickle.dump(functions, f)
            
        logging.info(f"processed {binary} ({path})")

    logging.info("done")