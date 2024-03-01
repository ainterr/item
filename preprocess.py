import os
import re
import json
import logging
import argparse

import lief
import magic
import capstone


def x86(instruction):
    pretokens = []

    # In most cases this is a single mnemonic, but can be two in the case
    # of special prefixes - e.g., REP
    mnemonics = instruction.mnemonic.split()
    for mnemonic in mnemonics:
        pretokens.append(mnemonic)

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
            # Memory addresses
            elif "[" in operand:
                # Optional size directives
                if "ptr" in operand:
                    size, _, operand = operand.split(maxsplit=2)
                    pretokens.append(size)
                # Optional segment indicators
                if ":" in operand:
                    segment, operand = operand.split(":")
                    pretokens.append(segment)

                operand = operand[1:-1]
                pretokens.append("[")

                split = re.split(r"(\+|-|\*)", operand)
                split = [o.strip() for o in split]

                for op in split:
                    if op.startswith("0x") or op.startswith("-0x"):
                        op = str(int(op, 16))

                    pretokens.append(op)

                pretokens.append("]")
            # Everything else should be a register
            else:
                pretokens.append(operand)

    return pretokens


def arm(instruction):
    pretokens = []

    pretokens.append(instruction.mnemonic)

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

                    if operand.startswith("#"):
                        operand = str(int(operand[1:], 16))

                    pretokens.append(operand)
                    pretokens.append("]")

                    if preindex:
                        pretokens.append("!")

                    expecting = False
                else:
                    pretokens.append(operand)
            # Offset syntax
            elif "[" in operand:
                pretokens.append("[")
                if "]" in operand:
                    pretokens.append(operand[1:-1])
                    pretokens.append("]")
                else:
                    pretokens.append(operand[1:])
                    pretokens.append("+")

                    expecting = True
            # Immediate values:
            elif operand.startswith("#"):
                try:
                    operand = str(int(operand[1:], 16))
                except ValueError:
                    operand = str(float(operand[1:]))

                pretokens.append(operand)
            # Shifted immediate values
            elif " " in operand:
                shift, operand = operand.split()
                pretokens.append(shift)

                if operand.startswith("#"):
                    operand = str(int(operand[1:], 16))

                pretokens.append(operand)

            # Everything else should be a register
            else:
                pretokens.append(operand)

        assert expecting is False

    return pretokens


def mips(instruction):
    pretokens = []

    pretokens.append(instruction.mnemonic)

    if instruction.op_str:
        operands = instruction.op_str.split(", ")

        for operand in operands:
            # Offset syntax
            if "(" in operand:
                offset, operand = operand.split("(")

                pretokens.append("[")

                assert operand[-1] == ")"

                pretokens.append(operand[1:-1])
                pretokens.append("+")

                if offset.startswith("0x") or offset.startswith("-0x"):
                    offset = str(int(offset, 16))

                pretokens.append(offset)

                pretokens.append("]")
            # Immediate value
            elif operand.startswith("0x") or operand.startswith("-0x"):
                operand = str(int(operand, 16))
                pretokens.append(operand)
            # Everything else should be a trivial value (register or base 10 immediate)
            else:
                if operand[0] == "$":
                    operand = operand[1:]

                pretokens.append(operand)

    return pretokens


def preprocess(instructions, parser):
    pretokens = []

    for instruction in instructions:
        logging.debug(
            f"0x{instruction.address:x}\t{instruction.mnemonic} {instruction.op_str}"
        )

        pretokens += parser(instruction)

        pretokens.append("[NEXT]")

    if pretokens and pretokens[-1] == "[NEXT]":
        pretokens.pop()

    return pretokens


def parse(binary):
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

    binary = lief.parse(binary)

    engine = capstone.Cs(arch, mode)
    engine.detail = True

    functions = {}
    for function in binary.functions:
        if function.size == 0:
            logging.debug(
                f"skipping empty function: {function.name}(0x{function.address:x})"
            )
            continue

        logging.debug(f"function: {function.name}(0x{function.address:x})")

        content = bytes(
            binary.get_content_from_virtual_address(function.address, function.size)
        )
        instructions = engine.disasm(content, function.address)

        if function.name:
            label = f"0x{function.address:x}:{function.name}"
        else:
            label = hex(function.address)

        functions[label] = preprocess(instructions, parser)

    # pretokens = set()
    # for function in functions.values():
    #    pretokens |= set(function)
    # nonnumbers = set()
    # for pretoken in pretokens:
    #    try:
    #        int(pretoken)
    #    except ValueError:
    #        nonnumbers.add(pretoken)
    # print(nonnumbers)

    return functions


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
        functions = parse(binary)

        name = os.path.normpath(binary).split(os.path.sep)
        name = "-".join(name[-arguments.path_parts :])

        path = os.path.join(arguments.output, name)
        path = f"{path}.json"

        with open(path, "w") as f:
            json.dump(functions, f)

        logging.info(f"processed {binary} ({path})")

    logging.info("done")
