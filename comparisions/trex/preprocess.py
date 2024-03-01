import os
import re
import json
import logging
import argparse

import lief
import magic
import capstone


def pretokenize(s):
    # Taken directly from the Trex source code.

    s = s.replace(",", " , ")
    s = s.replace("[", " [ ")
    s = s.replace("]", " ] ")
    s = s.replace(":", " : ")
    s = s.replace("*", " * ")
    s = s.replace("(", " ( ")
    s = s.replace(")", " ) ")
    s = s.replace("{", " { ")
    s = s.replace("}", " } ")
    s = s.replace("#", "")
    s = s.replace("$", "")
    s = s.replace("!", " ! ")

    s = re.sub(r"-(0[xX][0-9a-fA-F]+)", r"- \1", s)
    s = re.sub(r"-([0-9a-fA-F]+)", r"- \1", s)

    s = re.sub(r"0[xX][0-9a-fA-F]+", "hexvar", s)

    return s.split()


def preprocess(instructions):
    pretokens = []
    iposition = []
    aposition = []

    for i, instruction in enumerate(instructions):
        logging.debug(
            f"0x{instruction.address:x}\t{instruction.mnemonic} {instruction.op_str}"
        )

        p = pretokenize(f"{instruction.mnemonic} {instruction.op_str}")

        pretokens += p
        iposition += [i] * len(p)
        aposition += list(range(len(p)))

    return {
        "static": pretokens,
        "instruction": iposition,
        "argument": aposition,
    }


def parse(binary):
    with magic.Magic() as m:
        filemagic = m.id_filename(binary)

    if "x86-64" in filemagic:
        arch = capstone.CS_ARCH_X86
        mode = capstone.CS_MODE_64
    elif "aarch64" in filemagic:
        arch = capstone.CS_ARCH_ARM64
        mode = capstone.CS_MODE_ARM
    elif "MIPS64" in filemagic:
        arch = capstone.CS_ARCH_MIPS
        mode = capstone.CS_MODE_MIPS64
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
        instructions = list(engine.disasm(content, function.address))

        if len(instructions) == 0:
            logging.debug(
                f"failed to disassemble: {function.name}(0x{function.address:x})"
            )
            continue

        if function.name:
            label = f"0x{function.address:x}:{function.name}"
        else:
            label = hex(function.address)

        functions[label] = preprocess(instructions)

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
