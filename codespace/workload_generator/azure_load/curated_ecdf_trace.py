# from dataset import join_day_one, ecdf_trace_row, write_trace
# from trace_analyze import run_trace_csv
# import os
# import random
# import argparse
# import multiprocessing as mp
#
# """
# Use ECDFs to generate a trace from a specified number of functions from a curated pool
# """
#
# argparser = argparse.ArgumentParser()
# argparser.add_argument("--out-folder", '-o', help="The folder to store the output csv files into", required=True)
# argparser.add_argument("--data-path", '-d', help="The folder where the azure dataset has been downloaded to", required=True)
# argparser.add_argument("--force", '-f', action='store_true', help="Overwrite an existing trace that has the same number of functions")
# argparser.add_argument("--num-funcs", '-n', type=int, help="Number of functions to sample for the trace", required=True)
# argparser.add_argument("--duration", type=int, help="The length in minutes of the trace", default=60)
# argparser.add_argument("--scale", '-s', type=float, help="Scale IATs by this factor to adjust load. Numbers less than one will increase load.", default=1.0)
# argparser.add_argument("--verbose", '-v', action='store_true', help="Enable debug printing")
#
# # args = argparser.parse_args()
#
# args = argparser.parse_args([
#     "--out-folder", "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/azure_load/generated_trace",
#     "--data-path", "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/azure_load",
#     "--num-funcs", "5",
#     "--duration", "10",
#     "--scale", "1.0",
#     "--force",
#     "--verbose"
# ])
#
# dataset = join_day_one(args.data_path, args.force, debug=args.verbose)
#
# functions = ["0a0fb019e32b06095378171cc091223d3456c87badd55156f93a668e919cfca7",
# "1d6068092b0b3d0ff9e214242ac2678bd147313b03b1ae641eca88286c8ebbf3",
# "3ef5de7d89a7336bcbf7273c387b4094cafa203cd618440d8ca2adeac8e21c75",
# "9cfbc77f8131488e0261499dde8c0bb67a6c8460b28230918bd06ba63695afdc",
# "45ce273e28e1910770ead3770647a92b058206cc91c5574ee10afa1cc158086d",
# "33429bc85e5fc1a07053b627f7055c0d37300da4db44aaa857299627ac3e9b5c",
# "a9783835cbd68a2e234cf861799cf840df9ce4055cca238f04978125c187a0cc",
# "ce5f8cab397532fec35076960d07b43868222da16d5c3e67fef1d2b7e11b26c6",
# "db8aebcc4103b79b973d3cdb82c0eb8eedbdad9c08773beb51d09fa40479e8b9",
# "e81ba93f8513b173c53d51521c4569f0c5e160c2dc9ed7fd2cefddcb81dc4d76",
# "771db0dde8fb0d2121086b7c686f564444a6ed2a9aec805b9a4849a7d5cd425f",
# "f282dab639947c11df7328899c9c0dabbf532af1e22e97eca695a532554813f0",
# "d74c3a3cb196fe5568444924a2f27abe1383c4eafa12a5f801e883779f899782",
# "bc03a348453e61b1e27057ded9672d3e3903d8c0eb64b927cb4580e36ac3a962"]
#
# trace = []
# function_metadata = []
# metadata_save_pth = os.path.join(args.out_folder, "metadata-chosen-ecdf.csv")
# trace_save_pth = os.path.join(args.out_folder, "chosen-ecdf.csv")
#
# os.makedirs(args.out_folder, exist_ok=True)
#
# if not (os.path.exists(metadata_save_pth) and os.path.exists(trace_save_pth)) or args.force:
#   map_args = []
#   for func in random.sample(functions, args.num_funcs):
#     rows = dataset[dataset["HashFunction"] == func]
#     for index, row in rows.iterrows():
#       map_args.append( (func[:10], row, args.duration, args.scale) )
#
#   pool = mp.Pool()
#   ret = pool.starmap(ecdf_trace_row, map_args)
#
#   for traced_row, (func_name, cold_dur, warm_dur, mem) in ret:
#     trace += traced_row
#     function_metadata.append((func_name, cold_dur, warm_dur, mem))
#
#   out_trace = sorted(trace, key=lambda x:x[1])
#
#   write_trace(out_trace, function_metadata, trace_save_pth, metadata_save_pth)
#
#   print("done", trace_save_pth)
#   print("warm_pct, max_mem, max_running, mean_running, running_75th, running_90th")
#   print(*run_trace_csv(trace_save_pth, 0.80, metadata_save_pth))



from dataset import join_day_one, ecdf_trace_row, write_trace
from trace_analyze import run_trace_csv

import argparse
import multiprocessing as mp
import os
import random


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--out-folder",
        "-o",
        required=True,
        help="The folder to store the output CSV files",
    )
    argparser.add_argument(
        "--data-path",
        "-d",
        required=True,
        help="Folder containing the Azure dataset",
    )
    argparser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing output",
    )
    argparser.add_argument(
        "--num-funcs",
        "-n",
        type=int,
        required=True,
        help="Number of functions to sample",
    )
    argparser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Trace duration in minutes",
    )
    argparser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Scale IATs; values below 1 increase load",
    )
    argparser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug printing",
    )

    # args = argparser.parse_args()


    args = argparser.parse_args([
        "--out-folder", "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/azure_load/generated_trace",
        "--data-path", "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/azure_load",
        "--num-funcs", "4",
        "--duration", "10",
        "--scale", "0.2",
        "--force",
        "--verbose"
    ])

    dataset = join_day_one(
        args.data_path,
        args.force,
        debug=args.verbose,
    )

    functions = [
        "0a0fb019e32b06095378171cc091223d3456c87badd55156f93a668e919cfca7",
        "1d6068092b0b3d0ff9e214242ac2678bd147313b03b1ae641eca88286c8ebbf3",
        "3ef5de7d89a7336bcbf7273c387b4094cafa203cd618440d8ca2adeac8e21c75",
        "9cfbc77f8131488e0261499dde8c0bb67a6c8460b28230918bd06ba63695afdc",
        "45ce273e28e1910770ead3770647a92b058206cc91c5574ee10afa1cc158086d",
        "33429bc85e5fc1a07053b627f7055c0d37300da4db44aaa857299627ac3e9b5c",
        "a9783835cbd68a2e234cf861799cf840df9ce4055cca238f04978125c187a0cc",
        "ce5f8cab397532fec35076960d07b43868222da16d5c3e67fef1d2b7e11b26c6",
        "db8aebcc4103b79b973d3cdb82c0eb8eedbdad9c08773beb51d09fa40479e8b9",
        "e81ba93f8513b173c53d51521c4569f0c5e160c2dc9ed7fd2cefddcb81dc4d76",
        "771db0dde8fb0d2121086b7c686f564444a6ed2a9aec805b9a4849a7d5cd425f",
        "f282dab639947c11df7328899c9c0dabbf532af1e22e97eca695a532554813f0",
        "d74c3a3cb196fe5568444924a2f27abe1383c4eafa12a5f801e883779f899782",
        "bc03a348453e61b1e27057ded9672d3e3903d8c0eb64b927cb4580e36ac3a962",
    ]

    if args.num_funcs > len(functions):
        raise ValueError(
            f"--num-funcs must be at most {len(functions)}"
        )

    trace = []
    function_metadata = []

    os.makedirs(args.out_folder, exist_ok=True)

    metadata_save_pth = os.path.join(
        args.out_folder,
        "metadata-chosen-ecdf-scal.csv",
    )
    trace_save_pth = os.path.join(
        args.out_folder,
        "chosen-ecdf-5scal.csv",
    )

    if (
        not (
            os.path.exists(metadata_save_pth)
            and os.path.exists(trace_save_pth)
        )
        or args.force
    ):
        map_args = []

        random.seed(42)

        selected_functions = random.sample(
            functions,
            args.num_funcs,
        )

        for func in selected_functions:
            rows = dataset[
                dataset["HashFunction"] == func
            ]

            for _, row in rows.iterrows():
                map_args.append(
                    (
                        func[:10],
                        row,
                        args.duration,
                        args.scale,
                    )
                )

        with mp.Pool() as pool:
            ret = pool.starmap(
                ecdf_trace_row,
                map_args,
            )

        for traced_row, metadata in ret:
            # func_name, cold_dur, warm_dur, mem = metadata
            func_name, cold_dur, warm_dur, mem, mean_iat_ms = metadata
            trace.extend(traced_row)
            # function_metadata.append(
            #     (
            #         func_name,
            #         cold_dur,
            #         warm_dur,
            #         mem,
            #     )
            function_metadata.append(
                (
                    func_name,
                    cold_dur,
                    warm_dur,
                    mem,
                    mean_iat_ms,
                )

            )

        out_trace = sorted(
            trace,
            key=lambda x: x[1],
        )

        write_trace(
            out_trace,
            function_metadata,
            trace_save_pth,
            metadata_save_pth,
        )

        print("done", trace_save_pth)
        print(
            "warm_pct, max_mem, max_running, "
            "mean_running, running_75th, running_90th"
        )

        print(
            *run_trace_csv(
                trace_save_pth,
                0.80,
                metadata_save_pth,
            )
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()