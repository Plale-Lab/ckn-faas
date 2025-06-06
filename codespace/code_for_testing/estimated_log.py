# Read file and parse time values
with open("/home/exouser/ckn-faas/codespace/code_for_testing/shufflenet_est_time2.log", "r") as f:
    times = [float(line.strip()) for line in f if line.strip()]

# Compute and print cumulative sums
for i in range(len(times)):
    total = sum(times[i:])
    # print(f"Sum from line {i+1} to end: {total:.4f}")
    print(f"{total:.4f}")

# import numpy as np

# # Configuration
# input_file = "/home/exouser/ckn-faas/codespace/code_for_testing/shufflenet_est_time2.log"
# target_line_count = 271

# # Step 1: Read all lines
# with open(input_file, "r") as f:
#     lines = f.readlines()

# # Step 2: Count lines
# total_lines = len(lines)
# print(f"Total lines in file: {total_lines}")

# # Step 3: Downsample (evenly spaced selection)
# if target_line_count >= total_lines:
#     sampled_lines = lines  # no downsampling needed
# else:
#     indices = np.linspace(0, total_lines - 1, target_line_count, dtype=int)
#     sampled_lines = [lines[i] for i in indices]

# # Step 4: Output sampled lines
# for line in sampled_lines:
#     print(line.strip())