
import pandas as pd


def readf(filename):
	return pd.read_csv(filename, sep=",")


filenames = [
"invocations_per_function_md.anon.d01.csv",
"invocations_per_function_md.anon.d02.csv",
"invocations_per_function_md.anon.d03.csv",
"invocations_per_function_md.anon.d04.csv",
"invocations_per_function_md.anon.d05.csv",
"invocations_per_function_md.anon.d06.csv",
"invocations_per_function_md.anon.d07.csv",
"invocations_per_function_md.anon.d08.csv",
"invocations_per_function_md.anon.d09.csv",
"invocations_per_function_md.anon.d10.csv",
"invocations_per_function_md.anon.d11.csv",
"invocations_per_function_md.anon.d12.csv",
"invocations_per_function_md.anon.d13.csv",
"invocations_per_function_md.anon.d14.csv"
]

columns = ["%d" % x for x in range(1,1441)]

def printrate(filename):
	df = readf(filename)
	df["total_requests"] = df[columns].sum(axis=1)
	rate = df.total_requests.sum() / (24 * 60 * 60)
	print("%s   %d workloads, average rate %f" % (filename, len(df), rate))


for filename in filenames:
	printrate(filename)

