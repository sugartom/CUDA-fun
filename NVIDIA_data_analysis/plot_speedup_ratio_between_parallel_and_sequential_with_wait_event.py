import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

def parseTime(tt):
	unit = tt[-2:]
	if (unit == "us"):
		return float(tt[:-2]) * 0.001
	elif (unit == "ms"):
		return float(tt[:-2]) * 1
	else:
		return float(tt[:-1]) * 1000

def min(a, b):
	if (a < b):
		return a
	else:
		return b

def max(a, b):
	if (a < b):
		return b
	else:
		return a

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

input_list = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

parallel_duration = []
sequential_duration = []

for i in range(len(input_list)):
	f = open("./data/parallel_wait_event_1000/matmul_two_matrix_two_stream_matrix_size_%s.txt" % input_list[i], "r").readlines()

	if (i == 8 or i == 9 or i == 10):
		tmp = f[13].rstrip().split()
	else:
		tmp = f[11].rstrip().split()
	start_timestamp = parseTime(tmp[0])

	if (i == 8 or i == 9 or i == 10):
		tmp = f[411].rstrip().split()
	else:
		tmp = f[210].rstrip().split()
	end_timestamp = parseTime(tmp[0]) + parseTime(tmp[1])

	parallel_duration.append(end_timestamp - start_timestamp)

for i in range(len(input_list)):
	f = open("./data/sequential_wait_event_1000/matmul_two_matrix_two_stream_matrix_size_%s.txt" % input_list[i], "r").readlines()

	if (i == 8 or i == 9 or i == 10):
		tmp = f[13].rstrip().split()
	else:
		tmp = f[11].rstrip().split()
	start_timestamp = parseTime(tmp[0])

	if (i == 8 or i == 9 or i == 10):
		tmp = f[411].rstrip().split()
	else:
		tmp = f[210].rstrip().split()
	end_timestamp = parseTime(tmp[0]) + parseTime(tmp[1])

	sequential_duration.append(end_timestamp - start_timestamp)

speedup_ratio = []

for i in range(len(input_list)):
	speedup_ratio.append(sequential_duration[i] / parallel_duration[i])

print speedup_ratio

plt.plot(input_list, speedup_ratio, "b*-", label = "speedup ratio")

plt.legend()
plt.xlabel("matrix size")
plt.ylabel("speed up ratio (%)")

plt.xscale('log')
plt.savefig('fig/plot_speedup_ratio_between_parallel_and_sequential_with_wait_event_100.png')
plt.show()
