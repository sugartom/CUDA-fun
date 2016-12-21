import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

input_list = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

GFLOPS_wo_streams = []
GFLOPS_w_streams = []
GFLOPS_batched = []

for i in range(len(input_list)):
	f = open("./data/batchCUBLAS/output-%d.txt" % input_list[i], "r").readlines()
	GFLOPS_wo_streams.append(float(f[24].rstrip().split()[-1].split("=")[-1]))
	# print GFLOPS_wo_streams
	GFLOPS_w_streams.append(float(f[37].rstrip().split()[-1].split("=")[-1]))
	GFLOPS_batched.append(float(f[50].rstrip().split()[-1].split("=")[-1]))

for i in range(len(input_list)):
	plt.text(input_list[i], GFLOPS_batched[i] * 1.05, str(input_list[i]))

plt.plot(input_list, GFLOPS_wo_streams, "b*-", label = "GFLOPS_wo_streams")
plt.plot(input_list, GFLOPS_w_streams, "r*-", label = "GFLOPS_w_streams")
plt.plot(input_list, GFLOPS_batched, "g*-", label = "GFLOPS_batched")

plt.legend(loc='upper left')
plt.xlabel("matrix size")
plt.ylabel("GFLOPS")

plt.xscale('log')
plt.yscale('log')
plt.savefig('fig/plot_batchCUBLAS_s.png')
plt.show()