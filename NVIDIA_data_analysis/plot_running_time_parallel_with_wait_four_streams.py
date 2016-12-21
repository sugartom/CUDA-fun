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

class timestamp():
	def __init__(self, start, end):
		self.start = start
		self.end = end

f = open("./data/parallel_wait_event_four_streams/no_multiplication_0.01ms.txt", "r").readlines()

fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(111)

timestamp_dict = dict()
timestamp_dict["13"] = []
timestamp_dict["14"] = []
timestamp_dict["15"] = []
timestamp_dict["16"] = []

# tmp = f.readline().rstrip().split()
# # dealing with useless kernel
# timestamp_dict[tmp[17]].append(timestamp(parseTime(tmp[0]), parseTime(tmp[1])))

# # dealing with kernels in four streams
# for i in range(20):
# 	tmp = f.readline().rstrip().split()
# 	timestamp_dict[tmp[17]].append(timestamp(parseTime(tmp[0]), parseTime(tmp[1])))

for i in range(len(f)):
	tmp = f[i].rstrip().split()
	start = parseTime(tmp[0])
	end = start + parseTime(tmp[1])
	timestamp_dict[tmp[17]].append(timestamp(start, end))


for i in range(4):
	stream_index = str(13 + i)
	# x = []
	y = [4 - i, 4 - i]
	for j in range(len(timestamp_dict[stream_index])):
		ts = timestamp_dict[stream_index][j]
		x = []
		x.append(ts.start)
		x.append(ts.end)
		plt.plot(x, y, "*r-", label = "stream%d" % (1 + i))

axes = plt.gca()
axes.set_ylim([0.5, 4.5])

# plt.legend()
plt.xlabel("time (ms)")
plt.ylabel("stream index")

# plt.xscale('log')
plt.savefig('fig/plot_running_time_parallel_with_wait_four_streams_no_multiplication_0.01ms.png')
plt.show()
