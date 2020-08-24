import os
import csv
import math
import argparse
import configparser

from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="Input CSV file with intermediate values")
parser.add_argument(
    "--cfg", help="Config file to specify platform speed profiles", default="ndp.cfg"
)
args = parser.parse_args()

assert os.path.exists(args.csv), "csv file given does not exist!"
assert os.path.exists(args.cfg), "Config file given does not exist!"
config = configparser.ConfigParser()
config.read(args.cfg)

postproc = config["POSTPROC"]

csv_data = []
with open(args.csv, "r") as csvfile:
    lines = csvfile.readlines()[1:]
    for line in lines:
        toks = [tok.strip() for tok in line.split(",")]
        csv_data.append((float(toks[0]), int(toks[2])))

for profile in postproc:
    results = {}
    profile_interval_secs = float(postproc[profile])
    if profile_interval_secs < 0:
        profile_interval_secs = math.inf
    print("Processing profile {} in intervals of {}s".format(profile, profile_interval_secs))

    curr_min = 0
    sample_counts = deepcopy(csv_data)
    while len(sample_counts) > 0:
        curr_max = curr_min + profile_interval_secs
        cache = []
        for timecountpair in sample_counts:
            timestamp, count = timecountpair
            if curr_min <= timestamp and timestamp < curr_max:
                cache.append(timecountpair)

        # results[profile][(curr_min, curr_max)] = float(sum( [count for _,count in cache] )) / len( cache ) if len(cache) > 0 else 0
        results[(curr_min, curr_max)] = (
            float(sum([count for _, count in cache])) / len(cache) if len(cache) > 0 else 0
        )

        sample_counts = [x for x in sample_counts if x not in cache]

        curr_min = curr_max

    dst_csv_name = args.csv.replace("_intermediate.csv", "_{}.csv".format(profile))
    # dst_csv_name = '{}_{}.csv'.format( os.path.splitext(os.path.basename(args.csv))[0] ,profile )
    with open(dst_csv_name, "w") as csv_out:
        writer = csv.writer(csv_out, delimiter=",")
        writer.writerow(
            [
                "IntervalOf{}".format(
                    "Infinity" if math.isinf(profile_interval_secs) else int(profile_interval_secs)
                ),
                "AvgCount",
            ]
        )
        pairs = []
        for interval, avg in results.items():
            interval_index = (
                1
                if math.isinf(profile_interval_secs)
                else round(interval[1] / profile_interval_secs)
            )
            pairs.append((interval_index, round(avg, 2)))
        pairs = sorted(pairs, key=lambda x: x[0])
        for pair in pairs:
            writer.writerow(pair)
        writer.writerow(["Sum: ", round(sum(results.values()))])
    print("Profile: {} | Sum: {}".format(profile, round(sum(results.values()))))
# config parsing will read whatever is in [POSTPROC] where the keys will be in the csv title


# print out 2 numbers (total peeps for each profile)
# write out 2 csvs (each profile's csv giving the avrg count at each interval)
## videoname_timestamp_profilename.csv
## IntervalOf9 ; AvrgCount
##     1       ;   10.0
##     2       ;   13.1
