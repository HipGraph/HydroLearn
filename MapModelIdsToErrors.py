import glob
import os
import sys

if len(sys.argv) < 2:
    print("Give me checkpoint or evaluation directory!")
    quit()
directory = sys.argv[1]
if directory[-1] == os.sep:
    directory = directory[:-1]
configurations = []
paths = glob.glob(directory + os.sep + "*" + os.sep + "*Errors.txt")
for path in paths:
    fields = path.split(os.sep)
    print(fields)
    model_id = fields[-2]
    configurations.append("#######################")
    configurations.append("### Model ID: %-5s ###"%(model_id))
    configurations.append("#######################")
    with open(path, "r") as f:
        configuration = f.read()
    configurations.append(configuration)
    configurations.append("\n\n")
path = directory + os.sep + "ModelIdErrorMap.txt"
with open(path, "w") as f:
    f.write("\n".join(configurations))
