import os

def job_config():
    config = {
        "maxEvents": 500000
    }
    return config

def submit_jobs(sample):
    job_path = f"job_{sample}"
    os.system(f"mkdir -p {job_path}")

    os.system(f"cp inputs/{sample}.txt {job_path}/inputs.txt")
    os.system(f"cp rivet.py {job_path}/rivet.py")
    os.system(f"cp condor.jds {job_path}/condor.jds")
    os.system(f"cp $(voms-proxy-info --path) {job_path}/MyProxy")
    os.system(f"cp setup.sh {job_path}/run.sh")

    maxEvents = job_config()["maxEvents"]
    os.system(f"sed -i 's|@@maxEvents@@|{maxEvents}|g' {job_path}/rivet.py")

    os.system(f"sed -i 's|@@JobBatchName@@|{sample}|g' {job_path}/condor.jds")
    os.system(f"sed -i 's|@@JobPath@@|{job_path}|g' {job_path}/condor.jds")

    os.system(f"condor_submit {job_path}/condor.jds")

def main():
    for sample in os.listdir("inputs"):
        sample = sample.split(".")[0]
        submit_jobs(sample)

main()

