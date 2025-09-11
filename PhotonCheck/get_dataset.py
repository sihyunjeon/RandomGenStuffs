#!/usr/bin/env python3
import json
import os
import re
from collections import defaultdict
from rucio.client import Client

# Setup rucio environment
if "RUCIO_HOME" not in os.environ:
    os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/current"

def get_xrootd_sites_map():
    """Get RSE to xrootd mapping from SITECONF"""
    sites_xrootd_access = defaultdict(dict)
    
    sites = [(s, f"/cvmfs/cms.cern.ch/SITECONF/{s}/storage.json") 
             for s in os.listdir("/cvmfs/cms.cern.ch/SITECONF/") if s.startswith("T")]
    
    for site_name, conf in sites:
        if os.path.exists(conf):
            try:
                data = json.load(open(conf))
            except:
                continue  # Skip invalid JSON files
                
            for site in data:
                if site["type"] == "DISK" and site["rse"]:
                    for proc in site["protocols"]:
                        if (proc["protocol"] == "XRootD" and 
                            proc["access"] in ["global-ro", "global-rw"]):
                            if "prefix" in proc:
                                sites_xrootd_access[site["rse"]] = proc["prefix"]
                            elif "rules" in proc:
                                for rule in proc["rules"]:
                                    sites_xrootd_access[site["rse"]][rule["lfn"]] = rule["pfn"]
    return sites_xrootd_access

def get_pfn_for_site(path, rules):
    """Convert file path to full pfn using site rules"""
    if isinstance(rules, dict):
        for rule, pfn in rules.items():
            if m := re.match(rule, path):
                grs = m.groups()
                for i in range(len(grs)):
                    pfn = pfn.replace(f"${i+1}", grs[i])
                return pfn
    else:
        return rules + "/" + path.removeprefix("/")

def Get(inputs={}):

    client = Client()
    sites_xrootd_prefix = get_xrootd_sites_map()

    sites_regex = r"T[123]_(US|KR)_\w+" 
    output = {}
    for dataset_name, datasets in inputs.items():
        unique_paths = set()
        sites_counts = defaultdict(int)
        for dataset in datasets:
            for filedata in client.list_replicas([{"scope": "cms", "name": dataset}]):
                filename = filedata["name"]
                rses = filedata["rses"]
    
                for site in rses:
                    if re.search(sites_regex, site) and site in sites_xrootd_prefix:
                        full_url = get_pfn_for_site(filename, sites_xrootd_prefix[site])
                        if full_url:  # Only add if we got a valid URL
                            unique_paths.add(full_url)
                            sites_counts[site] += 1
                            break
        output[dataset_name] = {
            "files": sorted(unique_paths),
            "datasets": sorted(datasets)
        }
        with open(f"{dataset_name}.json", "w") as wf:
            json.dump(output[dataset_name], wf, indent=6)

    return output
